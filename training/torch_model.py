"""PyTorch implementation of the readable-llm architecture.

Matches readable_llm.py layer-by-layer:
- RMSNorm
- GQA (Grouped-Query Attention with RoPE and causal mask)
- MoE (Router with top-k gating and SwiGLU experts)
- DecoderBlock (Pre-LN residual connections)
- TorchModel (Embedding tied with LMHead)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from readable_llm import (
    VOCAB_SIZE,
    HIDDEN_SIZE,
    NUM_DECODER_BLOCKS,
    NUM_GROUPS,
    Q_HEADS,
    D_HEAD,
    HEAD_DIM,
    NUM_EXPERTS,
    TOP_K,
    INTER_SIZE,
    EPS,
)


class TorchRMSNorm(nn.Module):
    """Root Mean Square Normalization without mean-centering or learnable bias."""

    def __init__(self, hidden_size=HIDDEN_SIZE, eps=EPS):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


class TorchGQABlock(nn.Module):
    """Grouped-Query Attention block with RoPE, causal mask, and output projection."""

    def __init__(
        self,
        hidden_size=HIDDEN_SIZE,
        num_groups=NUM_GROUPS,
        q_heads=Q_HEADS,
        d_head=D_HEAD,
    ):
        super().__init__()
        self.rms_norm = TorchRMSNorm(hidden_size)
        self.num_groups = num_groups
        self.q_heads = q_heads
        self.d_head = d_head

        # w_q: [num_groups, q_heads, hidden_size, d_head]
        self.w_q = nn.Parameter(
            torch.randn(num_groups, q_heads, hidden_size, d_head) * 0.02
        )
        # w_k, w_v: [num_groups, hidden_size, d_head]
        self.w_k = nn.Parameter(torch.randn(num_groups, hidden_size, d_head) * 0.02)
        self.w_v = nn.Parameter(torch.randn(num_groups, hidden_size, d_head) * 0.02)
        # w_out: [hidden_size, hidden_size]
        self.w_out = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.02)

    def _apply_rope(self, x):
        # x: [..., seq_len, d_head]
        seq_len = x.shape[-2]
        d_head = x.shape[-1]
        out = x.clone()
        for pos in range(seq_len):
            for i in range(0, d_head, 2):
                freq = 1.0 / (10000.0 ** (i / d_head))
                angle = pos * freq
                cos_val = math.cos(angle)
                sin_val = math.sin(angle)
                x0 = x[..., pos, i]
                x1 = x[..., pos, i + 1]
                out[..., pos, i] = x0 * cos_val - x1 * sin_val
                out[..., pos, i + 1] = x0 * sin_val + x1 * cos_val
        return out

    def forward(self, x):
        # x: [seq_len, hidden_size]
        seq_len = x.shape[0]
        normed = self.rms_norm(x)

        group_outs = []
        for g in range(self.num_groups):
            # q: [q_heads, seq_len, d_head]
            q = torch.einsum("sd,hde->hse", normed, self.w_q[g])
            # k: [seq_len, d_head]
            k = normed @ self.w_k[g]
            # v: [seq_len, d_head]
            v = normed @ self.w_v[g]

            q = self._apply_rope(q)
            k = self._apply_rope(k)

            # scores: [q_heads, seq_len, seq_len]
            scores = torch.einsum("hqe,ke->hqk", q, k) / math.sqrt(self.d_head)
            mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
            scores = scores.masked_fill(~mask, float("-inf"))
            weights = F.softmax(scores, dim=-1)

            # head_outs: [q_heads, seq_len, d_head] -> [seq_len, q_heads * d_head]
            head_outs = torch.einsum("hqk,ke->hqe", weights, v)
            group_out = head_outs.permute(1, 0, 2).reshape(
                seq_len, self.q_heads * self.d_head
            )
            group_outs.append(group_out)

        # gqa_out: [seq_len, hidden_size]
        gqa_out = torch.cat(group_outs, dim=-1)
        return gqa_out @ self.w_out


class TorchMoEBlock(nn.Module):
    """Mixture of Experts block with top-k router and SwiGLU experts."""

    def __init__(
        self,
        hidden_size=HIDDEN_SIZE,
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
        inter_size=INTER_SIZE,
    ):
        super().__init__()
        self.rms_norm = TorchRMSNorm(hidden_size)
        self.num_experts = num_experts
        self.top_k = top_k

        # Router
        self.w_router = nn.Parameter(
            torch.randn(hidden_size, num_experts) * 0.02
        )

        # SwiGLU experts
        self.w_gate = nn.Parameter(
            torch.randn(num_experts, hidden_size, inter_size) * 0.02
        )
        self.w_up = nn.Parameter(
            torch.randn(num_experts, hidden_size, inter_size) * 0.02
        )
        self.w_down = nn.Parameter(
            torch.randn(num_experts, inter_size, hidden_size) * 0.02
        )

    def forward(self, x):
        # x: [seq_len, hidden_size]
        normed = self.rms_norm(x)

        # Router logits and top-k selection
        logits = normed @ self.w_router  # [seq_len, num_experts]
        top_logits, top_indices = torch.topk(logits, k=self.top_k, dim=-1)
        top_probs = F.softmax(top_logits, dim=-1)  # [seq_len, top_k]

        top_weights = torch.zeros_like(logits)
        top_weights.scatter_(dim=-1, index=top_indices, src=top_probs)

        # SwiGLU computation for each expert
        x_gate = torch.einsum("sd,edi->sei", normed, self.w_gate)
        x_up = torch.einsum("sd,edi->sei", normed, self.w_up)
        x_act = F.silu(x_gate) * x_up
        expert_outs = torch.einsum("sei,eid->sed", x_act, self.w_down)

        # Weighted aggregation
        moe_out = (top_weights.unsqueeze(-1) * expert_outs).sum(dim=1)
        return moe_out


class TorchDecoderBlock(nn.Module):
    """Encapsulates one GQA block and one MoE block with Pre-LN residuals."""

    def __init__(self):
        super().__init__()
        self.gqa_block = TorchGQABlock()
        self.moe_block = TorchMoEBlock()

    def forward(self, x):
        h = x + self.gqa_block(x)
        out = h + self.moe_block(h)
        return out


class TorchModel(nn.Module):
    """Full Transformer model with tied embedding and LM head weights."""

    def __init__(
        self,
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_blocks=NUM_DECODER_BLOCKS,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        nn.init.uniform_(self.embedding.weight, -0.02, 0.02)
        self.decoder_blocks = nn.ModuleList(
            [TorchDecoderBlock() for _ in range(num_blocks)]
        )
        self.final_norm = TorchRMSNorm(hidden_size)

    def forward(self, input_ids):
        # input_ids: [seq_len]
        x = self.embedding(input_ids)
        for block in self.decoder_blocks:
            x = block(x)
        normed = self.final_norm(x)
        # Tied weights projection to logits: [seq_len, vocab_size]
        logits = normed @ self.embedding.weight.T
        return logits


def export_torch_weights_to_dict(torch_model):
    """Converts TorchModel parameters to a flat dictionary matching readable_llm tensor names."""
    with torch.no_grad():
        weights = {}
        weights["embedding.embedding_table"] = torch_model.embedding.weight.tolist()
        for i, t_blk in enumerate(torch_model.decoder_blocks):
            prefix = f"decoder.decoder_blocks.{i}"
            weights[f"{prefix}.gqa_block.rms_norm.gamma"] = t_blk.gqa_block.rms_norm.weight.tolist()
            for g in range(NUM_GROUPS):
                weights[f"{prefix}.gqa_block.gqa.groups.{g}.w_q"] = t_blk.gqa_block.w_q[g].tolist()
                weights[f"{prefix}.gqa_block.gqa.groups.{g}.w_k"] = t_blk.gqa_block.w_k[g].tolist()
                weights[f"{prefix}.gqa_block.gqa.groups.{g}.w_v"] = t_blk.gqa_block.w_v[g].tolist()
            weights[f"{prefix}.gqa_block.out_matmul.w_matmul"] = t_blk.gqa_block.w_out.tolist()
            weights[f"{prefix}.moe_block.rms_norm.gamma"] = t_blk.moe_block.rms_norm.weight.tolist()
            weights[f"{prefix}.moe_block.router.w_router"] = t_blk.moe_block.w_router.tolist()
            for e in range(NUM_EXPERTS):
                weights[f"{prefix}.moe_block.moe.experts.{e}.w_gate"] = t_blk.moe_block.w_gate[e].tolist()
                weights[f"{prefix}.moe_block.moe.experts.{e}.w_up"] = t_blk.moe_block.w_up[e].tolist()
                weights[f"{prefix}.moe_block.moe.experts.{e}.w_down"] = t_blk.moe_block.w_down[e].tolist()
        weights["lm_head.gamma"] = torch_model.final_norm.weight.tolist()
        return weights


def load_weights_into_readable_model(py_model, weights_dict):
    """Loads flat weights dictionary into a readable_llm.Model instance."""
    # 1. Embedding & tied LM head
    py_model.embedding.embedding_table = weights_dict["embedding.embedding_table"]
    vocab_size = len(py_model.embedding.embedding_table)
    hidden_size = len(py_model.embedding.embedding_table[0])
    py_model.embedding.embedding_table_T = [
        [py_model.embedding.embedding_table[r][c] for r in range(vocab_size)]
        for c in range(hidden_size)
    ]
    py_model.lm_head.embedding_table_T = py_model.embedding.embedding_table_T
    py_model.lm_head.gamma = weights_dict["lm_head.gamma"]

    # 2. Decoder blocks
    for l_idx, blk in enumerate(py_model.decoder.decoder_blocks):
        prefix = f"decoder.decoder_blocks.{l_idx}"
        blk.gqa_block_layer.rms_norm.gamma = weights_dict[f"{prefix}.gqa_block.rms_norm.gamma"]
        blk.gqa_block_layer.out_matmul.w_matmul = weights_dict[f"{prefix}.gqa_block.out_matmul.w_matmul"]
        for g_idx, grp in enumerate(blk.gqa_block_layer.gqa.groups):
            grp.w_q = weights_dict[f"{prefix}.gqa_block.gqa.groups.{g_idx}.w_q"]
            grp.w_k = weights_dict[f"{prefix}.gqa_block.gqa.groups.{g_idx}.w_k"]
            grp.w_v = weights_dict[f"{prefix}.gqa_block.gqa.groups.{g_idx}.w_v"]

        blk.moe_block_layer.rms_norm.gamma = weights_dict[f"{prefix}.moe_block.rms_norm.gamma"]
        blk.moe_block_layer.router.w_router = weights_dict[f"{prefix}.moe_block.router.w_router"]
        for e_idx, exp in enumerate(blk.moe_block_layer.moe.experts):
            exp.w_gate = weights_dict[f"{prefix}.moe_block.moe.experts.{e_idx}.w_gate"]
            exp.w_up = weights_dict[f"{prefix}.moe_block.moe.experts.{e_idx}.w_up"]
            exp.w_down = weights_dict[f"{prefix}.moe_block.moe.experts.{e_idx}.w_down"]
