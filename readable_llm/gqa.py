from readable_llm.ops import matmul, rms_norm
from readable_llm.rope import rope
from readable_llm.attention import attention_head

def group_0(rms_out, w_q, w_k, w_v):
    # rms_out: [seq_len, hidden_size]
    # w_q: [q_heads, hidden_size, d_head]
    # w_k: [hidden_size, d_head]
    # w_v: [hidden_size, d_head]

    # Project inputs to q, k, v token-wise using our matmul helper
    # q: [q_heads, seq_len, d_head]
    # k: [seq_len, d_head]
    # v: [seq_len, d_head]
    q = [[matmul(token_vec, w) for token_vec in rms_out] for w in w_q]
    k = [matmul(token_vec, w_k) for token_vec in rms_out]
    v = [matmul(token_vec, w_v) for token_vec in rms_out]

    # Encode positions with rotary embedding
    # rope is token-wise and preserves tensor shape
    q = rope(q)
    k = rope(k)

    # Compute multi-head attention and concatenate heads
    # group_0_out: [seq_len, head_dim]
    group_0_out = attention_head(q, k, v)
    return group_0_out

class Group:
    def __init__(self, w_q, w_k, w_v):
        self.w_q = w_q
        self.w_k = w_k
        self.w_v = w_v

    def __call__(self, rms_out):
        return group_0(rms_out, self.w_q, self.w_k, self.w_v)

def gqa(rms_out, groups):
    # rms_out: [seq_len, hidden_size]
    seq_len = len(rms_out)

    # Each group computes attention on the full rms_out input
    # group_out: [seq_len, head_dim]
    group_outs = [group(rms_out) for group in groups]

    # Concatenate group outputs back along the column dimension
    # gqa_out: [seq_len, hidden_size]
    gqa_out = []
    # Iterate over the rows of gqa_out
    for t in range(seq_len):
        row = []
        # Iterate the groups to concat the outputs
        for out in group_outs:
            row.extend(out[t])
        gqa_out.append(row)
    return gqa_out

def out_matmul(gqa_out, w_matmul):
    # gqa_out: [seq_len, hidden_size]
    # w_matmul: [hidden_size, hidden_size]

    # gqa_block_out: [seq_len, hidden_size]
    gqa_block_out = [matmul(token_vec, w_matmul) for token_vec in gqa_out]
    return gqa_block_out

def gqa_block(gqa_block_in, rms_norm, gqa, out_matmul):
    # gqa_block_in: [seq_len, hidden_size]

    # rms_out: [seq_len, hidden_size]
    rms_out = rms_norm(gqa_block_in)

    # gqa_out: [seq_len, hidden_size]
    gqa_out = gqa(rms_out)

    # gqa_block_out: [seq_len, hidden_size]
    gqa_block_out = out_matmul(gqa_out)
    return gqa_block_out

class GQABlock:
    def __init__(self, gamma, groups, w_matmul):
        self.gamma = gamma
        self.groups = groups
        self.w_matmul = w_matmul

    def __call__(self, gqa_block_in):
        def norm_fn(t):
            return rms_norm(t, self.gamma)
        def gqa_fn(r):
            return gqa(r, self.groups)
        def out_fn(g):
            return out_matmul(g, self.w_matmul)
        return gqa_block(gqa_block_in, norm_fn, gqa_fn, out_fn)
