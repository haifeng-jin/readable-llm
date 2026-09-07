import random
from readable_llm.head import embedding, lm_head
from readable_llm.decoder import decoder, DecoderBlock
from readable_llm.gqa import Group, GQABlock
from readable_llm.moe import Expert, MoEBlock
from readable_llm.sampler import greedy_sampler

MAX_NEW_TOKENS = 128
EOS_TOKEN_ID = 2

def _make_matrix(rows, cols, scale=0.02, rng=None):
    if rng is None:
        rng = random.Random(42)
    return [[rng.uniform(-scale, scale) for _ in range(cols)] for _ in range(rows)]

class Model:
    def __init__(
        self,
        vocab_size=2000,
        hidden_size=12,
        num_decoder_blocks=2,
        num_groups=3,
        q_heads=2,
        d_head=2,
        num_experts=3,
        top_k=2,
        inter_size=16,
        seed=42,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_decoder_blocks = num_decoder_blocks
        self.num_groups = num_groups
        self.q_heads = q_heads
        self.d_head = d_head
        self.head_dim = q_heads * d_head
        self.num_experts = num_experts
        self.top_k = top_k
        self.inter_size = inter_size

        rng = random.Random(seed)

        # Embedding table: [vocab_size, hidden_size]
        self.embedding_table = _make_matrix(vocab_size, hidden_size, rng=rng)

        # Tied embeddings: transpose of embedding table for lm_head: [hidden_size, vocab_size]
        self.embedding_table_T = [
            [self.embedding_table[r][c] for r in range(vocab_size)]
            for c in range(hidden_size)
        ]
        self.lm_head_gamma = [1.0] * hidden_size

        # Stack of decoder blocks
        self.decoder_blocks = []
        for _ in range(num_decoder_blocks):
            # GQA block
            gqa_gamma = [1.0] * hidden_size
            groups = []
            for _ in range(num_groups):
                w_q = [_make_matrix(hidden_size, d_head, rng=rng) for _ in range(q_heads)]
                w_k = _make_matrix(hidden_size, d_head, rng=rng)
                w_v = _make_matrix(hidden_size, d_head, rng=rng)
                groups.append(Group(w_q, w_k, w_v))
            w_matmul = _make_matrix(hidden_size, hidden_size, rng=rng)
            gqa_layer = GQABlock(gqa_gamma, groups, w_matmul)

            # MoE block
            moe_gamma = [1.0] * hidden_size
            w_router = _make_matrix(hidden_size, num_experts, rng=rng)
            experts = []
            for _ in range(num_experts):
                w_gate = _make_matrix(hidden_size, inter_size, rng=rng)
                w_up = _make_matrix(hidden_size, inter_size, rng=rng)
                w_down = _make_matrix(inter_size, hidden_size, rng=rng)
                experts.append(Expert(w_gate, w_up, w_down))
            moe_layer = MoEBlock(moe_gamma, w_router, experts)

            self.decoder_blocks.append(DecoderBlock(gqa_layer, moe_layer))

    def embedding(self, input_ids):
        # input_ids: [seq_len]
        # embed_out: [seq_len, hidden_size]
        return embedding(input_ids, self.embedding_table)

    def decoder(self, embed_out):
        # embed_out: [seq_len, hidden_size]
        # decoder_out: [seq_len, hidden_size]
        return decoder(embed_out, self.decoder_blocks)

    def lm_head(self, decoder_out):
        # decoder_out: [seq_len, hidden_size]
        # logits: [vocab_size]
        return lm_head(decoder_out, self.lm_head_gamma, self.embedding_table_T)

    def predict(self, input_ids):
        # input_ids: [seq_len]
        # embed_out: [seq_len, hidden_size]
        embed_out = self.embedding(input_ids)
        # decoder_out: [seq_len, hidden_size]
        decoder_out = self.decoder(embed_out)
        # logits: [vocab_size]
        logits = self.lm_head(decoder_out)
        return logits

    def generate(self, input_ids, max_new_tokens=MAX_NEW_TOKENS):
        # input_ids: [seq_len]
        for _ in range(max_new_tokens):
            # next_token_id: int
            next_token_id = greedy_sampler(self, input_ids)
            # input_ids: [seq_len + 1]
            input_ids = input_ids + [next_token_id]
            if next_token_id == EOS_TOKEN_ID:
                break
        # input_ids: [total_seq_len]
        return input_ids
