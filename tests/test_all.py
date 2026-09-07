import unittest
import math
from readable_llm import (
    Tokenizer,
    vocab,
    split_tokens,
    matmul,
    add,
    silu,
    softmax,
    argmax,
    norm_token,
    rms_norm,
    rope_pair,
    rope_token,
    rope_2d,
    rope_3d,
    rope,
    attention_token,
    single_attention_head,
    attention_head,
    group_0,
    Group,
    gqa,
    out_matmul,
    gqa_block,
    GQABlock,
    expert_token,
    expert,
    Expert,
    route_token,
    router,
    moe_token,
    moe,
    moe_block,
    MoEBlock,
    decoder_block,
    DecoderBlock,
    decoder,
    lookup,
    embedding,
    matmul_token,
    logits_matmul,
    slice_last,
    lm_head,
    greedy_sampler,
    Model,
)

class TestReadableLLM(unittest.TestCase):
    def test_tokenizer(self):
        tokenizer = Tokenizer(vocab)
        text = "What is 1+1?"
        encoded = tokenizer.encode(text)
        self.assertEqual(encoded, [1867, 318, 352, 10, 16, 30])
        decoded = tokenizer.decode(encoded)
        self.assertEqual(decoded, "Whatis 1+1?")

    def test_ops(self):
        # matmul
        vec = [1.0, 2.0]
        mat = [[1.0, 0.0], [0.0, 1.0]]
        out = matmul(vec, mat)
        self.assertEqual(out, [1.0, 2.0])

        # add
        a = [[1.0, 2.0], [3.0, 4.0]]
        b = [[5.0, 6.0], [7.0, 8.0]]
        self.assertEqual(add(a, b), [[6.0, 8.0], [10.0, 12.0]])

        # silu
        self.assertAlmostEqual(silu(0.0), 0.0)

        # softmax
        probs = softmax([1.0, 1.0])
        self.assertAlmostEqual(probs[0], 0.5)
        self.assertAlmostEqual(probs[1], 0.5)

        # argmax
        self.assertEqual(argmax([0.1, 0.9, 0.4]), 1)

        # rms_norm
        v = [3.0, 4.0]
        gamma = [1.0, 1.0]
        normed = norm_token(v, gamma)
        # rms is sqrt((9+16)/2) = sqrt(12.5) ~= 3.5355
        self.assertEqual(len(normed), 2)

    def test_rope(self):
        # Test 2D
        x_2d = [[1.0, 2.0], [3.0, 4.0]]
        out_2d = rope(x_2d)
        self.assertEqual(len(out_2d), 2)
        self.assertEqual(len(out_2d[0]), 2)

        # Test 3D
        x_3d = [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]
        out_3d = rope(x_3d)
        self.assertEqual(len(out_3d), 2)
        self.assertEqual(len(out_3d[0]), 2)
        self.assertEqual(len(out_3d[0][0]), 2)

    def test_attention(self):
        seq_len = 3
        q_heads = 2
        d_head = 2
        head_dim = q_heads * d_head

        single_q = [[1.0, 0.0] for _ in range(seq_len)]
        k = [[1.0, 0.0] for _ in range(seq_len)]
        v = [[0.5, 0.5] for _ in range(seq_len)]

        # single_attention_head
        single_out = single_attention_head(single_q, k, v)
        self.assertEqual(len(single_out), seq_len)
        self.assertEqual(len(single_out[0]), d_head)

        # multi-head attention
        q = [single_q for _ in range(q_heads)]
        attn_out = attention_head(q, k, v)
        self.assertEqual(len(attn_out), seq_len)
        self.assertEqual(len(attn_out[0]), head_dim)

    def test_gqa_and_moe(self):
        seq_len = 3
        hidden_size = 12
        q_heads = 2
        d_head = 2
        head_dim = q_heads * d_head
        num_groups = 3
        num_experts = 3
        inter_size = 16

        dummy_in = [[0.1] * hidden_size for _ in range(seq_len)]

        # Group 0
        w_q = [[[0.01] * d_head for _ in range(hidden_size)] for _ in range(q_heads)]
        w_k = [[0.01] * d_head for _ in range(hidden_size)]
        w_v = [[0.01] * d_head for _ in range(hidden_size)]
        g0_out = group_0(dummy_in, w_q, w_k, w_v)
        self.assertEqual(len(g0_out), seq_len)
        self.assertEqual(len(g0_out[0]), head_dim)

        # GQA with 3 groups -> hidden_size = 12
        groups = [Group(w_q, w_k, w_v) for _ in range(num_groups)]
        gqa_out = gqa(dummy_in, groups)
        self.assertEqual(len(gqa_out), seq_len)
        self.assertEqual(len(gqa_out[0]), hidden_size)

        # Out matmul
        w_matmul = [[0.01] * hidden_size for _ in range(hidden_size)]
        gqa_block_out = out_matmul(gqa_out, w_matmul)
        self.assertEqual(len(gqa_block_out), seq_len)
        self.assertEqual(len(gqa_block_out[0]), hidden_size)

        # MoE
        w_router = [[0.01] * num_experts for _ in range(hidden_size)]
        top_w = router(dummy_in, w_router)
        self.assertEqual(len(top_w), seq_len)
        self.assertEqual(len(top_w[0]), num_experts)

        w_gate = [[0.01] * inter_size for _ in range(hidden_size)]
        w_up = [[0.01] * inter_size for _ in range(hidden_size)]
        w_down = [[0.01] * hidden_size for _ in range(inter_size)]
        experts = [Expert(w_gate, w_up, w_down) for _ in range(num_experts)]
        moe_out = moe(dummy_in, top_w, experts)
        self.assertEqual(len(moe_out), seq_len)
        self.assertEqual(len(moe_out[0]), hidden_size)

    def test_model_end_to_end(self):
        model = Model(
            vocab_size=1868,
            hidden_size=12,
            num_decoder_blocks=2,
            num_groups=3,
            q_heads=2,
            d_head=2,
            num_experts=3,
            top_k=2,
            inter_size=16,
            seed=42,
        )
        input_ids = [1867, 318, 352, 10, 16, 30]
        logits = model.predict(input_ids)
        self.assertEqual(len(logits), 1868)

        # Test generate
        output_ids = model.generate(input_ids, max_new_tokens=4)
        self.assertEqual(len(output_ids), len(input_ids) + 4)

if __name__ == "__main__":
    unittest.main()
