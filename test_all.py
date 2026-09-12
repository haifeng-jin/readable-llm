"""Unit tests for readable_llm.

Covers three things:
1. Every layer builds with zero arguments and produces the expected tensor shapes.
2. The standalone math functions return the values we expect on small inputs.
3. The exported weights in training/weights.json line up with what the model asks
   for, add up to 4,596 parameters, and still generate the trained sentence.

Run with: python test_all.py
"""

import unittest
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
    init_weights,
    Tokenizer,
    Layer,
    vocab,
    split_tokens,
    matmul,
    add,
    silu,
    softmax,
    argmax,
    norm_token,
    rms_norm,
    RMSNorm,
    rope_pair,
    rope_token,
    rope_2d,
    rope_3d,
    rope,
    attention_token,
    dot_product_attention,
    attention_head,
    group_0,
    Group,
    gqa,
    GQA,
    out_matmul,
    OutMatmul,
    gqa_block,
    GQABlock,
    expert_token,
    expert,
    Expert,
    route_token,
    router,
    Router,
    moe_token,
    moe,
    MoE,
    moe_block,
    MoEBlock,
    decoder_block,
    DecoderBlock,
    decoder,
    Decoder,
    lookup,
    embedding,
    Embedding,
    logits_matmul,
    slice_last,
    lm_head,
    LMHead,
    greedy_sampler,
    Model,
    pipeline,
)


class TestReadableLLM(unittest.TestCase):
    def test_tokenizer(self):
        tokenizer = Tokenizer(vocab)
        text = "What is 1+1?"
        encoded = tokenizer.encode(text)
        self.assertEqual(encoded, [3, 4, 5, 6, 7, 8])
        # Leading spaces live inside the tokens, so decoding round-trips exactly
        decoded = tokenizer.decode(encoded)
        self.assertEqual(decoded, text)

        # Test zero-argument default init
        default_tokenizer = Tokenizer()
        self.assertEqual(default_tokenizer.encode(text), encoded)

    def test_layer_base_class(self):
        all_layer_classes = [
            RMSNorm,
            Group,
            GQA,
            OutMatmul,
            GQABlock,
            Expert,
            Router,
            MoE,
            MoEBlock,
            DecoderBlock,
            Decoder,
            Embedding,
            LMHead,
            Model,
        ]
        for cls in all_layer_classes:
            self.assertTrue(issubclass(cls, Layer))

    def test_zero_argument_initialization(self):
        # Verify classes initialize with zero arguments
        group = Group()
        self.assertEqual(len(group.w_q), Q_HEADS)
        self.assertEqual(len(group.w_k), HIDDEN_SIZE)
        self.assertEqual(len(group.w_v), HIDDEN_SIZE)

        gqa_layer = GQA()
        self.assertEqual(len(gqa_layer.groups), NUM_GROUPS)
        for g in gqa_layer.groups:
            self.assertIsInstance(g, Group)

        out_matmul_layer = OutMatmul()
        self.assertEqual(len(out_matmul_layer.w_matmul), HIDDEN_SIZE)
        self.assertEqual(len(out_matmul_layer.w_matmul[0]), HIDDEN_SIZE)

        rms_norm_layer = RMSNorm()
        self.assertEqual(len(rms_norm_layer.gamma), HIDDEN_SIZE)

        gqa_block_layer = GQABlock()
        self.assertIsInstance(gqa_block_layer.rms_norm, RMSNorm)
        self.assertIsInstance(gqa_block_layer.gqa, GQA)
        self.assertIsInstance(gqa_block_layer.out_matmul, OutMatmul)

        expert_layer = Expert()
        self.assertEqual(len(expert_layer.w_gate), HIDDEN_SIZE)
        self.assertEqual(len(expert_layer.w_up), HIDDEN_SIZE)
        self.assertEqual(len(expert_layer.w_down), INTER_SIZE)

        router_layer = Router()
        self.assertEqual(len(router_layer.w_router), HIDDEN_SIZE)
        self.assertEqual(len(router_layer.w_router[0]), NUM_EXPERTS)

        moe_layer = MoE()
        self.assertEqual(len(moe_layer.experts), NUM_EXPERTS)
        for exp in moe_layer.experts:
            self.assertIsInstance(exp, Expert)

        moe_block_layer = MoEBlock()
        self.assertIsInstance(moe_block_layer.rms_norm, RMSNorm)
        self.assertIsInstance(moe_block_layer.router, Router)
        self.assertIsInstance(moe_block_layer.moe, MoE)

        decoder_block_layer = DecoderBlock()
        self.assertIsInstance(decoder_block_layer.gqa_block_layer, GQABlock)
        self.assertIsInstance(decoder_block_layer.moe_block_layer, MoEBlock)

        decoder_layer = Decoder()
        self.assertEqual(len(decoder_layer.decoder_blocks), NUM_DECODER_BLOCKS)
        for db in decoder_layer.decoder_blocks:
            self.assertIsInstance(db, DecoderBlock)

        embedding_layer = Embedding()
        self.assertEqual(len(embedding_layer.embedding_table), VOCAB_SIZE)
        self.assertEqual(len(embedding_layer.embedding_table[0]), HIDDEN_SIZE)
        self.assertEqual(len(embedding_layer.embedding_table_T), HIDDEN_SIZE)
        self.assertEqual(len(embedding_layer.embedding_table_T[0]), VOCAB_SIZE)

        lm_head_layer = LMHead()
        self.assertEqual(len(lm_head_layer.gamma), HIDDEN_SIZE)
        self.assertEqual(len(lm_head_layer.embedding_table_T), HIDDEN_SIZE)
        self.assertEqual(len(lm_head_layer.embedding_table_T[0]), VOCAB_SIZE)

        model = Model()
        self.assertIsInstance(model.embedding, Embedding)
        self.assertIsInstance(model.decoder, Decoder)
        self.assertIsInstance(model.lm_head, LMHead)

    def test_layers_forward_pass(self):
        input_ids = [3, 4, 5]
        embedding_layer = Embedding()
        embed_out = embedding_layer(input_ids)
        self.assertEqual(len(embed_out), 3)
        self.assertEqual(len(embed_out[0]), HIDDEN_SIZE)

        decoder_layer = Decoder()
        decoder_out = decoder_layer(embed_out)
        self.assertEqual(len(decoder_out), 3)
        self.assertEqual(len(decoder_out[0]), HIDDEN_SIZE)

        lm_head_layer = LMHead(embedding_layer.embedding_table_T)
        logits = lm_head_layer(decoder_out)
        self.assertEqual(len(logits), VOCAB_SIZE)

    def test_init_weights_helper(self):
        # 1D vector (normalization scale)
        vec = init_weights(HIDDEN_SIZE)
        self.assertEqual(len(vec), HIDDEN_SIZE)
        self.assertEqual(vec, [1.0] * HIDDEN_SIZE)

        # 2D matrix
        mat = init_weights(HIDDEN_SIZE, D_HEAD)
        self.assertEqual(len(mat), HIDDEN_SIZE)
        self.assertEqual(len(mat[0]), D_HEAD)

        # 3D tensor (query projection across heads)
        tensor_3d = init_weights(Q_HEADS, HIDDEN_SIZE, D_HEAD)
        self.assertEqual(len(tensor_3d), Q_HEADS)
        self.assertEqual(len(tensor_3d[0]), HIDDEN_SIZE)
        self.assertEqual(len(tensor_3d[0][0]), D_HEAD)

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

        # rms_norm function and class
        v = [3.0, 4.0]
        gamma = [1.0, 1.0]
        normed = norm_token(v, gamma)
        self.assertEqual(len(normed), 2)
        norm_layer = RMSNorm()
        self.assertEqual(len(norm_layer(v)), len(v))

    def test_rope(self):
        x_2d = [[1.0, 2.0], [3.0, 4.0]]
        out_2d = rope(x_2d)
        self.assertEqual(len(out_2d), 2)
        self.assertEqual(len(out_2d[0]), 2)

        x_3d = [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]
        out_3d = rope(x_3d)
        self.assertEqual(len(out_3d), 2)
        self.assertEqual(len(out_3d[0]), 2)
        self.assertEqual(len(out_3d[0][0]), 2)

    def test_attention(self):
        seq_len = 3
        q_head = [[1.0, 0.0] for _ in range(seq_len)]
        k = [[1.0, 0.0] for _ in range(seq_len)]
        v = [[0.5, 0.5] for _ in range(seq_len)]

        single_out = dot_product_attention(q_head, k, v)
        self.assertEqual(len(single_out), seq_len)
        self.assertEqual(len(single_out[0]), D_HEAD)

        q = [q_head for _ in range(Q_HEADS)]
        attn_out = attention_head(q, k, v)
        self.assertEqual(len(attn_out), seq_len)
        self.assertEqual(len(attn_out[0]), HEAD_DIM)

    def test_gqa_and_moe(self):
        seq_len = 3
        dummy_in = [[0.1] * HIDDEN_SIZE for _ in range(seq_len)]

        # Group class
        group_layer = Group()
        g0_out = group_layer(dummy_in)
        self.assertEqual(len(g0_out), seq_len)
        self.assertEqual(len(g0_out[0]), HEAD_DIM)

        # GQA class
        gqa_layer = GQA()
        gqa_out = gqa_layer(dummy_in)
        self.assertEqual(len(gqa_out), seq_len)
        self.assertEqual(len(gqa_out[0]), HIDDEN_SIZE)

        # OutMatmul class
        out_matmul_layer = OutMatmul()
        gqa_block_out = out_matmul_layer(gqa_out)
        self.assertEqual(len(gqa_block_out), seq_len)
        self.assertEqual(len(gqa_block_out[0]), HIDDEN_SIZE)

        # Full GQABlock
        gqa_block_layer = GQABlock()
        block_out = gqa_block_layer(dummy_in)
        self.assertEqual(len(block_out), seq_len)
        self.assertEqual(len(block_out[0]), HIDDEN_SIZE)

        # Router class
        router_layer = Router()
        top_w = router_layer(dummy_in)
        self.assertEqual(len(top_w), seq_len)
        self.assertEqual(len(top_w[0]), NUM_EXPERTS)

        # Expert class
        expert_layer = Expert()
        single_expert_out = expert_layer(dummy_in[0])
        self.assertEqual(len(single_expert_out), HIDDEN_SIZE)

        # MoE class
        moe_layer = MoE()
        moe_out = moe_layer(dummy_in, top_w)
        self.assertEqual(len(moe_out), seq_len)
        self.assertEqual(len(moe_out[0]), HIDDEN_SIZE)

        # Full MoEBlock
        moe_block_layer = MoEBlock()
        block_moe_out = moe_block_layer(dummy_in)
        self.assertEqual(len(block_moe_out), seq_len)
        self.assertEqual(len(block_moe_out[0]), HIDDEN_SIZE)

    def test_model_end_to_end(self):
        # Model initializes with zero arguments
        model = Model()
        input_ids = [3, 4, 5, 6, 7, 8]

        # Test predict method and __call__
        logits = model.predict(input_ids)
        self.assertEqual(len(logits), VOCAB_SIZE)
        self.assertEqual(model(input_ids), logits)

        # Test autoregressive generate
        output_ids = model.generate(input_ids, max_new_tokens=4)
        self.assertEqual(len(output_ids), len(input_ids) + 4)

    def test_pipeline(self):
        output = pipeline("What is 1+1?", max_new_tokens=2)
        self.assertIsInstance(output, str)
        self.assertTrue(output.startswith("What is 1+1?"))

    def test_weights_json_loading(self):
        import json
        import os
        import readable_llm

        weights_path = os.path.join(os.path.dirname(__file__), "training", "weights.json")
        if not os.path.exists(weights_path):
            self.skipTest("training/weights.json not found")

        # 1. Collect all tensor names requested by readable_llm.Model
        requested_names = []
        orig_init = readable_llm.Layer.init_weights

        def tracking_init(layer_self, tensor_name, *shape):
            full_name = f"{layer_self.name}.{tensor_name}"
            requested_names.append(full_name)
            return orig_init(layer_self, tensor_name, *shape)

        readable_llm.Layer.init_weights = tracking_init
        try:
            readable_llm.Model()
        finally:
            readable_llm.Layer.init_weights = orig_init

        # 2. Check exact key match with weights.json
        with open(weights_path, "r", encoding="utf-8") as f:
            weights_data = json.load(f)

        self.assertEqual(set(requested_names), set(weights_data.keys()))

        # 3. Test loading via WEIGHTS_PATH
        old_path = readable_llm.WEIGHTS_PATH
        try:
            readable_llm.WEIGHTS_PATH = weights_path
            readable_llm.clear_weights_cache()
            loaded_model = readable_llm.Model()
            self.assertEqual(
                loaded_model.embedding.embedding_table,
                weights_data["embedding.embedding_table"],
            )

            # Test generation with loaded weights
            tok = readable_llm.Tokenizer()
            gen_out = readable_llm.pipeline("What is 1+1?", tokenizer=tok, model=loaded_model)
            self.assertEqual(gen_out, "What is 1+1? It's 2.<eos>")
        finally:
            readable_llm.WEIGHTS_PATH = old_path
            readable_llm.clear_weights_cache()

    def test_parameter_count(self):
        """The README and docstrings advertise 4,596 parameters. Verify it."""
        import json
        import os

        weights_path = os.path.join(os.path.dirname(__file__), "training", "weights.json")
        if not os.path.exists(weights_path):
            self.skipTest("training/weights.json not found")

        with open(weights_path, "r", encoding="utf-8") as f:
            weights_data = json.load(f)

        def count(tensor):
            if isinstance(tensor, list):
                return sum(count(item) for item in tensor)
            return 1

        total = sum(count(tensor) for tensor in weights_data.values())
        self.assertEqual(total, 4596)


if __name__ == "__main__":
    unittest.main()
