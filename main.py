#!/usr/bin/env python3
"""End-to-end demonstration of readable-llm.

Walks through every step of the computation graph:
1. Tokenization: encodes input text to token IDs.
2. Embedding: maps token IDs to vectors.
3. Decoder Blocks:
   - GQA (Grouped-Query Attention with RoPE)
   - MoE (Mixture of Experts with top-k routing)
   - Residual additions & RMSNorm
4. LM Head: projects final representations to vocabulary logits.
5. Generation: autoregressively generates tokens using greedy sampling.
"""

from readable_llm import Tokenizer, Model, vocab

def main():
    print("=" * 60)
    print("readable-llm: The Anatomy of an LLM in Python")
    print("=" * 60)

    # Initialize tokenizer and model
    tokenizer = Tokenizer(vocab)
    vocab_size = max(vocab.values()) + 1

    model = Model(
        vocab_size=vocab_size,
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

    # Input prompt from the article
    prompt = "What is 1+1?"
    print(f"\n1. Input Text:\n   {prompt!r}")

    # Step 1: Tokenize
    input_ids = tokenizer.encode(prompt)
    print(f"\n2. Token IDs (seq_len={len(input_ids)}):\n   {input_ids}")

    # Step 2: Embedding lookup
    embed_out = model.embedding(input_ids)
    print(f"\n3. Embedding Output Shape:\n   [{len(embed_out)}, {len(embed_out[0])}] (seq_len, hidden_size)")

    # Step 3: Decoder pass
    decoder_out = model.decoder(embed_out)
    print(f"\n4. Decoder Output Shape:\n   [{len(decoder_out)}, {len(decoder_out[0])}] (seq_len, hidden_size)")

    # Step 4: LM Head & next-token logits
    logits = model.lm_head(decoder_out)
    print(f"\n5. LM Head Logits Length:\n   [{len(logits)}] (vocab_size)")

    # Step 5: Full generation
    print("\n6. Generating tokens autoregressively...")
    generated_ids = model.generate(input_ids, max_new_tokens=6)
    print(f"   Generated Token IDs:\n   {generated_ids}")

    decoded_text = tokenizer.decode(generated_ids)
    print(f"\n7. Decoded Output Text:\n   {decoded_text!r}")
    print("\nPipeline finished successfully!")

if __name__ == "__main__":
    main()
