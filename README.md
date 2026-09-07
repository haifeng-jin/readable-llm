# readable-llm

A clean, educational, and end-to-end runnable implementation of a modern Large Language Model in a single pure Python file, with zero external dependencies.

This repository implements every tensor operation described in the article **"The Anatomy of an LLM: Every tensor operation explained with code and diagrams"**, covering:

- **Tokenizer**: Subword splitting, integer encoding, and text decoding.
- **Embedding Layer**: Token ID vector lookup.
- **Root Mean Square Normalization (RMSNorm)**: Pre-normalization along feature dimensions.
- **Grouped-Query Attention (GQA)**:
  - Multi-group query heads with shared key and value projections.
  - Rotary Position Embedding (RoPE) applied token-wise.
  - Scaled dot-product attention with causal future-token masking.
  - Output linear projection.
- **Mixture of Experts (MoE)**:
  - Learned routing gating with top-k softmax selection.
  - SwiGLU feed-forward expert subnetworks (gate, up, silu, and down projections).
  - Sparse weighted output combination.
- **Decoder Blocks & Residual Connections**: Pre-LN residual stream chaining.
- **Language Model Head (LM Head)**: Final RMSNorm, tied transposed embedding projection, and last-token slicing.
- **Greedy Generation**: Autoregressive next-token prediction loop.

## Architecture Constants

All architectural sizes and hyperparameters are defined as explicit constants directly in the code:

| Constant | Value | Description |
|---|---|---|
| `VOCAB_SIZE` | 2,000 | Number of vocabulary tokens |
| `HIDDEN_SIZE` | 12 | Width of the residual stream |
| `NUM_DECODER_BLOCKS` | 2 | Stacked transformer layers |
| `NUM_GROUPS` | 3 | GQA group count |
| `Q_HEADS` | 2 | Query projections sharing a KV head |
| `D_HEAD` | 2 | Dimension per attention head |
| `HEAD_DIM` | 4 | `Q_HEADS * D_HEAD` (`HIDDEN_SIZE // NUM_GROUPS`) |
| `NUM_EXPERTS` | 3 | Total expert networks per MoE block |
| `TOP_K` | 2 | Number of routed experts per token |
| `INTER_SIZE` | 16 | SwiGLU hidden layer dimension |
| `MAX_NEW_TOKENS` | 128 | Maximum autoregressive generation steps |
| `EOS_TOKEN_ID` | 2 | End-of-sequence token ID |
| `EPS` | 1e-6 | Numerical stability epsilon for RMSNorm |

## Layer Hierarchy & Forward Pass

All neural network modules inherit from the common base class `Layer`:

- `Layer`: Defines the forward pass contract `.predict()`. Calling an instance directly invokes `.predict()`.
- `Group`: Forward pass projects inputs into Q, K, V, applies RoPE, and computes multi-head attention.
- `GQABlock`: Forward pass applies RMSNorm, runs all groups, concatenates results, and projects output.
- `Expert`: Forward pass executes the SwiGLU feed-forward network (`w_gate`, `w_up`, `silu`, `w_down`).
- `MoEBlock`: Forward pass applies RMSNorm, runs the router, and computes sparse expert outputs.
- `DecoderBlock`: Forward pass executes GQA and MoE with residual additions.
- `Model`: Top-level model executing embedding, decoder stack, and LM head.

## Project Structure

All production code is contained in a single standalone file:

```
readable-llm/
├── readable_llm.py   # Complete standalone production implementation
├── main.py           # Entrypoint demonstration
├── tests/
│   └── test_all.py   # Unit test suite
├── requirements.txt  # Zero external dependencies
├── pyproject.toml
└── README.md
```

## Quick Start

No third-party packages or C extensions are required. You only need Python 3.9+:

```bash
git clone https://github.com/haifeng-jin/readable-llm.git
cd readable-llm
```

### Run the Demo

Run the end-to-end demonstration script:

```bash
python3 readable_llm.py
```

or:

```bash
python3 main.py
```

### Run Tests

Run the full unit test suite:

```bash
python3 -m unittest discover tests
```
