# readable-llm

A clean, educational, and end-to-end runnable implementation of a modern Large Language Model in a single pure Python file, with zero external dependencies.

## What This Project Is (and Isn't)

To set expectations clearly, here is what this project optimizes for and what it deliberately avoids:

- **We are not minimizing lines of code.** This is not a code golf exercise. We favor clear, explicit variable names and step-by-step logic over dense one-liners or clever abstractions.
- **We are not maximizing runtime performance.** You will not find GPU kernels, C extensions, multithreading, or low-level vectorization here. Everything runs in standard, pure Python.
- **We are not modeling production inference engines.** Real-world inference systems rely on complex memory paging, continuous batching, and custom hardware kernels. We skip those optimizations to keep the core algorithms transparent and accessible.
- **We are not explaining mathematical proofs or training theory.** We do not focus on why certain loss functions converge or the mathematical theory behind neural network training.

**What we are optimizing for is readability for software engineers.**

If you are a developer who wants to understand how LLMs actually work on a compute level, this repo is for you. While our code does not reflect how computation happens inside a heavily optimized production engine, it is mathematically equivalent to it. Every matrix multiplication, attention score calculation, RoPE rotation, routing decision, and residual addition is laid bare using standard Python lists and arithmetic. You can step through any line with a debugger, inspect shapes at every intermediate step, and see the exact mechanics of a modern transformer without getting lost in framework magic.

## Overview

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
