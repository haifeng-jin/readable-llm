# readable-llm

A clean, educational, and end-to-end runnable implementation of a modern Large Language Model in Python.

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

## Architecture & Tensor Shapes

The default reference configuration matches the article:

| Hyperparameter | Symbol | Value | Description |
|---|---|---|---|
| Vocabulary Size | `vocab_size` | ~2,000 | Number of vocabulary tokens |
| Sequence Length | `seq_len` | 6 | Tokens in prompt ("What is 1+1?") |
| Hidden Dimension | `hidden_size` | 12 | Width of the residual stream |
| Attention Groups | `num_groups` | 3 | GQA group count |
| Query Heads per Group | `q_heads` | 2 | Query projections sharing a KV head |
| Head Dimension | `d_head` | 2 | Dimension per attention head |
| Group Dimension | `head_dim` | 4 | `q_heads * d_head` (12 // 3) |
| Number of Experts | `num_experts` | 3 | Total expert networks per MoE block |
| Active Experts | `TOP_K` | 2 | Number of routed experts per token |
| Expert Intermediate Size | `inter_size` | 16 | SwiGLU hidden layer dimension |
| Decoder Blocks | `num_blocks` | 2 | Stacked transformer layers |

## Project Structure

```
readable-llm/
├── readable_llm/
│   ├── __init__.py      # Package exports
│   ├── tokenizer.py     # Vocabulary and Tokenizer
│   ├── ops.py           # Matmul, Add, SiLU, Softmax, Argmax, RMSNorm
│   ├── rope.py          # Rotary Position Embedding (2D and 3D)
│   ├── attention.py     # Causal attention token, single head, multi-head
│   ├── gqa.py           # Grouped-Query Attention blocks and groups
│   ├── moe.py           # Top-K router and SwiGLU expert layers
│   ├── decoder.py       # Decoder block and stack
│   ├── head.py          # Embedding lookup and LM head
│   ├── sampler.py       # Greedy autoregressive sampler
│   └── model.py         # Full Model class and generate loop
├── main.py              # End-to-end runnable demonstration
├── tests/
│   └── test_all.py      # Complete test suite
├── requirements.txt
└── pyproject.toml
```

## Quick Start

### Installation

Clone the repository and install requirements:

```bash
git clone https://github.com/haifeng-jin/readable-llm.git
cd readable-llm
pip install -r requirements.txt
```

### Run the Demo

Run the end-to-end demonstration script:

```bash
python3 main.py
```

### Run Tests

Run the full unit test suite:

```bash
python3 -m unittest discover tests
```
