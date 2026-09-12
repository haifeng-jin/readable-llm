# readable-llm

A clean, educational, and end-to-end runnable implementation of a modern Large Language Model in a single pure Python file, with zero external dependencies.

**That file is [`readable_llm.py`](readable_llm.py).** Roughly 1,600 lines of standard library Python, importing nothing but `json`, `math`, `os`, and `random`. Read it and you have read the whole model.

This repository is the companion code for the article:
**[The Anatomy of an LLM: Every tensor operation explained with code and diagrams](https://haifengjin.com/the-anatomy-of-an-llm/)**.

## What This Project Is (and Isn't)

To set expectations clearly, here is what this project optimizes for and what it deliberately avoids:

- **We are not minimizing lines of code.**
- **We are not maximizing runtime performance.**
- **We are not explaining mathematical theory.**

**What we are optimizing for is readability for software engineers.**

If you are a developer who wants to understand how LLMs actually work on a compute level, this repo is for you. While our code does not reflect how computation happens inside a heavily optimized production engine, it is mathematically equivalent to it. Every matrix multiplication, attention score calculation, RoPE rotation, routing decision, and residual addition is laid bare using standard Python lists and arithmetic.

## Repository Layout

```
readable_llm.py        The model. This is the file to read.
test_all.py            Unit tests, including one that pins the parameter count.
training/
  train.py             Optional. Trains an equivalent PyTorch model.
  torch_model.py       Optional. The PyTorch mirror of readable_llm.py.
  export_weights.py    Optional. Dumps the trained weights to JSON.
  weights.json         The trained weights, already committed.
```

Everything under `training/` is how the weights were produced, and PyTorch is the only place in this repo that needs it. You can ignore that directory entirely and the model still runs.

## Code Organization

To make the code easy to navigate:
- **Classes organize weights**: Each layer class (such as `Embedding`, `GQA`, `MoE`, and `Model`) encapsulates learnable parameters and child modules in its immediate breakdown. Calling a class instance executes its `.predict()` method.
- **Standalone functions perform the compute**: All tensor transformations, matrix multiplications, normalizations, activations, and routing logic live in standalone pure functions.

When reading the code, look at classes to see how weights are structured, and follow standalone functions to see how the computation actually happens.

Every tensor is annotated with its shape, both in the docstrings and above each assignment. None of those shapes carry a batch dimension, because the model runs one sequence at a time.

## Reading Guide

`readable_llm.py` is split into banner-delimited sections, ordered from the smallest operations up to the full model:

| Section | What it covers |
| --- | --- |
| Vocabulary & Architecture Constants | The 12-token vocabulary and every size the model uses |
| Weight Initialization & Loading | Reading `weights.json`, with a seeded random fallback |
| Tokenizer | Greedy longest-match `encode` and `decode` |
| Base Neural Network Class | `Layer`, which names and holds weights |
| Basic Operations | `softmax`, `argmax`, `silu`, `matmul`, `add` |
| RMSNorm | Normalizing a token vector by its root mean square |
| RoPE | Rotating query and key pairs to encode position |
| Attention | Causal scaled dot-product attention |
| Grouped-Query Attention (GQA) | Query heads sharing one key head and one value head |
| Mixture of Experts (MoE) | The router and the SwiGLU experts it picks between |
| Decoder | Stacking GQA and MoE with residual connections |
| Embedding & LM Head | The token table and the tied output projection |
| Sampler | Greedy next-token selection |
| Model | Embedding, decoder, LM head, and the generation loop |
| Pipeline & Main | Text in, text out |

The article walks the same code in the opposite direction, starting from `pipeline` at the bottom of the file and drilling down. Either order works. Jump to `main()` at the end to follow along with the article, or start at the top to build the model up from arithmetic.

## Run the Demo

Run the end-to-end demonstration script:

```bash
python readable_llm.py
```

The script runs the prompt `"What is 1+1?"` and outputs:

```
What is 1+1? It's 2.<eos>
```

This tiny model was trained specifically on this single sentence, so it can only answer this one prompt. It automatically loads the pre-trained weights from `training/weights.json`. To run with random initialization instead, set `WEIGHTS_PATH = None` in `readable_llm.py`.

## Run Tests

Run the full unit test suite:

```bash
python test_all.py
```

## Train and Export Weights

The model itself needs nothing but a Python interpreter. Training is the one part that uses PyTorch, and it is entirely optional since the trained weights are already committed.

To re-train the 4,596-parameter (0.000005B) model from scratch and re-export the weights:

```bash
# 1. Train the PyTorch model (saves training/model.pt)
python training/train.py

# 2. Export weights to JSON with zero NumPy dependency (saves training/weights.json)
python training/export_weights.py
```

## License

[MIT](LICENSE)
