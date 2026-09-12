# readable-llm

A clean, educational, and end-to-end runnable implementation of a modern Large Language Model in a single pure Python file, with zero external dependencies.

This repository is the companion code for the article:
**[The Anatomy of an LLM: Every tensor operation explained with code and diagrams](https://haifengjin.com/the-anatomy-of-an-llm/)**.

## What This Project Is (and Isn't)

To set expectations clearly, here is what this project optimizes for and what it deliberately avoids:

- **We are not minimizing lines of code.**
- **We are not maximizing runtime performance.**
- **We are not explaining mathematical theory.**

**What we are optimizing for is readability for software engineers.**

If you are a developer who wants to understand how LLMs actually work on a compute level, this repo is for you. While our code does not reflect how computation happens inside a heavily optimized production engine, it is mathematically equivalent to it. Every matrix multiplication, attention score calculation, RoPE rotation, routing decision, and residual addition is laid bare using standard Python lists and arithmetic.

## Code Organization

To make the code easy to navigate:
- **Classes organize weights**: Each layer class (such as `Embedding`, `GQA`, `MoE`, and `Model`) encapsulates learnable parameters and child modules in its immediate breakdown. Calling a class instance executes its `.predict()` method.
- **Standalone functions perform the compute**: All tensor transformations, matrix multiplications, normalizations, activations, and routing logic live in standalone pure functions.

When reading the code, look at classes to see how weights are structured, and follow standalone functions to see how the computation actually happens.

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
