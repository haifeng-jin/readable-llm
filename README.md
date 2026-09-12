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

### Run Tests

Run the full unit test suite:

```bash
python test_all.py
```

## Training and Exporting Weights

While `readable_llm.py` runs with randomly initialized weights out of the box with zero external dependencies, the `training/` directory contains an equivalent PyTorch model to train this 4,596-parameter (0.000005B) model on the example sentence (`"What is 1+1? It's 2.<eos>"`).

### 1. Train the PyTorch model

Install PyTorch (`pip install torch`) and run:

```bash
python training/train.py
```

This trains for 100 epochs until loss drops below 0.01 and saves the checkpoint to `training/model.pt`.

### 2. Export weights to plain JSON

Convert the PyTorch checkpoint to a lightweight JSON file with zero NumPy dependency:

```bash
python training/export_weights.py
```

This generates `training/weights.json` (~100 KB) and verifies that `readable_llm.py` can load every tensor and generate the target output.

### 3. Load trained weights in pure Python

To run `readable_llm.py` with the exported weights, set `WEIGHTS_PATH` in `readable_llm.py`:

```python
WEIGHTS_PATH = "training/weights.json"
```

Then run:

```bash
python readable_llm.py
```

The model will automatically load each layer's weights from the JSON file and answer:

```
Whatis 1+1? It's 2.<eos>
```
