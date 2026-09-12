"""Train the PyTorch model on the single example sentence and save checkpoints.

This script trains the 4,596-parameter model on the single target sentence:
    "What is 1+1? It's 2.<eos>"
using AdamW with cross-entropy loss. Because training is performed exclusively
on this sentence, the resulting model specializes to answer only this single prompt.

Workflow:
1. Encodes target sentence into token IDs.
2. Sets input_ids = tokens[:-1] and target_ids = tokens[1:].
3. Trains for 101 epochs until loss drops below 0.01.
4. Verifies PyTorch autoregressive generation produces "Whatis 1+1? It's 2.<eos>".
5. Saves model weights to `training/model.pt`.
"""

import os
import sys
import warnings

# Ignore minor warnings if numpy is absent in pure-torch environments
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn.functional as F

from readable_llm import Tokenizer, vocab
from training.torch_model import TorchModel

CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "model.pt")


def generate(model, tokenizer, prompt, max_new_tokens=10):
    """Autoregressively generate next tokens using the PyTorch model."""
    token_ids = tokenizer.encode(prompt)
    curr = list(token_ids)
    for _ in range(max_new_tokens):
        inp = torch.tensor(curr, dtype=torch.long)
        logits = model(inp)[-1]
        next_id = logits.argmax().item()
        curr.append(next_id)
        if next_id == 2:  # <eos>
            break
    return tokenizer.decode(curr)


def train(epochs=101, lr=0.01, save_path=CHECKPOINT_PATH):
    tokenizer = Tokenizer(vocab)
    target_sentence = "What is 1+1? It's 2.<eos>"
    tokens = tokenizer.encode(target_sentence)

    print(f"Target sentence: {target_sentence}")
    print(f"Token IDs:       {tokens}")
    print(f"Tokens:          {[tokenizer.inv_vocab.get(t, f'<{t}>') for t in tokens]}")

    input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
    target_ids = torch.tensor(tokens[1:], dtype=torch.long)

    model = TorchModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    print("\n--- Training ---")
    for epoch in range(epochs):
        optimizer.zero_grad()
        logits = model(input_ids)
        loss = F.cross_entropy(logits, target_ids)
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:3d} | Loss: {loss.item():.4f}")

    # Verify PyTorch generation before saving
    prompt = "What is 1+1?"
    torch_out = generate(model, tokenizer, prompt)
    print("\n--- PyTorch Generation ---")
    print(f"Prompt:    '{prompt}'")
    print(f"Generated: '{torch_out}'")

    # Save PyTorch checkpoint
    torch.save(model.state_dict(), save_path)
    print(f"\nSaved PyTorch model weights to: {save_path}")


if __name__ == "__main__":
    train()
