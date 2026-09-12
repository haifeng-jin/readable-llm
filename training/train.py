"""Train the PyTorch model on the single example sentence and save a checkpoint.

This script trains the 4,596-parameter model on one sentence:
    "What is 1+1? It's 2.<eos>"
using AdamW with cross-entropy loss. Training on a single sentence is
deliberate. The point is a set of real weights small enough to ship as JSON,
not a model that generalizes. Memorizing one sentence is all we need, and it
is also all a model this size can do.

Workflow:
1. Encode the target sentence into token IDs.
2. Set input_ids = tokens[:-1] and target_ids = tokens[1:], so every position
   learns to predict the token that follows it.
3. Train for 101 epochs, printing the loss every 20.
4. Print an autoregressive sample, which should read
   "What is 1+1? It's 2.<eos>".
5. Save the weights to `training/model.pt`.

Run `training/export_weights.py` afterwards to turn the checkpoint into the
plain JSON that `readable_llm.py` reads.
"""

import os
import sys
import warnings

# Ignore minor warnings if numpy is absent in pure-torch environments
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn.functional as F

from readable_llm import EOS_TOKEN_ID, Tokenizer, vocab
from training.torch_model import TorchModel

CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "model.pt")


def generate(model, tokenizer, prompt, max_new_tokens=10):
    """Autoregressively generate next tokens using the PyTorch model.

    Mirrors `readable_llm.Model.generate`: predict, append, repeat until the
    model emits <eos>. Used here only to eyeball the result before saving.
    """
    token_ids = tokenizer.encode(prompt)
    curr = list(token_ids)
    for _ in range(max_new_tokens):
        inp = torch.tensor(curr, dtype=torch.long)
        # The model returns logits for every position; we only want the last one
        logits = model(inp)[-1]
        next_id = logits.argmax().item()
        curr.append(next_id)
        if next_id == EOS_TOKEN_ID:
            break
    return tokenizer.decode(curr)


def train(epochs=101, lr=0.01, save_path=CHECKPOINT_PATH):
    """Fit the model to the single target sentence and write the checkpoint.

    Args:
        epochs: int, number of full-batch gradient steps. The whole dataset is
            one sentence, so an epoch is a single step.
        lr: float, AdamW learning rate. High by normal standards, which is fine
            when the goal is to overfit 4,596 parameters to one sentence.
        save_path: str, where to write the PyTorch state dict.
    """
    tokenizer = Tokenizer(vocab)
    target_sentence = "What is 1+1? It's 2.<eos>"
    tokens = tokenizer.encode(target_sentence)

    print(f"Target sentence: {target_sentence}")
    print(f"Token IDs:       {tokens}")
    print(f"Tokens:          {[tokenizer.inv_vocab.get(t, f'<{t}>') for t in tokens]}")

    # Next-token prediction: position i in input_ids should predict position i
    # in target_ids, which is the token right after it in the sentence.
    input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
    target_ids = torch.tensor(tokens[1:], dtype=torch.long)

    model = TorchModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    print("\n--- Training ---")
    for epoch in range(epochs):
        optimizer.zero_grad()
        # logits: [seq_len, vocab_size], one prediction per position
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
