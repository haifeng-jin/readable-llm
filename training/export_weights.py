"""Convert saved PyTorch model weights to plain Python formats (JSON).

Zero dependency on NumPy: uses pure PyTorch .tolist() and Python's built-in json module.
The resulting JSON file can be loaded directly in pure Python environments without PyTorch.
"""

import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

import readable_llm
from readable_llm import Tokenizer, vocab
from readable_llm import Model as PurePyModel
from readable_llm import pipeline
from training.torch_model import (
    TorchModel,
    export_torch_weights_to_dict,
)

CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "model.pt")
EXPORT_JSON_PATH = os.path.join(os.path.dirname(__file__), "weights.json")


def convert_and_export(pt_path=CHECKPOINT_PATH, json_path=EXPORT_JSON_PATH):
    """Loads PyTorch checkpoint and exports weights to plain JSON."""
    if not os.path.exists(pt_path):
        raise FileNotFoundError(
            f"PyTorch weights file '{pt_path}' not found. Run training/train.py first."
        )

    # 1. Load PyTorch model
    model = TorchModel()
    state_dict = torch.load(pt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    # 2. Export to plain Python dictionary matching readable_llm tensor names
    weights_dict = export_torch_weights_to_dict(model)

    # 3. Save as standard JSON file
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(weights_dict, f)

    print(f"Exported plain Python weights to JSON: {json_path}")
    print(f"File size: {os.path.getsize(json_path) / 1024:.1f} KB")

    # 4. Verify loading into pure Python readable_llm.Model via WEIGHTS_PATH
    print("\n--- Verifying Pure Python Model with Exported Weights ---")
    readable_llm.WEIGHTS_PATH = json_path
    readable_llm._loaded_weights_cache = {}
    py_model = PurePyModel()

    extended_vocab = dict(vocab)
    extended_vocab.update({" It's": 632, ".": 4})
    tokenizer = Tokenizer(extended_vocab)

    prompt = "What is 1+1?"
    output = pipeline(prompt, tokenizer=tokenizer, model=py_model)
    print(f"Prompt:    '{prompt}'")
    print(f"Generated: '{output}'")


if __name__ == "__main__":
    convert_and_export()
