"""Convert saved PyTorch model weights to plain Python JSON format.

Zero dependency on NumPy:
Uses pure PyTorch `.tolist()` and Python's built-in `json` module.
The resulting JSON file contains plain Python nested lists of floats and can
be loaded directly in pure Python environments with zero external libraries.

Key Alignment:
The exported JSON dictionary is flat and keyed by the exact hierarchical dot-separated
names requested by `readable_llm.Layer.init_weights`:
    - "embedding.embedding_table"
    - "decoder.decoder_blocks.{i}.gqa_block.rms_norm.gamma"
    - "decoder.decoder_blocks.{i}.gqa_block.gqa.groups.{g}.w_q"
    - "decoder.decoder_blocks.{i}.gqa_block.out_matmul.w_matmul"
    - "decoder.decoder_blocks.{i}.moe_block.rms_norm.gamma"
    - "decoder.decoder_blocks.{i}.moe_block.router.w_router"
    - "decoder.decoder_blocks.{i}.moe_block.moe.experts.{e}.w_gate"
    - "lm_head.gamma"

When `readable_llm.WEIGHTS_PATH` points to this file, `readable_llm.Model()`
automatically loads each tensor during layer initialization.
"""

import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

import readable_llm
from readable_llm import Model as PurePyModel
from readable_llm import Tokenizer, pipeline
from training.torch_model import TorchModel, export_torch_weights_to_dict

CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "model.pt")
EXPORT_JSON_PATH = os.path.join(os.path.dirname(__file__), "weights.json")


def convert_and_export(pt_path=CHECKPOINT_PATH, json_path=EXPORT_JSON_PATH):
    """Load the PyTorch checkpoint, write weights.json, and check the result.

    Args:
        pt_path: str, the checkpoint written by train.py.
        json_path: str, where to write the plain JSON weights.
    """
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

    # 4. Verify the pure Python model picks the weights up and says the same
    #    thing the PyTorch model did. Point WEIGHTS_PATH at the file we just
    #    wrote and clear the cache, in case an earlier file was already read.
    print("\n--- Verifying Pure Python Model with Exported Weights ---")
    readable_llm.WEIGHTS_PATH = json_path
    readable_llm.clear_weights_cache()
    py_model = PurePyModel()

    tokenizer = Tokenizer()

    prompt = "What is 1+1?"
    output = pipeline(prompt, tokenizer=tokenizer, model=py_model)
    print(f"Prompt:    '{prompt}'")
    print(f"Generated: '{output}'")


if __name__ == "__main__":
    convert_and_export()
