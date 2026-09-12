"""readable-llm: The Anatomy of an LLM in a single pure Python file.

Companion code for the article "The Anatomy of an LLM":
https://haifengjin.com/the-anatomy-of-an-llm/

Overview:
---------
This repository provides a clean, educational, and end-to-end runnable
implementation of a modern Large Language Model using only standard library
Python. There are zero external dependencies: no PyTorch, no NumPy, and no
C extensions. Every tensor operation is laid bare using standard Python lists,
loops, and basic math.

Model Scale (0.000005B):
------------------------
This model contains exactly 4,596 parameters (approx. 0.000005B, or 4.6e-6B):
- Embedding table: 12 tokens * 12 hidden size = 144 parameters
- 2 Decoder blocks: 2 * 2,220 = 4,440 parameters
  - Each block has a Grouped-Query Attention (GQA) block (444 params)
    and a Mixture of Experts (MoE) block with 3 SwiGLU experts (1,776 params)
- Final LM head RMSNorm scale: 12 parameters
(The LM head output projection matrix is tied to the embedding table, so it
shares those 144 parameters rather than introducing new weights.)

Single Prompt Capability:
-------------------------
Because this tiny model was trained exclusively on the single sentence:
    "What is 1+1? It's 2.<eos>"
it can only answer this one prompt. Running pipeline("What is 1+1?") produces:
    "What is 1+1? It's 2.<eos>"
For any other prompt, the model will output nonsensical tokens because it
has not been trained on any other data.

Training & Weight Export:
-------------------------
The model weights were trained using an equivalent PyTorch model in
`training/train.py` for 101 epochs using AdamW, then exported to plain JSON
(`training/weights.json`) via `training/export_weights.py` without NumPy.

Weight Loading Mechanism:
-------------------------
By default, WEIGHTS_PATH points to "training/weights.json".
When layers are instantiated, their `init_weights()` method queries the
pre-trained weights using the layer's hierarchical dot-separated name
(e.g., "decoder.decoder_blocks.0.gqa_block.gqa.groups.0.w_q").
If WEIGHTS_PATH is set to None (or the file does not exist), the model
smoothly falls back to seeded pseudo-random weight initialization.

Code Organization:
------------------
- Classes organize weights and define layer hierarchy. Calling a class
  instance invokes its `predict()` method.
- Standalone pure functions execute the underlying mathematical operations
  (RMSNorm, RoPE, attention, SwiGLU, softmax, and matrix multiplication).

Tensor Shapes (No Batch Dimension):
-----------------------------------
Every tensor here is a nested Python list, and every one of them is commented
with its shape, both in the docstrings and above each assignment in the code.

None of those shapes have a batch dimension. Real frameworks put batch first
(`[batch_size, seq_len, hidden_size]`) so they can push many sequences through
the GPU at once, but that leading axis is a throughput trick, not part of the
architecture. This file runs one sequence at a time, so an activation is
`[seq_len, hidden_size]` and a single token vector is just `[hidden_size]`.
Dropping the batch axis removes one level of indexing from every loop.

Most dimension names in the shape comments are the constants defined below:
vocab_size, hidden_size, num_groups, q_heads, d_head, head_dim, num_experts,
inter_size, and TOP_K. The exception is seq_len, which is however many tokens
are in the sequence right now and grows by one with each generated token.

Reading Guide:
--------------
The file is ordered bottom-up, from the smallest operations to the full model:
vocabulary, weight loading, tokenizer, the Layer base class, basic math ops,
RMSNorm, RoPE, attention, GQA, MoE, decoder, embedding and LM head, sampler,
Model, and finally the pipeline.

The article walks the same code top-down instead, starting from `pipeline` at
the bottom of this file and drilling into each submodule. Either direction
works: jump to `main()` and `pipeline()` at the end to follow the article, or
start from the top to build the model up from arithmetic.
"""

import json
import math
import os
import random

# Path to pre-trained weights JSON file.
# Set to None to run with seeded pseudo-random initialization instead.
WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "training", "weights.json")

# ============================== Vocabulary & Architecture Constants ==============================

# Minimal 12-token vocabulary containing only the tokens needed for:
# "What is 1+1? It's 2.<eos>" plus special tokens (<pad>, <bos>, <eos>).
#
# Real tokenizers attach a leading space to the word that follows it, so " 1"
# (with a space) and "1" (without) are two different tokens with two different
# IDs. We keep that behavior here: " is", " It's", and " 2" all carry their
# leading space, which is why decoding a sequence is a plain string join.
vocab = {
    "<pad>": 0,    # Padding token
    "<bos>": 1,    # Beginning of sequence
    "<eos>": 2,    # End of sequence
    "What": 3,
    " is": 4,
    " 1": 5,
    "+": 6,
    "1": 7,
    "?": 8,
    " It's": 9,
    " 2": 10,
    ".": 11,
}

VOCAB_SIZE = len(vocab)          # 12: Total number of unique tokens in vocabulary
HIDDEN_SIZE = 12                 # 12: Vector dimension representing each token throughout the model
NUM_DECODER_BLOCKS = 2           # 2: Number of sequential transformer decoder blocks
NUM_GROUPS = 3                   # 3: Attention groups in Grouped-Query Attention (GQA)
Q_HEADS = 2                      # 2: Query heads per attention group (total query heads = 3 * 2 = 6)
D_HEAD = 2                       # 2: Dimension of each attention head vector
HEAD_DIM = Q_HEADS * D_HEAD      # 4: Output width of one group, also equals HIDDEN_SIZE // NUM_GROUPS
NUM_EXPERTS = 3                  # 3: Expert feed-forward networks in each MoE block
TOP_K = 2                        # 2: Number of top experts activated per token by the router
INTER_SIZE = 16                  # 16: Hidden intermediate dimension inside each SwiGLU expert

MAX_NEW_TOKENS = 128             # Maximum number of tokens generated in autoregressive loop
EOS_TOKEN_ID = 2                 # Stop generation immediately when this token is emitted (vocab["<eos>"])
EPS = 1e-6                       # Small constant added to variance in RMSNorm to prevent division by zero

# ============================== Weight Initialization & Loading Helper ==============================

# Seeded random number generator for reproducible fallback weights when WEIGHTS_PATH is None
_rng = random.Random(42)

# In-memory cache mapping file paths to loaded weight dictionaries, avoiding redundant disk reads
_loaded_weights_cache = {}

def _get_loaded_weight(name):
    """Retrieve a pre-trained tensor by its hierarchical name from the JSON weights file.

    The file is read once and kept in `_loaded_weights_cache`, because every
    layer asks for its own tensor and we do not want to re-parse the JSON
    hundreds of times while the model is being built.

    Args:
        name: str, dot-separated hierarchical tensor name
              (e.g., 'decoder.decoder_blocks.0.gqa_block.rms_norm.gamma')

    Returns:
        Nested list containing the tensor weights if found, otherwise None.
    """
    # No weights file configured, or an unnamed tensor: nothing to look up
    if not WEIGHTS_PATH or not name:
        return None

    # Read and cache the file the first time we need it
    if WEIGHTS_PATH not in _loaded_weights_cache and os.path.exists(WEIGHTS_PATH):
        with open(WEIGHTS_PATH, "r", encoding="utf-8") as f:
            _loaded_weights_cache[WEIGHTS_PATH] = json.load(f)

    cache = _loaded_weights_cache.get(WEIGHTS_PATH)
    if cache and name in cache:
        return cache[name]
    return None

def clear_weights_cache():
    """Forget any weights read from disk, so the next Model() re-reads WEIGHTS_PATH.

    Only needed when WEIGHTS_PATH changes at runtime, which happens in
    `training/export_weights.py` and in the tests.
    """
    _loaded_weights_cache.clear()

def init_weights(*shape, name=None):
    """Initialize a weight tensor by loading from JSON, or fall back to pseudo-random numbers.

    Args:
        *shape: tuple of int defining tensor dimensions (1D, 2D, or 3D).
        name: optional str specifying the hierarchical tensor name for JSON lookup.

    Returns:
        list: 1D, 2D, or 3D nested Python list of floats.
    """
    # 1. Attempt to load pre-trained tensor from JSON
    loaded = _get_loaded_weight(name)
    if loaded is not None:
        return loaded

    # 2. Fall back to pseudo-random initialization
    if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
        shape = tuple(shape[0])

    if len(shape) == 1:
        # 1D vectors (such as RMSNorm scale gamma) initialize to 1.0
        # weights_1d: [shape[0]]
        weights_1d = [1.0] * shape[0]
        return weights_1d
    elif len(shape) == 2:
        # 2D weight matrices initialize uniformly in [-0.02, 0.02]
        rows, cols = shape
        # weights_2d: [rows, cols]
        weights_2d = [[_rng.uniform(-0.02, 0.02) for _ in range(cols)] for _ in range(rows)]
        return weights_2d
    elif len(shape) == 3:
        # 3D weight tensors (such as grouped query weights [heads, rows, cols])
        heads, rows, cols = shape
        # weights_3d: [heads, rows, cols]
        weights_3d = [
            [[_rng.uniform(-0.02, 0.02) for _ in range(cols)] for _ in range(rows)]
            for _ in range(heads)
        ]
        return weights_3d
    raise ValueError(f"Unsupported shape: {shape}")

# ============================== Tokenizer ==============================

def split_tokens(text, vocab_dict=vocab):
    """Split input text into tokens using greedy longest-match prefix matching.

    Scans the string from left to right. At each position, it finds all vocabulary
    tokens that match the current prefix and selects the longest match. If no token
    matches and the character is a space, the space is skipped. Otherwise, the single
    character is emitted as a fallback token.

    Real tokenizers use Byte Pair Encoding, which learns its merges from data and
    can represent any input. Greedy longest-match is close enough to show the idea
    without the training machinery.

    Args:
        text: str, raw input text to tokenize.
        vocab_dict: dict mapping token string -> token integer ID.

    Returns:
        list of str: extracted token strings.
    """
    # tokens: [seq_len] token strings, grown one match at a time
    tokens = []
    i = 0
    while i < len(text):
        # Find all vocabulary entries matching the prefix starting at index i
        matches = [t for t in vocab_dict if text[i:].startswith(t)]
        if matches:
            # Greedy longest-match tokenization
            best = max(matches, key=len)
            tokens.append(best)
            i += len(best)
        elif text[i] == " ":
            # Skip unmapped whitespace
            i += 1
        else:
            # Fallback for unknown single characters. Our 12-token vocabulary has
            # no <unk> entry, so these characters have no ID and encode() will
            # reject them below.
            tokens.append(text[i])
            i += 1
    return tokens

class Tokenizer:
    """Converts raw text to integer token IDs and decodes token IDs back to text."""

    def __init__(self, vocab=vocab):
        """Initialize tokenizer with vocabulary dictionary and build inverse lookup.

        Args:
            vocab: dict mapping token string -> token ID int.
        """
        self.vocab = vocab
        # Inverse mapping: token ID int -> token string, used by decode()
        self.inv_vocab = {token_id: token for token, token_id in vocab.items()}

    def encode(self, text):
        """Tokenize input text and convert tokens into integer token IDs.

        Raises KeyError if the text contains anything outside our 12-token
        vocabulary, which in practice means anything other than the prompt
        "What is 1+1?" and the sentence the model was trained on.

        Args:
            text: str, raw input text.

        Returns:
            list of int: token IDs of length [seq_len].
        """
        # tokens: [seq_len] token strings
        tokens = split_tokens(text, self.vocab)
        # input_ids: [seq_len] token IDs
        input_ids = [self.vocab[token] for token in tokens]
        return input_ids

    def decode(self, token_ids):
        """Reconstruct string from integer token IDs.

        Args:
            token_ids: list of int, token IDs of length [seq_len].

        Returns:
            str: decoded text string.
        """
        return "".join(self.inv_vocab.get(token_id, f"<{token_id}>") for token_id in token_ids)

# ============================== Base Neural Network Class ==============================

class Layer:
    """Base class for all neural network modules in readable-llm.

    Similar to torch.nn.Module, Layer encapsulates learnable parameters and
    provides a consistent interface:
    1. Hierarchical naming: Each layer has a `name` (e.g. 'decoder.decoder_blocks.0').
       Calling `self.init_weights('gamma', HIDDEN_SIZE)` constructs the full path
       'decoder.decoder_blocks.0.gamma' for pre-trained weights lookup.
    2. Execution: Calling an instance executes its `predict()` method.
    """

    def __init__(self, name=None):
        """
        Args:
            name: optional str specifying this layer's hierarchical identifier.
                  Defaults to the lowercased class name if omitted.
        """
        self.name = name or self.__class__.__name__.lower()

    def init_weights(self, tensor_name, *shape):
        """Initialize or load weights for a named tensor belonging to this layer.

        Args:
            tensor_name: str, local tensor identifier (e.g. 'gamma', 'w_q', 'w_router').
            *shape: tuple of int, tensor dimensions.

        Returns:
            list: populated weight tensor (loaded from JSON if available, else random).
        """
        full_name = f"{self.name}.{tensor_name}"
        return init_weights(*shape, name=full_name)

    def __call__(self, *args, **kwargs):
        """Forward all calls to the layer's predict method."""
        return self.predict(*args, **kwargs)

    def predict(self, *args, **kwargs):
        """Compute the layer's forward pass. Must be overridden by subclasses."""
        raise NotImplementedError

# ============================== Basic Operations ==============================

def softmax(scores):
    """Convert unnormalized real-valued scores into probabilities that sum to 1.0.

    Applies the numerically stable softmax formula:
        softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
    Subtracting the maximum prevents float overflow in math.exp without altering
    the resulting probability distribution.

    Called twice in this file: once on attention scores, once on router logits.

    Args:
        scores: list of float [n], raw unnormalized scores.

    Returns:
        list of float [n]: normalized probabilities summing to 1.0.
    """
    # 1. Find max value for numerical stability to prevent float overflow
    max_score = max(scores)
    # 2. Exponentiate shifted scores
    # exp_scores: [n]
    exp_scores = [math.exp(s - max_score) for s in scores]
    # 3. Sum of exponentiated values
    sum_exp = sum(exp_scores)
    # 4. Normalize to get probability distribution
    # probs: [n]
    probs = [s / sum_exp for s in exp_scores]
    return probs

def argmax(logits):
    """Find the index of the highest score (greedy token selection).

    Args:
        logits: list of float [vocab_size], output scores across vocabulary.

    Returns:
        int: index corresponding to the maximum score.
    """
    max_idx = 0
    for i in range(len(logits)):
        if logits[i] > logits[max_idx]:
            max_idx = i
    return max_idx

def silu(x):
    """Compute the Sigmoid Linear Unit (SiLU / Swish) activation: x * sigmoid(x).

    Formula:
        silu(x) = x / (1.0 + exp(-x))
    Used extensively in modern transformer architectures (such as LLaMA and Gemma)
    within the SwiGLU expert feed-forward networks.

    Args:
        x: float, scalar input.

    Returns:
        float: activated scalar value.
    """
    return x / (1.0 + math.exp(-x))

def matmul(vec, matrix):
    """Compute standard vector-matrix multiplication: vec @ matrix.

    Takes a 1D vector of length [in_dim] and a 2D matrix of shape [in_dim, out_dim],
    and computes the dot product of vec with each column of the matrix:
        output[j] = sum(vec[k] * matrix[k][j] for k in range(in_dim))

    Args:
        vec: list of float [in_dim]
        matrix: list of list of float [in_dim, out_dim]

    Returns:
        list of float [out_dim]
    """
    in_dim = len(vec)
    out_dim = len(matrix[0])
    # output: [out_dim]
    output = []
    # Compute dot product between vec and each column of matrix
    for col in range(out_dim):
        dot_product = sum(vec[k] * matrix[k][col] for k in range(in_dim))
        output.append(dot_product)
    return output

def add(tensor_a, tensor_b):
    """Element-wise addition of two 2D tensors for residual (skip) connections.

    Computes tensor_a + tensor_b element by element:
        output[i][j] = tensor_a[i][j] + tensor_b[i][j]
    Residual connections allow gradients to flow directly through deep stacks
    and prevent degradation during training.

    Args:
        tensor_a: list of list of float [seq_len, hidden_size]
        tensor_b: list of list of float [seq_len, hidden_size]

    Returns:
        list of list of float [seq_len, hidden_size]
    """
    seq_len = len(tensor_a)
    hidden_size = len(tensor_a[0])

    # Walk the two tensors position by position and sum the matching cells
    # output: [seq_len, hidden_size]
    output = [[0.0] * hidden_size for _ in range(seq_len)]
    for i in range(seq_len):
        for j in range(hidden_size):
            output[i][j] = tensor_a[i][j] + tensor_b[i][j]
    return output

# ============================== RMSNorm ==============================

def norm_token(token_vec, gamma):
    """Apply Root Mean Square Layer Normalization (RMSNorm) to a single token vector.

    Unlike standard LayerNorm, RMSNorm does not center the inputs by subtracting the
    mean; it only normalizes by the root mean square of the elements:
        RMS(x) = sqrt( mean(x^2) + eps )
        norm_x = (x / RMS(x)) * gamma

    Call path: model.predict -> ... -> rms_norm -> norm_token

    Args:
        token_vec: list of float [hidden_size]
        gamma: list of float [hidden_size], learnable scale parameters

    Returns:
        list of float [hidden_size]: normalized and scaled token vector
    """
    # 1. Compute sum of squares: sum(x_i^2)
    sum_of_squares = 0.0
    for v in token_vec:
        sum_of_squares += v ** 2

    # 2. Compute root mean square with epsilon to prevent division by zero:
    rms = (sum_of_squares / len(token_vec) + EPS) ** 0.5

    # 3. Scale each element by gamma: (x_i / rms) * gamma_i
    # output: [hidden_size]
    output = []
    for i in range(len(token_vec)):
        output.append(token_vec[i] / rms * gamma[i])
    return output

def rms_norm(tensor, gamma):
    """Apply RMSNorm across all token vectors in a sequence tensor.

    Call path: model.predict -> (lm_head | gqa_block | moe_block) -> rms_norm

    Args:
        tensor: list of list of float [seq_len, hidden_size]
        gamma: list of float [hidden_size], learnable scale parameters

    Returns:
        list of list of float [seq_len, hidden_size]
    """
    # rms_out: [seq_len, hidden_size]
    rms_out = [norm_token(token_vec, gamma) for token_vec in tensor]
    return rms_out

class RMSNorm(Layer):
    """Encapsulates RMS normalization scale weights (gamma).

    RMSNorm is applied before attention (in GQABlock), before routing (in MoEBlock),
    and before the final logit projection (in LMHead).
    """

    def __init__(self, name="rms_norm"):
        super().__init__(name)
        # gamma: [hidden_size] = [12], scales initialize to 1.0
        self.gamma = self.init_weights("gamma", HIDDEN_SIZE)

    def predict(self, tensor_or_vec):
        """Execute RMS normalization on a sequence tensor or a single token vector.

        Call path: model.predict -> (lm_head | gqa_block | moe_block) -> rms_norm

        Args:
            tensor_or_vec: [seq_len, hidden_size] or [hidden_size]

        Returns:
            [seq_len, hidden_size] or [hidden_size]
        """
        if isinstance(tensor_or_vec[0], list):
            return rms_norm(tensor_or_vec, self.gamma)
        return norm_token(tensor_or_vec, self.gamma)

# ============================== RoPE ==============================

def rope_pair(x0, x1, pos, i, d_head):
    """Apply 2D Rotary Position Embedding rotation to a pair of coordinates.

    RoPE rotates adjacent coordinate pairs (x0, x1) in the complex plane by an
    angle proportional to token position and inverse frequency:
        theta = pos / (10000 ** (i / d_head))
        [x0'] = [cos(theta)  -sin(theta)] [x0]
        [x1']   [sin(theta)   cos(theta)] [x1]
    When dot products are taken between rotated query and key vectors later in
    attention, the absolute positions cancel out, leaving only relative distance
    (pos_q - pos_k).

    Call path: model.predict -> decoder -> decoder_block -> gqa_block -> gqa -> group_0 -> rope -> rope_token -> rope_pair

    Args:
        x0: float, first feature in the pair
        x1: float, second feature in the pair
        pos: int, token sequence position index (0, 1, 2, ...)
        i: int, feature dimension index (0, 2, 4, ...)
        d_head: int, total dimension of the attention head

    Returns:
        tuple of (float, float): rotated (x0', x1') coordinates
    """
    # Inverse frequency for dimension i
    freq = 1.0 / (10000 ** (i / d_head))
    # Rotation angle for token at position pos
    angle = pos * freq
    cos_val = math.cos(angle)
    sin_val = math.sin(angle)
    # Apply standard 2D rotation matrix
    return x0 * cos_val - x1 * sin_val, x0 * sin_val + x1 * cos_val

def rope_token(token_vec, pos):
    """Apply RoPE rotation to all coordinate pairs in a single head vector.

    Call path: model.predict -> decoder -> decoder_block -> gqa_block -> gqa -> group_0 -> rope -> rope_token

    Args:
        token_vec: list of float [d_head]
        pos: int, token sequence position

    Returns:
        list of float [d_head]: position-encoded head vector
    """
    d_head = len(token_vec)
    # output: [d_head], filled two entries at a time
    output = []
    # Process head vector in adjacent pairs: (0, 1), (2, 3), ...
    for i in range(0, d_head, 2):
        # r0, r1: scalars, the rotated version of the (i, i + 1) pair
        r0, r1 = rope_pair(token_vec[i], token_vec[i + 1], pos, i, d_head)
        output.append(r0)
        output.append(r1)
    return output

def rope_2d(x):
    """Apply RoPE to a 2D tensor across all token positions (e.g., Key projections).

    Call path: model.predict -> decoder -> decoder_block -> gqa_block -> gqa -> group_0 -> rope -> rope_2d

    Args:
        x: list of list of float [seq_len, d_head]

    Returns:
        list of list of float [seq_len, d_head]
    """
    # output: [seq_len, d_head]
    output = [rope_token(token_vec, pos) for pos, token_vec in enumerate(x)]
    return output

def rope_3d(x):
    """Apply RoPE to a 3D tensor across all heads and positions (e.g., Query projections).

    Call path: model.predict -> decoder -> decoder_block -> gqa_block -> gqa -> group_0 -> rope -> rope_3d

    Args:
        x: list of list of list of float [q_heads, seq_len, d_head]

    Returns:
        list of list of list of float [q_heads, seq_len, d_head]
    """
    # output: [q_heads, seq_len, d_head]
    output = [rope_2d(head) for head in x]
    return output

def rope(x):
    """Generic RoPE dispatcher supporting both 2D (keys) and 3D (queries) inputs.

    Call path: model.predict -> decoder -> decoder_block -> gqa_block -> gqa -> group_0 -> rope

    Args:
        x: [seq_len, d_head] or [q_heads, seq_len, d_head]

    Returns:
        [seq_len, d_head] or [q_heads, seq_len, d_head]
    """
    if isinstance(x[0][0], list):
        return rope_3d(x)
    return rope_2d(x)

# ============================== Attention ==============================

def attention_token(q_token, k_T, v, i):
    """Compute causal scaled dot-product attention for a single query token.

    Computes:
        dot_products = q_token @ k_T
        scores  = dot_products[:i + 1] / sqrt(d_head)  # Causal mask: keep tokens 0..i
        weights = softmax(scores) padded with zeros for the future positions
        output  = weights @ v

    Call path: model.predict -> decoder -> decoder_block -> gqa_block -> gqa -> group_0 -> attention_head -> dot_product_attention -> attention_token

    Args:
        q_token: list of float [d_head], query vector for token at position i
        k_T: list of list of float [d_head, seq_len], transposed key matrix
        v: list of list of float [seq_len, d_head], value matrix
        i: int, position index of the current query token

    Returns:
        list of float [d_head]: attention output vector for token i
    """
    d_head = len(q_token)
    seq_len = len(v)

    # 1. Compute raw similarity dot products against every key: [seq_len]
    # dot_products: [seq_len]
    dot_products = matmul(q_token, k_T)

    # 2. Causal masking plus scaling. We only keep positions 0..i, so token i
    #    never sees the future, and we divide by sqrt(d_head) to stop the dot
    #    products from growing with head width and saturating the softmax.
    # scores: [i + 1]
    scores = []
    for j in range(i + 1):
        scores.append(dot_products[j] / (d_head ** 0.5))

    # 3. Turn the visible scores into probabilities, then pad the masked-out
    #    future positions with 0.0 so the vector lines up with v: [seq_len]
    # weights: [seq_len]
    weights = softmax(scores) + [0.0] * (seq_len - (i + 1))

    # 4. Weighted sum of value vectors: sum(weight_j * v_j)
    # token_out: [d_head]
    token_out = matmul(weights, v)
    return token_out

def dot_product_attention(q_head, k, v):
    """Compute causal self-attention across the full sequence for a single head.

    Call path: model.predict -> decoder -> decoder_block -> gqa_block -> gqa -> group_0 -> attention_head -> dot_product_attention

    Args:
        q_head: list of list of float [seq_len, d_head]
        k: list of list of float [seq_len, d_head]
        v: list of list of float [seq_len, d_head]

    Returns:
        list of list of float [seq_len, d_head]
    """
    seq_len = len(k)
    d_head = len(k[0])

    # Transpose key matrix from [seq_len, d_head] to [d_head, seq_len] for fast dot products
    # k_T: [d_head, seq_len]
    k_T = [[k[row][col] for row in range(seq_len)] for col in range(d_head)]

    # Compute attention output for each token in the sequence
    # head_out: [seq_len, d_head]
    head_out = []
    for i, q_token in enumerate(q_head):
        # token_out: [d_head]
        token_out = attention_token(q_token, k_T, v, i)
        head_out.append(token_out)
    return head_out

def attention_head(q, k, v):
    """Execute multi-head attention within a single GQA group and concatenate head outputs.

    In Grouped-Query Attention, multiple query heads (q_heads) share the same single
    key head (k) and value head (v).

    Call path: model.predict -> decoder -> decoder_block -> gqa_block -> gqa -> group_0 -> attention_head

    Args:
        q: list of list of list of float [q_heads, seq_len, d_head]
        k: list of list of float [seq_len, d_head]
        v: list of list of float [seq_len, d_head]

    Returns:
        list of list of float [seq_len, head_dim]: concatenated query head outputs
    """
    q_heads = len(q)
    seq_len = len(k)

    # Compute attention for each query head independently against the shared (k, v)
    # head_outs: [q_heads, seq_len, d_head]
    head_outs = [dot_product_attention(q_head, k, v) for q_head in q]

    # Concatenate the head vectors for each token: [q_heads * d_head] = [head_dim]
    # out: [seq_len, head_dim]
    out = []
    for t in range(seq_len):
        # token_out: [head_dim], one token's slice of every head's output
        token_out = []
        for h in range(q_heads):
            token_out.extend(head_outs[h][t])
        out.append(token_out)
    return out

# ============================== Grouped-Query Attention (GQA) ==============================

def group_0(rms_out, w_q, w_k, w_v):
    """Compute attention for a single GQA group.

    Projects normalized token representations into queries, keys, and values,
    applies Rotary Position Embeddings (RoPE) to queries and keys, and computes
    multi-head attention.

    The name follows the article's diagrams, which walk through the first group
    (`group_0`) in detail. Every group runs this exact same code with its own
    copy of the weights.

    Call path: model.predict -> decoder -> decoder_block -> gqa_block -> gqa -> group_0

    Args:
        rms_out: list of list of float [seq_len, hidden_size], normalized input
        w_q: list of list of list of float [q_heads, hidden_size, d_head]
        w_k: list of list of float [hidden_size, d_head]
        w_v: list of list of float [hidden_size, d_head]

    Returns:
        list of list of float [seq_len, head_dim]: concatenated group attention output
    """
    # 1. Project input to Q (multiple heads), K (single head), and V (single head)
    # q: [q_heads, seq_len, d_head]
    q = [[matmul(token_vec, w) for token_vec in rms_out] for w in w_q]
    # k: [seq_len, d_head]
    k = [matmul(token_vec, w_k) for token_vec in rms_out]
    # v: [seq_len, d_head]
    v = [matmul(token_vec, w_v) for token_vec in rms_out]

    # 2. Inject relative positional information via RoPE into queries and keys
    # q: [q_heads, seq_len, d_head], same shape, rotated in place
    q = rope(q)
    # k: [seq_len, d_head], same shape, rotated in place
    k = rope(k)

    # 3. Compute causal multi-head attention and concatenate query heads
    # group_0_out: [seq_len, head_dim] where head_dim = q_heads * d_head
    group_0_out = attention_head(q, k, v)
    return group_0_out

class Group(Layer):
    """Encapsulates Q, K, and V projection weights for a single attention group.

    In Grouped-Query Attention, each group has multiple query heads (Q_HEADS)
    sharing a single key head and a single value head.
    """

    def __init__(self, name="group_0"):
        super().__init__(name)
        # w_q: [q_heads, hidden_size, d_head] = [2, 12, 2]
        self.w_q = self.init_weights("w_q", Q_HEADS, HIDDEN_SIZE, D_HEAD)
        # w_k: [hidden_size, d_head] = [12, 2]
        self.w_k = self.init_weights("w_k", HIDDEN_SIZE, D_HEAD)
        # w_v: [hidden_size, d_head] = [12, 2]
        self.w_v = self.init_weights("w_v", HIDDEN_SIZE, D_HEAD)

    def predict(self, rms_out):
        """Execute grouped query attention for this group.

        Call path: model.predict -> decoder -> decoder_block -> gqa_block -> gqa -> group_0

        Args:
            rms_out: [seq_len, hidden_size]

        Returns:
            [seq_len, head_dim]
        """
        return group_0(rms_out, self.w_q, self.w_k, self.w_v)

def gqa(rms_out, groups):
    """Compute Grouped-Query Attention across all groups and concatenate along channel dimension.

    Iterates over NUM_GROUPS groups (each producing [seq_len, head_dim]), and
    concatenates their outputs along the hidden dimension to produce [seq_len, hidden_size].

    Call path: model.predict -> decoder -> decoder_block -> gqa_block -> gqa

    Args:
        rms_out: list of list of float [seq_len, hidden_size]
        groups: list of Group

    Returns:
        list of list of float [seq_len, hidden_size]
    """
    seq_len = len(rms_out)

    # 1. Run each attention group
    # group_outs: [num_groups, seq_len, head_dim]
    group_outs = [group(rms_out) for group in groups]

    # 2. Concatenate outputs across groups: num_groups * head_dim = 3 * 4 = 12 (hidden_size)
    # gqa_out: [seq_len, hidden_size]
    gqa_out = []
    for t in range(seq_len):
        # row: [hidden_size], one token's slice of every group's output
        row = []
        for out in group_outs:
            row.extend(out[t])
        gqa_out.append(row)
    return gqa_out

class GQA(Layer):
    """Encapsulates grouped-query attention across all groups."""

    def __init__(self, name="gqa"):
        super().__init__(name)
        # groups: num_groups Group layers, each holding its own w_q, w_k, w_v
        self.groups = [Group(name=f"{self.name}.groups.{i}") for i in range(NUM_GROUPS)]

    def predict(self, rms_out):
        """Run every attention group and concatenate their outputs.

        Call path: model.predict -> decoder -> decoder_block -> gqa_block -> gqa

        Args:
            rms_out: [seq_len, hidden_size]

        Returns:
            [seq_len, hidden_size]
        """
        return gqa(rms_out, self.groups)

def out_matmul(gqa_out, w_matmul):
    """Project concatenated multi-head attention outputs back through output matrix W_out.

    Call path: model.predict -> decoder -> decoder_block -> gqa_block -> out_matmul

    Args:
        gqa_out: list of list of float [seq_len, hidden_size]
        w_matmul: list of list of float [hidden_size, hidden_size]

    Returns:
        list of list of float [seq_len, hidden_size]
    """
    # gqa_block_out: [seq_len, hidden_size]
    gqa_block_out = [matmul(token_vec, w_matmul) for token_vec in gqa_out]
    return gqa_block_out

class OutMatmul(Layer):
    """Encapsulates output projection matrix for GQA."""

    def __init__(self, name="out_matmul"):
        super().__init__(name)
        # w_matmul: [hidden_size, hidden_size] = [12, 12]
        self.w_matmul = self.init_weights("w_matmul", HIDDEN_SIZE, HIDDEN_SIZE)

    def predict(self, gqa_out):
        """Project the concatenated attention output back to the residual stream.

        Call path: model.predict -> decoder -> decoder_block -> gqa_block -> out_matmul

        Args:
            gqa_out: [seq_len, hidden_size]

        Returns:
            [seq_len, hidden_size]
        """
        return out_matmul(gqa_out, self.w_matmul)

def gqa_block(gqa_block_in, rms_norm, gqa, out_matmul):
    """Complete GQA sub-layer: Pre-RMSNorm -> GQA -> OutMatmul projection.

    Call path: model.predict -> decoder -> decoder_block -> gqa_block

    Args:
        gqa_block_in: [seq_len, hidden_size]
        rms_norm: RMSNorm
        gqa: GQA
        out_matmul: OutMatmul

    Returns:
        [seq_len, hidden_size]
    """
    # 1. Pre-normalization
    # rms_out: [seq_len, hidden_size]
    rms_out = rms_norm(gqa_block_in)

    # 2. Grouped-Query Attention
    # gqa_out: [seq_len, hidden_size]
    gqa_out = gqa(rms_out)

    # 3. Output projection
    # gqa_block_out: [seq_len, hidden_size]
    gqa_block_out = out_matmul(gqa_out)
    return gqa_block_out

class GQABlock(Layer):
    """Encapsulates normalization, grouped-query attention, and output projection."""

    def __init__(self, name="gqa_block"):
        super().__init__(name)
        self.rms_norm = RMSNorm(name=f"{self.name}.rms_norm")
        self.gqa = GQA(name=f"{self.name}.gqa")
        self.out_matmul = OutMatmul(name=f"{self.name}.out_matmul")

    def predict(self, gqa_block_in):
        """Normalize, attend, and project the attention sub-layer.

        Call path: model.predict -> decoder -> decoder_block -> gqa_block

        Args:
            gqa_block_in: [seq_len, hidden_size]

        Returns:
            [seq_len, hidden_size]
        """
        return gqa_block(gqa_block_in, self.rms_norm, self.gqa, self.out_matmul)

# ============================== Mixture of Experts (MoE) ==============================

def expert_token(token_vec, w_gate, w_up, w_down):
    """Compute SwiGLU feed-forward transformation for a single token in an expert.

    SwiGLU (Swish-Gated Linear Unit) computes:
        output = (silu(token_vec @ w_gate) * (token_vec @ w_up)) @ w_down

    Call path: model.predict -> decoder -> decoder_block -> moe_block -> moe -> expert -> expert_token

    Args:
        token_vec: list of float [hidden_size]
        w_gate: list of list of float [hidden_size, inter_size]
        w_up: list of list of float [hidden_size, inter_size]
        w_down: list of list of float [inter_size, hidden_size]

    Returns:
        list of float [hidden_size]: expert output vector
    """
    # 1. Gate projection: [hidden_size] -> [inter_size]
    # x_gate: [inter_size]
    x_gate = matmul(token_vec, w_gate)

    # 2. Up projection: [hidden_size] -> [inter_size]
    # x_up: [inter_size]
    x_up = matmul(token_vec, w_up)

    # 3. Apply SiLU (Swish) non-linear activation to gate projection
    # x_act: [inter_size]
    x_act = [silu(x) for x in x_gate]

    # 4. Element-wise multiply activated gate with up projection
    inter_size = len(x_act)
    # x_inter: [inter_size]
    x_inter = [x_act[i] * x_up[i] for i in range(inter_size)]

    # 5. Down projection back to hidden size: [inter_size] -> [hidden_size]
    # x_down: [hidden_size]
    x_down = matmul(x_inter, w_down)
    return x_down

def expert(tensor, w_gate, w_up, w_down):
    """Apply SwiGLU expert feed-forward network across all tokens in a sequence.

    Call path: model.predict -> decoder -> decoder_block -> moe_block -> moe -> expert

    Args:
        tensor: list of list of float [seq_len, hidden_size]
        w_gate: [hidden_size, inter_size]
        w_up: [hidden_size, inter_size]
        w_down: [inter_size, hidden_size]

    Returns:
        list of list of float [seq_len, hidden_size]
    """
    # output: [seq_len, hidden_size]
    output = [expert_token(token_vec, w_gate, w_up, w_down) for token_vec in tensor]
    return output

class Expert(Layer):
    """Encapsulates SwiGLU projection weights for a single expert."""

    def __init__(self, name="expert"):
        super().__init__(name)
        # w_gate: [hidden_size, inter_size] = [12, 16]
        self.w_gate = self.init_weights("w_gate", HIDDEN_SIZE, INTER_SIZE)
        # w_up: [hidden_size, inter_size] = [12, 16]
        self.w_up = self.init_weights("w_up", HIDDEN_SIZE, INTER_SIZE)
        # w_down: [inter_size, hidden_size] = [16, 12]
        self.w_down = self.init_weights("w_down", INTER_SIZE, HIDDEN_SIZE)

    def predict(self, token_vec_or_tensor):
        """Run this expert's SwiGLU feed-forward network.

        Call path: model.predict -> decoder -> decoder_block -> moe_block -> moe -> expert

        Args:
            token_vec_or_tensor: [seq_len, hidden_size] or [hidden_size]

        Returns:
            [seq_len, hidden_size] or [hidden_size]
        """
        if isinstance(token_vec_or_tensor[0], list):
            return expert(token_vec_or_tensor, self.w_gate, self.w_up, self.w_down)
        return expert_token(token_vec_or_tensor, self.w_gate, self.w_up, self.w_down)

def route_token(token_vec, w_router):
    """Compute top-k routing weights for a single token.

    Projects the token vector into expert affinity logits, selects the top-k
    experts with highest scores, and computes normalized softmax probabilities
    for those chosen experts while setting inactive experts to 0.0.

    Call path: model.predict -> decoder -> decoder_block -> moe_block -> router -> route_token

    Args:
        token_vec: list of float [hidden_size]
        w_router: list of list of float [hidden_size, num_experts]

    Returns:
        list of float [num_experts]: sparse routing weights summing to 1.0
    """
    # 1. Project token vector to router logits: [hidden_size] @ [hidden_size, num_experts]
    # logits: [num_experts]
    logits = matmul(token_vec, w_router)

    # 2. Sort expert indices by descending logit values
    num_experts = len(logits)
    # indices: [num_experts]
    indices = [i for i in range(num_experts)]
    # sorted_indices: [num_experts]
    sorted_indices = sorted(indices, key=lambda i: logits[i], reverse=True)

    # 3. Select top-k experts
    # top_indices: [TOP_K]
    top_indices = sorted_indices[:TOP_K]
    # top_logits: [TOP_K]
    top_logits = [logits[i] for i in top_indices]

    # 4. Compute softmax probabilities over the selected top-k experts
    # top_probs: [TOP_K]
    top_probs = softmax(top_logits)

    # 5. Populate sparse weight vector (selected experts get probabilities; others get 0.0)
    # top_weights: [num_experts]
    top_weights = [0.0] * num_experts
    for k in range(TOP_K):
        top_weights[top_indices[k]] = top_probs[k]
    return top_weights

def router(rms_out, w_router):
    """Compute routing weights for every token in the sequence.

    Call path: model.predict -> decoder -> decoder_block -> moe_block -> router

    Args:
        rms_out: list of list of float [seq_len, hidden_size]
        w_router: list of list of float [hidden_size, num_experts]

    Returns:
        list of list of float [seq_len, num_experts]
    """
    # top_weights: [seq_len, num_experts]
    top_weights = [route_token(token_vec, w_router) for token_vec in rms_out]
    return top_weights

class Router(Layer):
    """Encapsulates routing weights to select top-k experts."""

    def __init__(self, name="router"):
        super().__init__(name)
        # w_router: [hidden_size, num_experts] = [12, 3]
        self.w_router = self.init_weights("w_router", HIDDEN_SIZE, NUM_EXPERTS)

    def predict(self, rms_out):
        """Score the experts and return sparse routing weights per token.

        Call path: model.predict -> decoder -> decoder_block -> moe_block -> router

        Args:
            rms_out: [seq_len, hidden_size]

        Returns:
            [seq_len, num_experts]
        """
        return router(rms_out, self.w_router)

def moe_token(token_vec, top_weights, experts):
    """Compute MoE output for a single token as the weighted sum of expert outputs.

    Only the TOP_K experts the router picked contribute. The rest have a weight
    of exactly 0.0, and we skip them instead of multiplying their output by
    zero. That skip is the entire point of a Mixture of Experts: the model can
    hold many experts while each token only pays for a few of them.

    The article's diagram draws the multiply-by-zero version, because showing
    every expert makes the routing easier to see. Both produce the same numbers.

    Call path: model.predict -> decoder -> decoder_block -> moe_block -> moe -> moe_token

    Args:
        token_vec: list of float [hidden_size]
        top_weights: list of float [num_experts], routing probabilities
        experts: list of Expert

    Returns:
        list of float [hidden_size]: combined expert output
    """
    hidden_size = len(token_vec)
    # moe_out: [hidden_size], the running weighted sum of expert outputs
    moe_out = [0.0] * hidden_size

    # Accumulate weighted contributions from active experts
    for i in range(len(experts)):
        # Experts the router did not pick have weight 0.0 and never run
        if top_weights[i] == 0.0:
            continue
        # expert_out: [hidden_size]
        expert_out = experts[i](token_vec)
        for j in range(hidden_size):
            moe_out[j] += top_weights[i] * expert_out[j]
    return moe_out

def moe(rms_out, top_weights, experts):
    """Compute Mixture of Experts across all tokens in the sequence.

    Call path: model.predict -> decoder -> decoder_block -> moe_block -> moe

    Args:
        rms_out: list of list of float [seq_len, hidden_size]
        top_weights: list of list of float [seq_len, num_experts]
        experts: list of Expert

    Returns:
        list of list of float [seq_len, hidden_size]
    """
    # moe_out: [seq_len, hidden_size]
    moe_out = [
        moe_token(token_vec, weights, experts)
        for token_vec, weights in zip(rms_out, top_weights)
    ]
    return moe_out

class MoE(Layer):
    """Encapsulates the collection of experts and weighted aggregation."""

    def __init__(self, name="moe"):
        super().__init__(name)
        # experts: num_experts Expert layers, each holding w_gate, w_up, w_down
        self.experts = [Expert(name=f"{self.name}.experts.{i}") for i in range(NUM_EXPERTS)]

    def predict(self, rms_out, top_weights):
        """Blend the expert outputs using the router's weights.

        Call path: model.predict -> decoder -> decoder_block -> moe_block -> moe

        Args:
            rms_out: [seq_len, hidden_size]
            top_weights: [seq_len, num_experts]

        Returns:
            [seq_len, hidden_size]
        """
        return moe(rms_out, top_weights, self.experts)

def moe_block(moe_in, rms_norm, router, moe):
    """Complete MoE sub-layer: Pre-RMSNorm -> Router -> MoE aggregation.

    Call path: model.predict -> decoder -> decoder_block -> moe_block

    Args:
        moe_in: [seq_len, hidden_size]
        rms_norm: RMSNorm
        router: Router
        moe: MoE

    Returns:
        [seq_len, hidden_size]
    """
    # 1. Pre-normalization
    # rms_out: [seq_len, hidden_size]
    rms_out = rms_norm(moe_in)

    # 2. Route tokens to determine expert gating weights
    # top_weights: [seq_len, num_experts]
    top_weights = router(rms_out)

    # 3. Compute expert feed-forward outputs and aggregate by weights
    # moe_out: [seq_len, hidden_size]
    moe_out = moe(rms_out, top_weights)
    return moe_out

class MoEBlock(Layer):
    """Encapsulates normalization, router, and mixture of experts."""

    def __init__(self, name="moe_block"):
        super().__init__(name)
        self.rms_norm = RMSNorm(name=f"{self.name}.rms_norm")
        self.router = Router(name=f"{self.name}.router")
        self.moe = MoE(name=f"{self.name}.moe")

    def predict(self, moe_in):
        """Normalize, route, and mix the experts for this sub-layer.

        Call path: model.predict -> decoder -> decoder_block -> moe_block

        Args:
            moe_in: [seq_len, hidden_size]

        Returns:
            [seq_len, hidden_size]
        """
        return moe_block(moe_in, self.rms_norm, self.router, self.moe)

# ============================== Decoder ==============================

def decoder_block(decoder_in, gqa_block, moe_block):
    """Execute a single transformer decoder block with dual residual connections.

    Structure:
        1. GQA sub-block: residual_1 = decoder_in + gqa_block(decoder_in)
        2. MoE sub-block: output = residual_1 + moe_block(residual_1)

    Call path: model.predict -> decoder -> decoder_block

    Args:
        decoder_in: list of list of float [seq_len, hidden_size]
        gqa_block: GQABlock
        moe_block: MoEBlock

    Returns:
        list of list of float [seq_len, hidden_size]
    """
    # 1. Grouped-Query Attention sub-block with residual connection
    # gqa_out: [seq_len, hidden_size]
    gqa_out = gqa_block(decoder_in)
    # residual_1: [seq_len, hidden_size]
    residual_1 = add(decoder_in, gqa_out)

    # 2. Mixture of Experts sub-block with residual connection
    # moe_out: [seq_len, hidden_size]
    moe_out = moe_block(residual_1)
    # decoder_out: [seq_len, hidden_size]
    decoder_out = add(residual_1, moe_out)
    return decoder_out

class DecoderBlock(Layer):
    """Encapsulates one GQA block and one MoE block."""

    def __init__(self, name="decoder_block"):
        super().__init__(name)
        self.gqa_block_layer = GQABlock(name=f"{self.name}.gqa_block")
        self.moe_block_layer = MoEBlock(name=f"{self.name}.moe_block")

    def predict(self, decoder_in):
        """Run the attention and expert sub-layers with residual connections.

        Call path: model.predict -> decoder -> decoder_block

        Args:
            decoder_in: [seq_len, hidden_size]

        Returns:
            [seq_len, hidden_size]
        """
        return decoder_block(decoder_in, self.gqa_block_layer, self.moe_block_layer)

def decoder(embed_out, decoder_blocks):
    """Pass sequence representations sequentially through all stacked decoder blocks.

    Call path: model.predict -> decoder

    Args:
        embed_out: list of list of float [seq_len, hidden_size]
        decoder_blocks: list of DecoderBlock

    Returns:
        list of list of float [seq_len, hidden_size]
    """
    # decoder_out: [seq_len, hidden_size], updated in place by each block
    decoder_out = embed_out
    for block in decoder_blocks:
        decoder_out = block(decoder_out)
    return decoder_out

class Decoder(Layer):
    """Encapsulates the sequential stack of decoder blocks."""

    def __init__(self, name="decoder"):
        super().__init__(name)
        self.decoder_blocks = [
            DecoderBlock(name=f"{self.name}.decoder_blocks.{i}")
            for i in range(NUM_DECODER_BLOCKS)
        ]

    def predict(self, embed_out):
        """Run the token representations through the whole decoder stack.

        Call path: model.predict -> decoder

        Args:
            embed_out: [seq_len, hidden_size]

        Returns:
            [seq_len, hidden_size]
        """
        return decoder(embed_out, self.decoder_blocks)

# ============================== Embedding & LM Head ==============================

def lookup(token_id, embedding_table):
    """Look up continuous embedding vector for a single token ID.

    Call path: model.predict -> embedding -> lookup

    Args:
        token_id: int
        embedding_table: list of list of float [vocab_size, hidden_size]

    Returns:
        list of float [hidden_size]: token vector
    """
    # token_vec: [hidden_size]
    token_vec = embedding_table[token_id]
    return token_vec

def embedding(input_ids, embedding_table):
    """Map a sequence of token IDs to a sequence of dense vectors via table lookup.

    Call path: model.predict -> embedding

    Args:
        input_ids: list of int [seq_len]
        embedding_table: list of list of float [vocab_size, hidden_size]

    Returns:
        list of list of float [seq_len, hidden_size]
    """
    # embed_out: [seq_len, hidden_size]
    embed_out = [lookup(token_id, embedding_table) for token_id in input_ids]
    return embed_out

class Embedding(Layer):
    """Encapsulates token embedding table and lookup."""

    def __init__(self, name="embedding"):
        super().__init__(name)
        # embedding_table: [vocab_size, hidden_size] = [12, 12]
        self.embedding_table = self.init_weights("embedding_table", VOCAB_SIZE, HIDDEN_SIZE)
        # Precompute the transpose for the tied LM head projection
        # embedding_table_T: [hidden_size, vocab_size] = [12, 12]
        self.embedding_table_T = [
            [self.embedding_table[r][c] for r in range(VOCAB_SIZE)]
            for c in range(HIDDEN_SIZE)
        ]

    def predict(self, input_ids):
        """Turn token IDs into their embedding vectors.

        Call path: model.predict -> embedding

        Args:
            input_ids: [seq_len]

        Returns:
            [seq_len, hidden_size]
        """
        return embedding(input_ids, self.embedding_table)

def logits_matmul(rms_out, embedding_table_T):
    """Project normalized hidden states to vocabulary logits: [seq_len, hidden_size] @ [hidden_size, vocab_size].

    Call path: model.predict -> lm_head -> logits_matmul

    Args:
        rms_out: list of list of float [seq_len, hidden_size]
        embedding_table_T: list of list of float [hidden_size, vocab_size]

    Returns:
        list of list of float [seq_len, vocab_size]
    """
    # all_logits: [seq_len, vocab_size]
    all_logits = [matmul(token_vec, embedding_table_T) for token_vec in rms_out]
    return all_logits

def slice_last(all_logits):
    """Extract logits corresponding to the final token position in the sequence.

    In causal autoregressive generation, each token position predicts the next token.
    To predict the continuation after the prompt, we only need the distribution
    emitted by the final token.

    Call path: model.predict -> lm_head -> slice_last

    Args:
        all_logits: list of list of float [seq_len, vocab_size]

    Returns:
        list of float [vocab_size]: logits for predicting the next token
    """
    # logits: [vocab_size]
    logits = all_logits[-1]
    return logits

def lm_head(decoder_out, gamma, embedding_table_T):
    """Compute vocabulary prediction logits for the next token from decoder output.

    Sequence:
        1. Final RMS normalization across the sequence
        2. Projection to vocabulary space using tied embedding weights
        3. Slice the last token's logits

    Call path: model.predict -> lm_head

    Args:
        decoder_out: list of list of float [seq_len, hidden_size]
        gamma: list of float [hidden_size], final RMSNorm scale
        embedding_table_T: list of list of float [hidden_size, vocab_size]

    Returns:
        list of float [vocab_size]: prediction scores across vocabulary
    """
    # rms_out: [seq_len, hidden_size]
    rms_out = rms_norm(decoder_out, gamma)
    # all_logits: [seq_len, vocab_size]
    all_logits = logits_matmul(rms_out, embedding_table_T)
    # logits: [vocab_size]
    logits = slice_last(all_logits)
    return logits

class LMHead(Layer):
    """Encapsulates final RMS normalization and projection to vocabulary logits."""

    def __init__(self, embedding_table_T=None, name="lm_head"):
        """
        Args:
            embedding_table_T: optional [hidden_size, vocab_size] matrix to tie to.
                Model always passes the transposed embedding table here. When it is
                omitted, we allocate a separate projection matrix instead, which is
                what models with untied weights do.
            name: str, hierarchical identifier for weight lookup.
        """
        super().__init__(name)
        # gamma: [hidden_size]
        self.gamma = self.init_weights("gamma", HIDDEN_SIZE)
        # Weight tying: share embedding table transpose to reduce parameters
        # embedding_table_T: [hidden_size, vocab_size]
        self.embedding_table_T = (
            embedding_table_T
            if embedding_table_T is not None
            else self.init_weights("embedding_table_T", HIDDEN_SIZE, VOCAB_SIZE)
        )

    def predict(self, decoder_out):
        """Turn the final hidden states into next-token logits.

        Call path: model.predict -> lm_head

        Args:
            decoder_out: [seq_len, hidden_size]

        Returns:
            [vocab_size]
        """
        return lm_head(decoder_out, self.gamma, self.embedding_table_T)

# ============================== Sampler ==============================

def greedy_sampler(model, input_ids):
    """Predict the next token ID greedily by taking the argmax of model logits.

    Args:
        model: Model instance
        input_ids: list of int [seq_len], current token sequence

    Returns:
        int: highest-scoring next token ID
    """
    # Forward pass through model to get logits for next token
    # logits: [vocab_size]
    logits = model.predict(input_ids)
    # Greedily pick the token ID with the maximum probability score
    # next_token_id: int
    next_token_id = argmax(logits)
    return next_token_id

# ============================== Model Class ==============================

class Model(Layer):
    """Top-level LLM architecture encapsulating embedding, decoder, and LM head.

    Contains 4,596 total parameters:
    - embedding: token table of 12 x 12 (144 params)
    - decoder: 2 blocks x 2,220 params (4,440 params)
        - gqa_block: 444 = 12 gamma + 3 groups x 96 + 144 output projection
        - moe_block: 1,776 = 12 gamma + 36 router + 3 experts x 576
    - lm_head: final RMSNorm gamma (12 params), with the output projection tied
      to the embedding table rather than adding 144 more parameters
    """

    def __init__(self, name="model"):
        super().__init__(name)
        # Seed RNG so random fallback weights are reproducible across runs.
        # Only matters when WEIGHTS_PATH is None or the file is missing.
        _rng.seed(42)
        # embedding.embedding_table: [vocab_size, hidden_size]
        self.embedding = Embedding(name="embedding")
        self.decoder = Decoder(name="decoder")
        # Tie the LM head projection matrix to the embedding table transpose
        # embedding_table_T: [hidden_size, vocab_size]
        self.lm_head = LMHead(self.embedding.embedding_table_T, name="lm_head")

    def predict(self, input_ids):
        """Execute complete forward pass to obtain logits for the next token.

        Pipeline:
            input_ids [seq_len]
              -> Embedding -> [seq_len, hidden_size]
              -> Decoder   -> [seq_len, hidden_size]
              -> LM Head   -> [vocab_size]

        Args:
            input_ids: list of int [seq_len]

        Returns:
            list of float [vocab_size]: next-token prediction logits
        """
        # embed_out: [seq_len, hidden_size]
        embed_out = self.embedding(input_ids)
        # decoder_out: [seq_len, hidden_size]
        decoder_out = self.decoder(embed_out)
        # logits: [vocab_size]
        logits = self.lm_head(decoder_out)
        return logits

    def generate(self, input_ids, max_new_tokens=MAX_NEW_TOKENS):
        """Autoregressively generate next tokens one by one until EOS or max length.

        Every iteration re-runs the whole forward pass over the entire sequence.
        Production engines avoid that with a KV cache, which stores the keys and
        values already computed for earlier tokens. We skip it here: caching adds
        bookkeeping that has nothing to do with the architecture itself.

        Args:
            input_ids: list of int [seq_len], prompt token IDs
            max_new_tokens: int, maximum number of new tokens to emit

        Returns:
            list of int [total_seq_len]: prompt plus generated token IDs
        """
        for _ in range(max_new_tokens):
            # Predict one token from everything generated so far
            # next_token_id: int
            next_token_id = greedy_sampler(self, input_ids)
            # Append it, so the next pass sees a sequence one token longer
            # input_ids: [seq_len + 1]
            input_ids = input_ids + [next_token_id]
            # Stop immediately when end-of-sequence token is generated
            if next_token_id == EOS_TOKEN_ID:
                break
        # input_ids: [total_seq_len]
        return input_ids

# ============================== Pipeline & Main ==============================

def pipeline(prompt, tokenizer=None, model=None, max_new_tokens=MAX_NEW_TOKENS):
    """Convenience end-to-end pipeline: takes a text prompt and returns generated text.

    Args:
        prompt: str, input text prompt (e.g. "What is 1+1?")
        tokenizer: optional Tokenizer instance (creates default if omitted)
        model: optional Model instance (creates default if omitted)
        max_new_tokens: int, maximum new tokens to generate

    Returns:
        str: full generated text string
    """
    if tokenizer is None:
        tokenizer = Tokenizer()
    if model is None:
        model = Model()

    # 1. Encode text prompt to integer token IDs
    # input_ids: [seq_len]
    input_ids = tokenizer.encode(prompt)

    # 2. Autoregressively generate new token IDs
    # output_ids: [total_seq_len]
    output_ids = model.generate(input_ids, max_new_tokens=max_new_tokens)

    # 3. Decode token IDs back to human-readable string
    # output_text: str
    output_text = tokenizer.decode(output_ids)
    return output_text

def main():
    """Run the model on the one prompt it knows and print the generated output.

    Prints "What is 1+1? It's 2.<eos>" when the trained weights are in place.
    """
    prompt = "What is 1+1?"
    output = pipeline(prompt)
    print(output)

if __name__ == "__main__":
    main()
