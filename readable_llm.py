"""readable-llm: The Anatomy of an LLM in a single Python file.

Companion code for the article "The Anatomy of an LLM":
https://haifengjin.com/the-anatomy-of-an-llm/

Code Organization:
- Classes organize weights and layer hierarchy for different modules.
- Standalone functions perform the complex compute and tensor operations.
"""

import math
import random

# ============================== Model Architecture Constants ==============================

VOCAB_SIZE = 2000
HIDDEN_SIZE = 12
NUM_DECODER_BLOCKS = 2
NUM_GROUPS = 3
Q_HEADS = 2
D_HEAD = 2
HEAD_DIM = Q_HEADS * D_HEAD  # 4
NUM_EXPERTS = 3
TOP_K = 2
INTER_SIZE = 16

MAX_NEW_TOKENS = 128
EOS_TOKEN_ID = 2
EPS = 1e-6

# ============================== Weight Initialization Helper ==============================

_rng = random.Random(42)

def _init_weights(*shape, val=None, scale=0.02):
    """
    Args:
        *shape: tuple of int
        val: float
        scale: float

    Returns:
        list
    """
    if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
        # shape: tuple of int
        shape = tuple(shape[0])

    if len(shape) == 1:
        # fill_val: float
        fill_val = 1.0 if val is None else val
        # weights_1d: [shape[0]]
        weights_1d = [fill_val] * shape[0]
        return weights_1d
    elif len(shape) == 2:
        # rows: int
        # cols: int
        rows, cols = shape
        # weights_2d: [rows, cols]
        weights_2d = [[_rng.uniform(-scale, scale) for _ in range(cols)] for _ in range(rows)]
        return weights_2d
    elif len(shape) == 3:
        # heads: int
        # rows: int
        # cols: int
        heads, rows, cols = shape
        # weights_3d: [heads, rows, cols]
        weights_3d = [
            [[_rng.uniform(-scale, scale) for _ in range(cols)] for _ in range(rows)]
            for _ in range(heads)
        ]
        return weights_3d
    raise ValueError(f"Unsupported shape: {shape}")

init_weights = _init_weights
_make_matrix = _init_weights

# ============================== Vocabulary & Tokenizer ==============================

vocab = {
    "<pad>": 0,
    "<bos>": 1,
    "<eos>": 2,
    "+": 10,
    "1": 16,
    " 2": 17,
    "2": 18,
    "?": 30,
    "is": 318,
    " 1": 352,
    "What": 1867,
}

def split_tokens(text, vocab_dict=vocab):
    """
    Args:
        text: str
        vocab_dict: dict

    Returns:
        list of str
    """
    # tokens: list of str
    tokens = []
    # i: int
    i = 0
    while i < len(text):
        # matches: list of str
        matches = [t for t in vocab_dict if text[i:].startswith(t)]
        if matches:
            # best: str
            best = max(matches, key=len)
            tokens.append(best)
            i += len(best)
        elif text[i] == " ":
            i += 1
        else:
            tokens.append(text[i])
            i += 1
    return tokens

class Tokenizer:
    def __init__(self, vocab=vocab):
        """
        Args:
            vocab: dict
        """
        self.vocab = vocab
        self.inv_vocab = {token_id: token for token, token_id in vocab.items()}

    def encode(self, text):
        """
        Args:
            text: str

        Returns:
            [seq_len]
        """
        # tokens: list of str
        tokens = split_tokens(text, self.vocab)
        # input_ids: [seq_len]
        input_ids = [self.vocab[token] for token in tokens]
        return input_ids

    def decode(self, token_ids):
        """
        Args:
            token_ids: [seq_len]

        Returns:
            str
        """
        # decoded_str: str
        decoded_str = "".join(self.inv_vocab.get(token_id, f"<{token_id}>") for token_id in token_ids)
        return decoded_str

# ============================== Base Neural Network Class ==============================

class Layer:
    """Base class for all neural network modules."""

    def __call__(self, *args, **kwargs):
        """
        Args:
            *args: tuple
            **kwargs: dict

        Returns:
            any
        """
        return self.predict(*args, **kwargs)

    def predict(self, *args, **kwargs):
        """
        Args:
            *args: tuple
            **kwargs: dict
        """
        raise NotImplementedError

# ============================== Basic Operations ==============================

def softmax(logits):
    """
    Args:
        logits: [n]

    Returns:
        [n]
    """
    # max_val: float
    max_val = max(logits)
    # exps: [n]
    exps = [math.exp(x - max_val) for x in logits]
    # sum_exps: float
    sum_exps = sum(exps)
    # probs: [n]
    probs = [x / sum_exps for x in exps]
    return probs

def argmax(logits):
    """
    Args:
        logits: [vocab_size]

    Returns:
        int
    """
    # max_idx: int
    max_idx = 0
    for i in range(len(logits)):
        if logits[i] > logits[max_idx]:
            max_idx = i
    return max_idx

def silu(x):
    """
    Args:
        x: float

    Returns:
        float
    """
    return x / (1.0 + math.exp(-x))

def matmul(vec, matrix):
    """
    Args:
        vec: [in_dim]
        matrix: [in_dim, out_dim]

    Returns:
        [out_dim]
    """
    # in_dim: int
    in_dim = len(vec)
    # out_dim: int
    out_dim = len(matrix[0])
    # output: [out_dim]
    output = []
    for col in range(out_dim):
        # dot_product: float
        dot_product = sum(vec[k] * matrix[k][col] for k in range(in_dim))
        output.append(dot_product)
    return output

def add(tensor_a, tensor_b):
    """
    Args:
        tensor_a: [seq_len, hidden_size]
        tensor_b: [seq_len, hidden_size]

    Returns:
        [seq_len, hidden_size]
    """
    # seq_len: int
    seq_len = len(tensor_a)
    # hidden_size: int
    hidden_size = len(tensor_a[0])

    # output: [seq_len, hidden_size]
    output = [[0.0] * hidden_size for _ in range(seq_len)]
    for i in range(seq_len):
        for j in range(hidden_size):
            output[i][j] = float(tensor_a[i][j]) + float(tensor_b[i][j])
    return output

# ============================== RMSNorm ==============================

def norm_token(token_vec, gamma):
    """
    model.predict -> ... -> rms_norm -> norm_token

    Args:
        token_vec: [hidden_size]
        gamma: [hidden_size]

    Returns:
        [hidden_size]
    """
    # sum_of_squares: float
    sum_of_squares = 0.0
    for v in token_vec:
        sum_of_squares += v ** 2
    # rms: float
    rms = (sum_of_squares / len(token_vec) + EPS) ** 0.5

    # output: [hidden_size]
    output = []
    for i in range(len(token_vec)):
        output.append(token_vec[i] / rms * gamma[i])
    return output

def rms_norm(tensor, gamma):
    """
    model.predict -> (lm_head | gqa_block | moe_block) -> rms_norm

    Args:
        tensor: [seq_len, hidden_size]
        gamma: [hidden_size]

    Returns:
        [seq_len, hidden_size]
    """
    # rms_out: [seq_len, hidden_size]
    rms_out = [norm_token(token_vec, gamma) for token_vec in tensor]
    return rms_out

class RMSNorm(Layer):
    """Encapsulates RMS normalization scale weights."""

    def __init__(self):
        self.gamma = _init_weights(HIDDEN_SIZE)

    def predict(self, tensor_or_vec):
        """
        model.predict -> (lm_head | gqa_block | moe_block) -> rms_norm

        Args:
            tensor_or_vec: [seq_len, hidden_size] or [hidden_size]

        Returns:
            [seq_len, hidden_size] or [hidden_size]
        """
        if isinstance(tensor_or_vec[0], list):
            return rms_norm(tensor_or_vec, self.gamma)
        return norm_token(tensor_or_vec, self.gamma)

RmsNorm = RMSNorm

# ============================== RoPE ==============================

def rope_pair(x0, x1, pos, i, d_head):
    """
    model.predict -> decoder -> decoder_block -> gqa_block -> gqa -> group_0 -> rope -> rope_token -> rope_pair

    Args:
        x0: float
        x1: float
        pos: int
        i: int
        d_head: int

    Returns:
        (float, float)
    """
    # freq: float
    freq = 1.0 / (10000 ** (i / d_head))
    # angle: float
    angle = pos * freq
    # cos_val: float
    cos_val = math.cos(angle)
    # sin_val: float
    sin_val = math.sin(angle)
    return x0 * cos_val - x1 * sin_val, x0 * sin_val + x1 * cos_val

def rope_token(token_vec, pos):
    """
    model.predict -> decoder -> decoder_block -> gqa_block -> gqa -> group_0 -> rope -> rope_token

    Args:
        token_vec: [d_head]
        pos: int

    Returns:
        [d_head]
    """
    # d_head: int
    d_head = len(token_vec)
    # output: [d_head]
    output = []
    for i in range(0, d_head, 2):
        # r0: float
        # r1: float
        r0, r1 = rope_pair(token_vec[i], token_vec[i + 1], pos, i, d_head)
        output.append(r0)
        output.append(r1)
    return output

def rope_2d(x):
    """
    model.predict -> decoder -> decoder_block -> gqa_block -> gqa -> group_0 -> rope -> rope_2d

    Args:
        x: [seq_len, d_head]

    Returns:
        [seq_len, d_head]
    """
    # output: [seq_len, d_head]
    output = [rope_token(token_vec, pos) for pos, token_vec in enumerate(x)]
    return output

def rope_3d(x):
    """
    model.predict -> decoder -> decoder_block -> gqa_block -> gqa -> group_0 -> rope -> rope_3d

    Args:
        x: [q_heads, seq_len, d_head]

    Returns:
        [q_heads, seq_len, d_head]
    """
    # output: [q_heads, seq_len, d_head]
    output = [rope_2d(head) for head in x]
    return output

def rope(x):
    """
    model.predict -> decoder -> decoder_block -> gqa_block -> gqa -> group_0 -> rope

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
    """
    model.predict -> decoder -> decoder_block -> gqa_block -> gqa -> group_0 -> attention_head -> single_attention_head -> attention_token

    Args:
        q_token: [d_head]
        k_T: [d_head, seq_len]
        v: [seq_len, d_head]
        i: int

    Returns:
        [d_head]
    """
    # d_head: int
    d_head = len(q_token)

    # raw_scores: [seq_len]
    raw_scores = matmul(q_token, k_T)

    # scaled_scores: [seq_len]
    scaled_scores = [s / (d_head ** 0.5) for s in raw_scores]

    # masked_scores: [i + 1]
    masked_scores = [scaled_scores[j] for j in range(i + 1)]

    # weights: [i + 1]
    weights = softmax(masked_scores)

    # weights: [seq_len]
    weights = weights + [0.0] * (len(raw_scores) - len(weights))

    # token_out: [d_head]
    token_out = matmul(weights, v)
    return token_out

def single_attention_head(single_q, k, v):
    """
    model.predict -> decoder -> decoder_block -> gqa_block -> gqa -> group_0 -> attention_head -> single_attention_head

    Args:
        single_q: [seq_len, d_head]
        k: [seq_len, d_head]
        v: [seq_len, d_head]

    Returns:
        [seq_len, d_head]
    """
    # seq_len: int
    seq_len = len(k)
    # d_head: int
    d_head = len(k[0])

    # k_T: [d_head, seq_len]
    k_T = [[k[row][col] for row in range(seq_len)] for col in range(d_head)]

    # head_out: [seq_len, d_head]
    head_out = []
    for i, q_token in enumerate(single_q):
        # token_out: [d_head]
        token_out = attention_token(q_token, k_T, v, i)
        head_out.append(token_out)
    return head_out

def attention_head(q, k, v):
    """
    model.predict -> decoder -> decoder_block -> gqa_block -> gqa -> group_0 -> attention_head

    Args:
        q: [q_heads, seq_len, d_head]
        k: [seq_len, d_head]
        v: [seq_len, d_head]

    Returns:
        [seq_len, head_dim]
    """
    # q_heads: int
    q_heads = len(q)
    # seq_len: int
    seq_len = len(k)

    # head_outs: [q_heads, seq_len, d_head]
    head_outs = [single_attention_head(single_q, k, v) for single_q in q]

    # group_out: [seq_len, head_dim]
    group_out = []
    for t in range(seq_len):
        # row: [head_dim]
        row = []
        for h in range(q_heads):
            row.extend(head_outs[h][t])
        group_out.append(row)
    return group_out

# ============================== Grouped-Query Attention (GQA) ==============================

def group_0(rms_out, w_q, w_k, w_v):
    """
    model.predict -> decoder -> decoder_block -> gqa_block -> gqa -> group_0

    Args:
        rms_out: [seq_len, hidden_size]
        w_q: [q_heads, hidden_size, d_head]
        w_k: [hidden_size, d_head]
        w_v: [hidden_size, d_head]

    Returns:
        [seq_len, head_dim]
    """
    # q: [q_heads, seq_len, d_head]
    q = [[matmul(token_vec, w) for token_vec in rms_out] for w in w_q]
    # k: [seq_len, d_head]
    k = [matmul(token_vec, w_k) for token_vec in rms_out]
    # v: [seq_len, d_head]
    v = [matmul(token_vec, w_v) for token_vec in rms_out]

    # q: [q_heads, seq_len, d_head]
    q = rope(q)
    # k: [seq_len, d_head]
    k = rope(k)

    # group_0_out: [seq_len, head_dim]
    group_0_out = attention_head(q, k, v)
    return group_0_out

class Group(Layer):
    """Encapsulates Q, K, and V projection weights for a single attention group."""

    def __init__(self):
        self.w_q = _init_weights(Q_HEADS, HIDDEN_SIZE, D_HEAD)
        self.w_k = _init_weights(HIDDEN_SIZE, D_HEAD)
        self.w_v = _init_weights(HIDDEN_SIZE, D_HEAD)

    def predict(self, rms_out):
        """
        model.predict -> decoder -> decoder_block -> gqa_block -> gqa -> group_0

        Args:
            rms_out: [seq_len, hidden_size]

        Returns:
            [seq_len, head_dim]
        """
        return group_0(rms_out, self.w_q, self.w_k, self.w_v)

def gqa(rms_out, groups):
    """
    model.predict -> decoder -> decoder_block -> gqa_block -> gqa

    Args:
        rms_out: [seq_len, hidden_size]
        groups: list of Group

    Returns:
        [seq_len, hidden_size]
    """
    # seq_len: int
    seq_len = len(rms_out)

    # group_outs: [num_groups, seq_len, head_dim]
    group_outs = [group(rms_out) for group in groups]

    # gqa_out: [seq_len, hidden_size]
    gqa_out = []
    for t in range(seq_len):
        # row: [hidden_size]
        row = []
        for out in group_outs:
            row.extend(out[t])
        gqa_out.append(row)
    return gqa_out

class GQA(Layer):
    """Encapsulates grouped-query attention across all groups."""

    def __init__(self):
        self.groups = [Group() for _ in range(NUM_GROUPS)]

    def predict(self, rms_out):
        """
        model.predict -> decoder -> decoder_block -> gqa_block -> gqa

        Args:
            rms_out: [seq_len, hidden_size]

        Returns:
            [seq_len, hidden_size]
        """
        return gqa(rms_out, self.groups)

def out_matmul(gqa_out, w_matmul):
    """
    model.predict -> decoder -> decoder_block -> gqa_block -> out_matmul

    Args:
        gqa_out: [seq_len, hidden_size]
        w_matmul: [hidden_size, hidden_size]

    Returns:
        [seq_len, hidden_size]
    """
    # gqa_block_out: [seq_len, hidden_size]
    gqa_block_out = [matmul(token_vec, w_matmul) for token_vec in gqa_out]
    return gqa_block_out

class OutMatmul(Layer):
    """Encapsulates output projection matrix for GQA."""

    def __init__(self):
        self.w_matmul = _init_weights(HIDDEN_SIZE, HIDDEN_SIZE)

    def predict(self, gqa_out):
        """
        model.predict -> decoder -> decoder_block -> gqa_block -> out_matmul

        Args:
            gqa_out: [seq_len, hidden_size]

        Returns:
            [seq_len, hidden_size]
        """
        return out_matmul(gqa_out, self.w_matmul)

def gqa_block(gqa_block_in, rms_norm, gqa, out_matmul):
    """
    model.predict -> decoder -> decoder_block -> gqa_block

    Args:
        gqa_block_in: [seq_len, hidden_size]
        rms_norm: RMSNorm
        gqa: GQA
        out_matmul: OutMatmul

    Returns:
        [seq_len, hidden_size]
    """
    # rms_out: [seq_len, hidden_size]
    rms_out = rms_norm(gqa_block_in)

    # gqa_out: [seq_len, hidden_size]
    gqa_out = gqa(rms_out)

    # gqa_block_out: [seq_len, hidden_size]
    gqa_block_out = out_matmul(gqa_out)
    return gqa_block_out

class GQABlock(Layer):
    """Encapsulates normalization, grouped-query attention, and output projection."""

    def __init__(self):
        self.rms_norm = RMSNorm()
        self.gqa = GQA()
        self.out_matmul = OutMatmul()

    def predict(self, gqa_block_in):
        """
        model.predict -> decoder -> decoder_block -> gqa_block

        Args:
            gqa_block_in: [seq_len, hidden_size]

        Returns:
            [seq_len, hidden_size]
        """
        return gqa_block(gqa_block_in, self.rms_norm, self.gqa, self.out_matmul)

# ============================== Mixture of Experts (MoE) ==============================

def expert_token(token_vec, w_gate, w_up, w_down):
    """
    model.predict -> decoder -> decoder_block -> moe_block -> moe -> expert -> expert_token

    Args:
        token_vec: [hidden_size]
        w_gate: [hidden_size, inter_size]
        w_up: [hidden_size, inter_size]
        w_down: [inter_size, hidden_size]

    Returns:
        [hidden_size]
    """
    # x_gate: [inter_size]
    x_gate = matmul(token_vec, w_gate)

    # x_up: [inter_size]
    x_up = matmul(token_vec, w_up)

    # x_act: [inter_size]
    x_act = [silu(x) for x in x_gate]

    # inter_size: int
    inter_size = len(x_act)
    # x_inter: [inter_size]
    x_inter = [x_act[i] * x_up[i] for i in range(inter_size)]

    # x_down: [hidden_size]
    x_down = matmul(x_inter, w_down)
    return x_down

def expert(tensor, w_gate, w_up, w_down):
    """
    model.predict -> decoder -> decoder_block -> moe_block -> moe -> expert

    Args:
        tensor: [seq_len, hidden_size]
        w_gate: [hidden_size, inter_size]
        w_up: [hidden_size, inter_size]
        w_down: [inter_size, hidden_size]

    Returns:
        [seq_len, hidden_size]
    """
    # output: [seq_len, hidden_size]
    output = [expert_token(token_vec, w_gate, w_up, w_down) for token_vec in tensor]
    return output

class Expert(Layer):
    """Encapsulates SwiGLU projection weights for a single expert."""

    def __init__(self):
        self.w_gate = _init_weights(HIDDEN_SIZE, INTER_SIZE)
        self.w_up = _init_weights(HIDDEN_SIZE, INTER_SIZE)
        self.w_down = _init_weights(INTER_SIZE, HIDDEN_SIZE)

    def predict(self, token_vec_or_tensor):
        """
        model.predict -> decoder -> decoder_block -> moe_block -> moe -> expert

        Args:
            token_vec_or_tensor: [seq_len, hidden_size] or [hidden_size]

        Returns:
            [seq_len, hidden_size] or [hidden_size]
        """
        if isinstance(token_vec_or_tensor[0], list):
            return expert(token_vec_or_tensor, self.w_gate, self.w_up, self.w_down)
        return expert_token(token_vec_or_tensor, self.w_gate, self.w_up, self.w_down)

def route_token(token_vec, w_router):
    """
    model.predict -> decoder -> decoder_block -> moe_block -> router -> route_token

    Args:
        token_vec: [hidden_size]
        w_router: [hidden_size, num_experts]

    Returns:
        [num_experts]
    """
    # logits: [num_experts]
    logits = matmul(token_vec, w_router)

    # num_experts: int
    num_experts = len(logits)

    # indices: [num_experts]
    indices = [i for i in range(num_experts)]

    # sorted_indices: [num_experts]
    sorted_indices = sorted(indices, key=lambda i: logits[i], reverse=True)

    # top_indices: [top_k]
    top_indices = sorted_indices[:TOP_K]

    # top_logits: [top_k]
    top_logits = [logits[i] for i in top_indices]

    # top_probs: [top_k]
    top_probs = softmax(top_logits)

    # top_weights: [num_experts]
    top_weights = [0.0] * num_experts
    for k in range(TOP_K):
        top_weights[top_indices[k]] = top_probs[k]
    return top_weights

def router(rms_out, w_router):
    """
    model.predict -> decoder -> decoder_block -> moe_block -> router

    Args:
        rms_out: [seq_len, hidden_size]
        w_router: [hidden_size, num_experts]

    Returns:
        [seq_len, num_experts]
    """
    # top_weights: [seq_len, num_experts]
    top_weights = [route_token(token_vec, w_router) for token_vec in rms_out]
    return top_weights

class Router(Layer):
    """Encapsulates routing weights to select top-k experts."""

    def __init__(self):
        self.w_router = _init_weights(HIDDEN_SIZE, NUM_EXPERTS)

    def predict(self, rms_out):
        """
        model.predict -> decoder -> decoder_block -> moe_block -> router

        Args:
            rms_out: [seq_len, hidden_size]

        Returns:
            [seq_len, num_experts]
        """
        return router(rms_out, self.w_router)

def moe_token(token_vec, top_weights, experts):
    """
    model.predict -> decoder -> decoder_block -> moe_block -> moe -> moe_token

    Args:
        token_vec: [hidden_size]
        top_weights: [num_experts]
        experts: list of Expert

    Returns:
        [hidden_size]
    """
    # hidden_size: int
    hidden_size = len(token_vec)

    # moe_out: [hidden_size]
    moe_out = [0.0] * hidden_size
    for i in range(len(experts)):
        # expert_out: [hidden_size]
        expert_out = experts[i](token_vec)
        for j in range(hidden_size):
            moe_out[j] += top_weights[i] * expert_out[j]
    return moe_out

def moe(rms_out, top_weights, experts):
    """
    model.predict -> decoder -> decoder_block -> moe_block -> moe

    Args:
        rms_out: [seq_len, hidden_size]
        top_weights: [seq_len, num_experts]
        experts: list of Expert

    Returns:
        [seq_len, hidden_size]
    """
    # moe_out: [seq_len, hidden_size]
    moe_out = [
        moe_token(token_vec, weights, experts)
        for token_vec, weights in zip(rms_out, top_weights)
    ]
    return moe_out

class MoE(Layer):
    """Encapsulates the collection of experts and weighted aggregation."""

    def __init__(self):
        self.experts = [Expert() for _ in range(NUM_EXPERTS)]

    def predict(self, rms_out, top_weights):
        """
        model.predict -> decoder -> decoder_block -> moe_block -> moe

        Args:
            rms_out: [seq_len, hidden_size]
            top_weights: [seq_len, num_experts]

        Returns:
            [seq_len, hidden_size]
        """
        return moe(rms_out, top_weights, self.experts)

def moe_block(moe_in, rms_norm, router, moe):
    """
    model.predict -> decoder -> decoder_block -> moe_block

    Args:
        moe_in: [seq_len, hidden_size]
        rms_norm: RMSNorm
        router: Router
        moe: MoE

    Returns:
        [seq_len, hidden_size]
    """
    # rms_out: [seq_len, hidden_size]
    rms_out = rms_norm(moe_in)

    # top_weights: [seq_len, num_experts]
    top_weights = router(rms_out)

    # moe_out: [seq_len, hidden_size]
    moe_out = moe(rms_out, top_weights)
    return moe_out

class MoEBlock(Layer):
    """Encapsulates normalization, router, and mixture of experts."""

    def __init__(self):
        self.rms_norm = RMSNorm()
        self.router = Router()
        self.moe = MoE()

    def predict(self, moe_in):
        """
        model.predict -> decoder -> decoder_block -> moe_block

        Args:
            moe_in: [seq_len, hidden_size]

        Returns:
            [seq_len, hidden_size]
        """
        return moe_block(moe_in, self.rms_norm, self.router, self.moe)

# ============================== Decoder ==============================

def decoder_block(decoder_in, gqa_block, moe_block):
    """
    model.predict -> decoder -> decoder_block

    Args:
        decoder_in: [seq_len, hidden_size]
        gqa_block: GQABlock
        moe_block: MoEBlock

    Returns:
        [seq_len, hidden_size]
    """
    # gqa_out: [seq_len, hidden_size]
    gqa_out = gqa_block(decoder_in)
    # residual_1: [seq_len, hidden_size]
    residual_1 = add(decoder_in, gqa_out)

    # moe_out: [seq_len, hidden_size]
    moe_out = moe_block(residual_1)

    # decoder_out: [seq_len, hidden_size]
    decoder_out = add(residual_1, moe_out)
    return decoder_out

class DecoderBlock(Layer):
    """Encapsulates one GQA block and one MoE block."""

    def __init__(self):
        self.gqa_block_layer = GQABlock()
        self.moe_block_layer = MoEBlock()

    def predict(self, decoder_in):
        """
        model.predict -> decoder -> decoder_block

        Args:
            decoder_in: [seq_len, hidden_size]

        Returns:
            [seq_len, hidden_size]
        """
        return decoder_block(decoder_in, self.gqa_block_layer, self.moe_block_layer)

def decoder(embed_out, decoder_blocks):
    """
    model.predict -> decoder

    Args:
        embed_out: [seq_len, hidden_size]
        decoder_blocks: list of DecoderBlock

    Returns:
        [seq_len, hidden_size]
    """
    # decoder_out: [seq_len, hidden_size]
    decoder_out = embed_out
    for block in decoder_blocks:
        decoder_out = block(decoder_out)
    return decoder_out

class Decoder(Layer):
    """Encapsulates the sequential stack of decoder blocks."""

    def __init__(self):
        self.decoder_blocks = [DecoderBlock() for _ in range(NUM_DECODER_BLOCKS)]

    def predict(self, embed_out):
        """
        model.predict -> decoder

        Args:
            embed_out: [seq_len, hidden_size]

        Returns:
            [seq_len, hidden_size]
        """
        return decoder(embed_out, self.decoder_blocks)

# ============================== Embedding & LM Head ==============================

def lookup(token_id, embedding_table):
    """
    model.predict -> embedding -> lookup

    Args:
        token_id: int
        embedding_table: [vocab_size, hidden_size]

    Returns:
        [hidden_size]
    """
    # token_vec: [hidden_size]
    token_vec = embedding_table[token_id]
    return token_vec

def embedding(input_ids, embedding_table):
    """
    model.predict -> embedding

    Args:
        input_ids: [seq_len]
        embedding_table: [vocab_size, hidden_size]

    Returns:
        [seq_len, hidden_size]
    """
    # embed_out: [seq_len, hidden_size]
    embed_out = [lookup(token_id, embedding_table) for token_id in input_ids]
    return embed_out

class Embedding(Layer):
    """Encapsulates token embedding table and lookup."""

    def __init__(self):
        self.embedding_table = _init_weights(VOCAB_SIZE, HIDDEN_SIZE)
        self.embedding_table_T = [
            [self.embedding_table[r][c] for r in range(VOCAB_SIZE)]
            for c in range(HIDDEN_SIZE)
        ]

    def predict(self, input_ids):
        """
        model.predict -> embedding

        Args:
            input_ids: [seq_len]

        Returns:
            [seq_len, hidden_size]
        """
        return embedding(input_ids, self.embedding_table)

def logits_matmul(rms_out, embedding_table_T):
    """
    model.predict -> lm_head -> logits_matmul

    Args:
        rms_out: [seq_len, hidden_size]
        embedding_table_T: [hidden_size, vocab_size]

    Returns:
        [seq_len, vocab_size]
    """
    # all_logits: [seq_len, vocab_size]
    all_logits = [matmul(token_vec, embedding_table_T) for token_vec in rms_out]
    return all_logits

def slice_last(all_logits):
    """
    model.predict -> lm_head -> slice_last

    Args:
        all_logits: [seq_len, vocab_size]

    Returns:
        [vocab_size]
    """
    # logits: [vocab_size]
    logits = all_logits[-1]
    return logits

def lm_head(decoder_out, gamma, embedding_table_T):
    """
    model.predict -> lm_head

    Args:
        decoder_out: [seq_len, hidden_size]
        gamma: [hidden_size]
        embedding_table_T: [hidden_size, vocab_size]

    Returns:
        [vocab_size]
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

    def __init__(self, embedding_table_T=None):
        """
        Args:
            embedding_table_T: [hidden_size, vocab_size]
        """
        self.gamma = _init_weights(HIDDEN_SIZE)
        self.embedding_table_T = (
            embedding_table_T
            if embedding_table_T is not None
            else _init_weights(HIDDEN_SIZE, VOCAB_SIZE)
        )

    def predict(self, decoder_out):
        """
        model.predict -> lm_head

        Args:
            decoder_out: [seq_len, hidden_size]

        Returns:
            [vocab_size]
        """
        return lm_head(decoder_out, self.gamma, self.embedding_table_T)

LmHead = LMHead

# ============================== Sampler ==============================

def greedy_sampler(model, input_ids):
    """
    Args:
        model: Model
        input_ids: [seq_len]

    Returns:
        int
    """
    # logits: [vocab_size]
    logits = model.predict(input_ids)
    # next_token_id: int
    next_token_id = argmax(logits)
    return next_token_id

# ============================== Model Class ==============================

class Model(Layer):
    """Top-level LLM architecture encapsulating embedding, decoder, and LM head."""

    def __init__(self):
        _rng.seed(42)
        self.embedding = Embedding()
        self.decoder = Decoder()
        self.lm_head = LMHead(self.embedding.embedding_table_T)

    def predict(self, input_ids):
        """
        Args:
            input_ids: [seq_len]

        Returns:
            [vocab_size]
        """
        # embed_out: [seq_len, hidden_size]
        embed_out = self.embedding(input_ids)
        # decoder_out: [seq_len, hidden_size]
        decoder_out = self.decoder(embed_out)
        # logits: [vocab_size]
        logits = self.lm_head(decoder_out)
        return logits

    def generate(self, input_ids, max_new_tokens=MAX_NEW_TOKENS):
        """
        Args:
            input_ids: [seq_len]
            max_new_tokens: int

        Returns:
            [total_seq_len]
        """
        for _ in range(max_new_tokens):
            # next_token_id: int
            next_token_id = greedy_sampler(self, input_ids)
            # input_ids: [seq_len + 1]
            input_ids = input_ids + [next_token_id]
            if next_token_id == EOS_TOKEN_ID:
                break
        return input_ids

# ============================== Pipeline & Main ==============================

def pipeline(prompt, tokenizer=None, model=None, max_new_tokens=10):
    """
    Args:
        prompt: str
        tokenizer: Tokenizer
        model: Model
        max_new_tokens: int

    Returns:
        str
    """
    if tokenizer is None:
        tokenizer = Tokenizer()
    if model is None:
        model = Model()
    # input_ids: [seq_len]
    input_ids = tokenizer.encode(prompt)
    # output_ids: [total_seq_len]
    output_ids = model.generate(input_ids, max_new_tokens=max_new_tokens)
    # output_text: str
    output_text = tokenizer.decode(output_ids)
    return output_text

def main():
    # output: str
    output = pipeline("What is 1+1?")
    print(output)

if __name__ == "__main__":
    main()
