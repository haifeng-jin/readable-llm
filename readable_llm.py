"""readable-llm: The Anatomy of an LLM in a single Python file.

A clean, educational, and end-to-end runnable implementation of a modern
Large Language Model in pure Python with zero external dependencies.
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

def _make_matrix(rows, cols, scale=0.02):
    return [[_rng.uniform(-scale, scale) for _ in range(cols)] for _ in range(rows)]

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
    # Chops the input string into individual tokens from vocab
    tokens = []
    i = 0
    while i < len(text):
        matches = [t for t in vocab_dict if text[i:].startswith(t)]
        if matches:
            best = max(matches, key=len)
            tokens.append(best)
            i += len(best)
        elif text[i] == " ":
            # Skip unmapped standalone whitespace if not part of a token
            i += 1
        else:
            tokens.append(text[i])
            i += 1
    return tokens

class Tokenizer:
    def __init__(self, vocab=vocab):
        self.vocab = vocab
        self.inv_vocab = {token_id: token for token, token_id in vocab.items()}

    def encode(self, text):
        tokens = split_tokens(text, self.vocab)
        # input_ids: [seq_len]
        input_ids = [self.vocab[token] for token in tokens]
        return input_ids

    def decode(self, token_ids):
        # token_ids: [seq_len]
        return "".join(self.inv_vocab.get(token_id, f"<{token_id}>") for token_id in token_ids)

# ============================== Base Neural Network Class ==============================

class Layer:
    """Base class for all neural network modules.
    
    Every layer implements .predict() as its forward pass method.
    Calling an instance directly delegates to .predict().
    """

    def predict(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

# ============================== Math & Basic Tensor Ops ==============================

def argmax(logits):
    # logits: [vocab_size]
    # max_idx: int
    max_idx = 0
    for i in range(len(logits)):
        if logits[i] > logits[max_idx]:
            max_idx = i
    return max_idx

def softmax(scores):
    # Pure Python softmax on a list of numerical scores
    max_score = max(scores)
    exp_scores = [math.exp(s - max_score) for s in scores]
    sum_exp = sum(exp_scores)
    return [s / sum_exp for s in exp_scores]

def silu(x):
    return x / (1.0 + math.exp(-x))

def matmul(vec, matrix):
    # vec: [in_dim]
    # matrix: [in_dim, out_dim]
    in_dim = len(vec)
    out_dim = len(matrix[0])
    output = []
    for col in range(out_dim):
        dot_product = sum(vec[k] * matrix[k][col] for k in range(in_dim))
        output.append(dot_product)
    return output

def add(tensor_a, tensor_b):
    # tensor_a: [seq_len, hidden_size]
    # tensor_b: [seq_len, hidden_size]
    seq_len = len(tensor_a)
    hidden_size = len(tensor_a[0])

    # output: [seq_len, hidden_size]
    output = [[0.0] * hidden_size for _ in range(seq_len)]
    for i in range(seq_len):
        for j in range(hidden_size):
            output[i][j] = float(tensor_a[i][j]) + float(tensor_b[i][j])
    return output

# ============================== RMSNorm ==============================

def norm_token(token_vec, gamma):
    # token_vec: [hidden_size]
    sum_of_squares = 0.0
    for v in token_vec:
        sum_of_squares += v ** 2
    rms = (sum_of_squares / len(token_vec)) ** 0.5

    # output: [hidden_size]
    output = []
    for i in range(len(token_vec)):
        output.append(token_vec[i] / (rms + EPS) * gamma[i])
    return output

def rms_norm(tensor, gamma):
    # tensor: [seq_len, hidden_size]
    # rms_out: [seq_len, hidden_size]
    rms_out = [norm_token(token_vec, gamma) for token_vec in tensor]
    return rms_out

class RMSNorm(Layer):
    """Encapsulates RMS normalization scale weights."""

    def __init__(self):
        self.gamma = [1.0] * HIDDEN_SIZE

    def predict(self, tensor_or_vec):
        if isinstance(tensor_or_vec[0], list):
            return rms_norm(tensor_or_vec, self.gamma)
        return norm_token(tensor_or_vec, self.gamma)

RmsNorm = RMSNorm

# ============================== RoPE ==============================

def rope_pair(x0, x1, pos, i, d_head):
    # The variables in this function are all scalars.
    freq = 1.0 / (10000 ** (i / d_head))
    angle = pos * freq
    cos_val = math.cos(angle)
    sin_val = math.sin(angle)
    return x0 * cos_val - x1 * sin_val, x0 * sin_val + x1 * cos_val

def rope_token(token_vec, pos):
    # token_vec: [d_head]
    # pos: scalar
    d_head = len(token_vec)
    # output: [d_head]
    output = []
    # Iterate over consecutive coordinate pairs: (x0, x1), (x2, x3), ...
    for i in range(0, d_head, 2):
        r0, r1 = rope_pair(token_vec[i], token_vec[i + 1], pos, i, d_head)
        output.append(r0)
        output.append(r1)
    return output

def rope_2d(x):
    # x: [seq_len, d_head]
    # return: [seq_len, d_head]
    return [rope_token(token_vec, pos) for pos, token_vec in enumerate(x)]

def rope_3d(x):
    # x: [q_heads, seq_len, d_head]
    # return: [q_heads, seq_len, d_head]
    return [rope_2d(head) for head in x]

def rope(x):
    # x: [seq_len, d_head] or [q_heads, seq_len, d_head]
    if isinstance(x[0][0], list):
        return rope_3d(x)
    return rope_2d(x)

# ============================== Attention ==============================

def attention_token(q_token, k_T, v, i):
    # q_token: [d_head]
    # k_T: [d_head, seq_len]
    # v: [seq_len, d_head]
    d_head = len(q_token)
    seq_len = len(v)

    # Compute raw dot products against all keys: [seq_len]
    dot_products = matmul(q_token, k_T)

    # Scale scores up to the current token position i: [i + 1]
    scores = []
    for j in range(i + 1):
        scores.append(dot_products[j] / math.sqrt(d_head))

    # weights: [seq_len]
    # Pad zeros for future tokens to make length [seq_len]
    weights = list(softmax(scores)) + [0.0] * (seq_len - (i + 1))

    # Compute weighted sum of values: [d_head]
    token_out = matmul(weights, v)
    return token_out

def single_attention_head(single_q, k, v):
    # single_q: [seq_len, d_head]
    # k: [seq_len, d_head]
    # v: [seq_len, d_head]
    seq_len = len(k)
    d_head = len(k[0])

    # Transpose k to compute dot products with q: [d_head, seq_len]
    k_T = [[k[row][col] for row in range(seq_len)] for col in range(d_head)]

    # head_out: [seq_len, d_head]
    head_out = []
    for i, q_token in enumerate(single_q):
        # token_out: [d_head]
        token_out = attention_token(q_token, k_T, v, i)
        head_out.append(token_out)
    return head_out

def attention_head(q, k, v):
    # q: [q_heads, seq_len, d_head]
    # k: [seq_len, d_head]
    # v: [seq_len, d_head]
    seq_len = len(k)

    # Compute attention for each query head
    # head_outs: [q_heads, seq_len, d_head]
    head_outs = []
    for single_q in q:
        # head_out: [seq_len, d_head]
        head_out = single_attention_head(single_q, k, v)
        head_outs.append(head_out)

    # Concatenate all head outputs
    # out: [seq_len, head_dim]
    out = []
    for t in range(seq_len):
        # token_out: [head_dim]
        # head_dim == q_heads * d_head
        token_out = []
        for head_out in head_outs:
            # head_out: [seq_len, d_head]
            token_out.extend(head_out[t])
        out.append(token_out)
    return out

# ============================== Grouped-Query Attention (GQA) ==============================

def group_0(rms_out, w_q, w_k, w_v):
    # rms_out: [seq_len, hidden_size]
    # w_q: [q_heads, hidden_size, d_head]
    # w_k: [hidden_size, d_head]
    # w_v: [hidden_size, d_head]

    # Project inputs to q, k, v token-wise using our matmul helper
    # q: [q_heads, seq_len, d_head]
    # k: [seq_len, d_head]
    # v: [seq_len, d_head]
    q = [[matmul(token_vec, w) for token_vec in rms_out] for w in w_q]
    k = [matmul(token_vec, w_k) for token_vec in rms_out]
    v = [matmul(token_vec, w_v) for token_vec in rms_out]

    # Encode positions with rotary embedding
    # rope is token-wise and preserves tensor shape
    q = rope(q)
    k = rope(k)

    # Compute multi-head attention and concatenate heads
    # group_0_out: [seq_len, head_dim]
    group_0_out = attention_head(q, k, v)
    return group_0_out

class Group(Layer):
    """Encapsulates Q, K, and V projection weights for a single attention group."""

    def __init__(self):
        self.w_q = [_make_matrix(HIDDEN_SIZE, D_HEAD) for _ in range(Q_HEADS)]
        self.w_k = _make_matrix(HIDDEN_SIZE, D_HEAD)
        self.w_v = _make_matrix(HIDDEN_SIZE, D_HEAD)

    def predict(self, rms_out):
        return group_0(rms_out, self.w_q, self.w_k, self.w_v)

def gqa(rms_out, groups):
    # rms_out: [seq_len, hidden_size]
    seq_len = len(rms_out)

    # Each group computes attention on the full rms_out input
    # group_out: [seq_len, head_dim]
    group_outs = [group(rms_out) for group in groups]

    # Concatenate group outputs back along the column dimension
    # gqa_out: [seq_len, hidden_size]
    gqa_out = []
    # Iterate over the rows of gqa_out
    for t in range(seq_len):
        row = []
        # Iterate the groups to concat the outputs
        for out in group_outs:
            row.extend(out[t])
        gqa_out.append(row)
    return gqa_out

class GQA(Layer):
    """Encapsulates grouped-query attention across all groups."""

    def __init__(self):
        self.groups = [Group() for _ in range(NUM_GROUPS)]

    def predict(self, rms_out):
        return gqa(rms_out, self.groups)

def out_matmul(gqa_out, w_matmul):
    # gqa_out: [seq_len, hidden_size]
    # w_matmul: [hidden_size, hidden_size]

    # gqa_block_out: [seq_len, hidden_size]
    gqa_block_out = [matmul(token_vec, w_matmul) for token_vec in gqa_out]
    return gqa_block_out

class OutMatmul(Layer):
    """Encapsulates output projection matrix for GQA."""

    def __init__(self):
        self.w_matmul = _make_matrix(HIDDEN_SIZE, HIDDEN_SIZE)

    def predict(self, gqa_out):
        return out_matmul(gqa_out, self.w_matmul)

def gqa_block(gqa_block_in, rms_norm, gqa, out_matmul):
    # gqa_block_in: [seq_len, hidden_size]

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
        return gqa_block(gqa_block_in, self.rms_norm, self.gqa, self.out_matmul)

# ============================== Mixture of Experts (MoE) ==============================

def expert_token(token_vec, w_gate, w_up, w_down):
    # token_vec: [hidden_size]
    # w_gate, w_up: [hidden_size, inter_size]
    # w_down: [inter_size, hidden_size]

    # x_gate: [inter_size]
    x_gate = matmul(token_vec, w_gate)

    # x_up: [inter_size]
    x_up = matmul(token_vec, w_up)

    # x_act: [inter_size]
    x_act = [silu(x) for x in x_gate]

    # x_inter: [inter_size]
    inter_size = len(x_act)
    x_inter = [x_act[i] * x_up[i] for i in range(inter_size)]

    # x_down: [hidden_size]
    x_down = matmul(x_inter, w_down)
    return x_down

def expert(tensor, w_gate, w_up, w_down):
    # tensor: [seq_len, hidden_size]
    # output: [seq_len, hidden_size]
    output = [expert_token(token_vec, w_gate, w_up, w_down) for token_vec in tensor]
    return output

class Expert(Layer):
    """Encapsulates SwiGLU projection weights for a single expert."""

    def __init__(self):
        self.w_gate = _make_matrix(HIDDEN_SIZE, INTER_SIZE)
        self.w_up = _make_matrix(HIDDEN_SIZE, INTER_SIZE)
        self.w_down = _make_matrix(INTER_SIZE, HIDDEN_SIZE)

    def predict(self, token_vec_or_tensor):
        if isinstance(token_vec_or_tensor[0], list):
            return expert(token_vec_or_tensor, self.w_gate, self.w_up, self.w_down)
        return expert_token(token_vec_or_tensor, self.w_gate, self.w_up, self.w_down)

def route_token(token_vec, w_router):
    # token_vec: [hidden_size]
    # w_router:  [hidden_size, num_experts]

    # logits: [num_experts]
    # example value: [1.2, 1.6, 0.3]
    logits = matmul(token_vec, w_router)

    num_experts = len(logits)

    # indices: [num_experts]
    # example value: [0, 1, 2]
    indices = [i for i in range(num_experts)]

    # sorted_indices: [num_experts]
    # example value: [1, 0, 2]
    sorted_indices = sorted(indices, key=lambda i: logits[i], reverse=True)

    # top_indices: [TOP_K]
    # example value: [1, 0]
    top_indices = sorted_indices[:TOP_K]

    # top_logits: [TOP_K]
    # example value: [1.6, 1.2]
    top_logits = [logits[i] for i in top_indices]

    # top_probs: [TOP_K]
    # example value: [0.6, 0.4]
    top_probs = softmax(top_logits)

    # top_weights: [num_experts]
    # example value: [0.4, 0.6, 0]
    top_weights = [0.0] * num_experts
    for k in range(TOP_K):
        top_weights[top_indices[k]] = top_probs[k]
    return top_weights

def router(rms_out, w_router):
    # rms_out: [seq_len, hidden_size]
    # top_weights: [seq_len, num_experts]
    top_weights = [route_token(token_vec, w_router) for token_vec in rms_out]
    return top_weights

class Router(Layer):
    """Encapsulates routing weights to select top-k experts."""

    def __init__(self):
        self.w_router = _make_matrix(HIDDEN_SIZE, NUM_EXPERTS)

    def predict(self, rms_out):
        return router(rms_out, self.w_router)

def moe_token(token_vec, top_weights, experts):
    # token_vec: [hidden_size]
    # top_weights: [num_experts]
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
    # rms_out: [seq_len, hidden_size]
    # top_weights: [seq_len, num_experts]

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
        return moe(rms_out, top_weights, self.experts)

def moe_block(moe_in, rms_norm, router, moe):
    # moe_in: [seq_len, hidden_size]

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
        return moe_block(moe_in, self.rms_norm, self.router, self.moe)

# ============================== Decoder ==============================

def decoder_block(decoder_in, gqa_block, moe_block):
    # decoder_in: [seq_len, hidden_size]

    # gqa_out: [seq_len, hidden_size]
    gqa_out = gqa_block(decoder_in)
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
        return decoder_block(decoder_in, self.gqa_block_layer, self.moe_block_layer)

def decoder(embed_out, decoder_blocks):
    # embed_out: [seq_len, hidden_size]
    # decoder_blocks: list of decoder_block layers

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
        return decoder(embed_out, self.decoder_blocks)

# ============================== Embedding & LM Head ==============================

def lookup(token_id, embedding_table):
    # token_id: int
    # embedding_table: [vocab_size, hidden_size]
    # token_vec: [hidden_size]
    token_vec = embedding_table[token_id]
    return token_vec

def embedding(input_ids, embedding_table):
    # input_ids: [seq_len]
    # embedding_table: [vocab_size, hidden_size]
    # embed_out: [seq_len, hidden_size]
    embed_out = [lookup(token_id, embedding_table) for token_id in input_ids]
    return embed_out

class Embedding(Layer):
    """Encapsulates token embedding table and lookup."""

    def __init__(self):
        self.embedding_table = _make_matrix(VOCAB_SIZE, HIDDEN_SIZE)
        self.embedding_table_T = [
            [self.embedding_table[r][c] for r in range(VOCAB_SIZE)]
            for c in range(HIDDEN_SIZE)
        ]

    def predict(self, input_ids):
        return embedding(input_ids, self.embedding_table)

def matmul_token(token_vec, embedding_table_T):
    # token_vec: [hidden_size]
    # embedding_table_T: [hidden_size, vocab_size]
    # This function is just a standard matmul.
    vocab_size = len(embedding_table_T[0])
    hidden_size = len(token_vec)

    # logits: [vocab_size]
    logits = []
    for col in range(vocab_size):
        dot_product = sum(token_vec[k] * embedding_table_T[k][col] for k in range(hidden_size))
        logits.append(dot_product)
    return logits

def logits_matmul(rms_out, embedding_table_T):
    # rms_out: [seq_len, hidden_size]
    # embedding_table_T: [hidden_size, vocab_size]

    # all_logits: [seq_len, vocab_size]
    all_logits = [matmul_token(token_vec, embedding_table_T) for token_vec in rms_out]
    return all_logits

def slice_last(all_logits):
    # all_logits: [seq_len, vocab_size]
    # logits: [vocab_size]
    logits = all_logits[-1]
    return logits

def lm_head(decoder_out, gamma, embedding_table_T):
    # decoder_out: [seq_len, hidden_size]
    # gamma: [hidden_size]
    # embedding_table_T: [hidden_size, vocab_size]

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
        self.gamma = [1.0] * HIDDEN_SIZE
        self.embedding_table_T = (
            embedding_table_T
            if embedding_table_T is not None
            else _make_matrix(HIDDEN_SIZE, VOCAB_SIZE)
        )

    def predict(self, decoder_out):
        return lm_head(decoder_out, self.gamma, self.embedding_table_T)

LmHead = LMHead

# ============================== Sampler ==============================

def greedy_sampler(model, input_ids):
    # input_ids: [seq_len]
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
        # input_ids: [seq_len]
        # embed_out: [seq_len, HIDDEN_SIZE]
        embed_out = self.embedding(input_ids)
        # decoder_out: [seq_len, HIDDEN_SIZE]
        decoder_out = self.decoder(embed_out)
        # logits: [VOCAB_SIZE]
        logits = self.lm_head(decoder_out)
        return logits

    def generate(self, input_ids, max_new_tokens=MAX_NEW_TOKENS):
        # input_ids: [seq_len]
        for _ in range(max_new_tokens):
            # next_token_id: int
            next_token_id = greedy_sampler(self, input_ids)
            # input_ids: [seq_len + 1]
            input_ids = input_ids + [next_token_id]
            if next_token_id == EOS_TOKEN_ID:
                break
        # input_ids: [total_seq_len]
        return input_ids

# ============================== Pipeline & Main ==============================

def pipeline(prompt, tokenizer=None, model=None, max_new_tokens=10):
    # prompt: str
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
    output = pipeline("What is 1+1?")
    print(output)

if __name__ == "__main__":
    main()
