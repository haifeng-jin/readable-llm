import math
import numpy as np

EPS = 1e-6

def argmax(logits):
    # logits: [vocab_size]
    # max_idx: int
    max_idx = 0
    for i in range(len(logits)):
        if logits[i] > logits[max_idx]:
            max_idx = i
    return max_idx

def softmax(x):
    # Supports both 1D list and numpy array
    arr = np.asarray(x, dtype=float)
    exp_x = np.exp(arr - np.max(arr))
    return exp_x / np.sum(exp_x)

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
