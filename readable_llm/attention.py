import math
from readable_llm.ops import matmul, softmax

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
