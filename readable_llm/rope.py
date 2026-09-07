import math

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
