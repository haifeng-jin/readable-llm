import numpy as np
from readable_llm.ops import matmul, silu, softmax, rms_norm

TOP_K = 2

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

class Expert:
    def __init__(self, w_gate, w_up, w_down):
        self.w_gate = w_gate
        self.w_up = w_up
        self.w_down = w_down

    def __call__(self, token_vec_or_tensor):
        if isinstance(token_vec_or_tensor[0], list):
            return expert(token_vec_or_tensor, self.w_gate, self.w_up, self.w_down)
        return expert_token(token_vec_or_tensor, self.w_gate, self.w_up, self.w_down)

def route_token(token_vec, w_router):
    # token_vec: [hidden_size]
    # w_router:  [hidden_size, num_experts]

    # logits: [num_experts]
    # example value: [1.2, 1.6, 0.3]
    if isinstance(token_vec, list) and isinstance(w_router, list):
        logits = matmul(token_vec, w_router)
    else:
        logits = token_vec @ w_router

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
        top_weights[top_indices[k]] = float(top_probs[k])
    return top_weights

def router(rms_out, w_router):
    # rms_out: [seq_len, hidden_size]
    # top_weights: [seq_len, num_experts]
    top_weights = [route_token(token_vec, w_router) for token_vec in rms_out]
    return top_weights

def moe_token(token_vec, top_weights, experts):
    # token_vec: [hidden_size]
    # top_weights: [num_experts]
    hidden_size = len(token_vec)

    # moe_out: [hidden_size]
    moe_out = np.zeros(hidden_size)
    for i in range(len(experts)):
        # expert_out: [hidden_size]
        expert_out = experts[i](token_vec)
        moe_out += top_weights[i] * np.asarray(expert_out)
    return moe_out.tolist()

def moe(rms_out, top_weights, experts):
    # rms_out: [seq_len, hidden_size]
    # top_weights: [seq_len, num_experts]

    # moe_out: [seq_len, hidden_size]
    moe_out = [
        moe_token(token_vec, weights, experts)
        for token_vec, weights in zip(rms_out, top_weights)
    ]
    return moe_out

def moe_block(moe_in, rms_norm, router, moe):
    # moe_in: [seq_len, hidden_size]

    # rms_out: [seq_len, hidden_size]
    rms_out = rms_norm(moe_in)

    # top_weights: [seq_len, num_experts]
    top_weights = router(rms_out)

    # moe_out: [seq_len, hidden_size]
    moe_out = moe(rms_out, top_weights)
    return moe_out

class MoEBlock:
    def __init__(self, gamma, w_router, experts):
        self.gamma = gamma
        self.w_router = w_router
        self.experts = experts

    def __call__(self, moe_in):
        def norm_fn(t):
            return rms_norm(t, self.gamma)
        def router_fn(r):
            return router(r, self.w_router)
        def moe_fn(r, w):
            return moe(r, w, self.experts)
        return moe_block(moe_in, norm_fn, router_fn, moe_fn)
