from readable_llm.ops import argmax

def greedy_sampler(model, input_ids):
    # input_ids: [seq_len]
    # logits: [vocab_size]
    logits = model.predict(input_ids)
    # next_token_id: int
    next_token_id = argmax(logits)
    return next_token_id
