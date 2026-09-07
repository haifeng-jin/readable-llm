from readable_llm.ops import rms_norm

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
