# Vocabulary mapping tokens to unique integer IDs
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
    def __init__(self, vocab):
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
