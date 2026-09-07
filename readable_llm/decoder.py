from readable_llm.ops import add

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

class DecoderBlock:
    def __init__(self, gqa_block_layer, moe_block_layer):
        self.gqa_block_layer = gqa_block_layer
        self.moe_block_layer = moe_block_layer

    def __call__(self, decoder_in):
        return decoder_block(decoder_in, self.gqa_block_layer, self.moe_block_layer)

def decoder(embed_out, decoder_blocks):
    # embed_out: [seq_len, hidden_size]
    # decoder_blocks: list of decoder_block layers

    # decoder_out: [seq_len, hidden_size]
    decoder_out = embed_out
    for block in decoder_blocks:
        decoder_out = block(decoder_out)
    return decoder_out
