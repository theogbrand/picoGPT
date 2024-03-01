import numpy as np

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def softmax(x):
    # normalise with max for numerical stability
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def layer_norm(x, g, b, eps: float = 1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    x = (x - mean) / np.sqrt(variance + eps)  # eps for zero errors, normalize x to have mean=0 and var=1 over last axis
    return g * x + b  # scale and offset with gamma/beta params


def linear(x, w, b):  # [m, in], [in, out], [out] -> [m, out]
    return x @ w + b  # matrix multiply and add bias


def ffn(x, c_fc, c_proj):
    # project up dims -> "squashing function" like ReLu
    a = gelu(linear(x, **c_fc))  # [n_seq, n_embd] -> [n_seq, 4*n_embd]

    # project back down
    x = linear(a, **c_proj)  # [n_seq, 4*n_embd] -> [n_seq, n_embd]

    return x


def attention(q, k, v, mask): # [n_q, d_k], [n_k, d_k], [n_k, d_v] -> [n_q, d_v]
    return softmax(q @ k.T / np.sqrt(q.shape[-1]) + mask) @ v


# def casual_self_attention(x, c_attn, c_proj):
#     x = linear(x, **c_attn) # q,k,v projections in single matrice for parallel compute, these are "attn_wts"

#     (q, k, v) = np.split(x, 3, axis=-1) # [n_seq, 3*n_embd] -> [n_seq, n_embd]

#     casaul_mask = (1 - np.tri(x.shape[0], dtype=x.dtype)) * -1e10

#     # learn self-attn
#     x = attention(q, k, v, casaul_mask)

#     # out proj
#     x = linear(x, **c_proj)  # [n_seq, n_embd] -> [n_seq, n_embd]

#     return x


def mha(x, c_attn, c_proj, n_head):
    x = linear(x, **c_attn) # q,k,v projections in single contiguous matrice for parallel compute, these are "attn_wts"

    # split on the last dim, into 3 parallel independent matrices
    qkv = np.split(x, 3, axis=-1) # [n_seq, 3*n_embd] -> [3, n_seq, n_embd]

    # split again for n_head parallel independent matrice, each is a "head"
    qkv_heads = list(map(lambda x: np.split(x, n_head, axis=-1), qkv))  # [3, n_seq, n_embd] -> [3, n_head, n_seq, n_embd/n_head]

    casaul_mask = (1 - np.tri(x.shape[0], dtype=x.dtype)) * -1e10

    # zip because want attn between each independent qi,ki,vi heads where i is n_heads
    out_heads = [attention(q, k, v, casaul_mask) for q, k, v in zip(*qkv_heads)] # [3, n_head, n_seq, n_embd/n_head] -> [n_head, n_seq, n_embd/n_head]

    # merge heads, previously stacked by row, now squash back into contiguous 
    x = np.hstack(out_heads) # [n_head, n_seq, n_embd/n_head] -> [n_seq, n_embd]

    # out proj
    x = linear(x, **c_proj)  # [n_seq, n_embd] -> [n_seq, n_embd]

    return x


def transformer_block(x, mlp, attn, ln_1, ln_2, n_head):
    # multi-head causal self attention
    x = x + mha(layer_norm(x, **ln_1), **attn, n_head=n_head) # [n_seq, n_embd]

    # position-wise feed forward network
    x = x + ffn(layer_norm(x, **ln_2), **mlp)  # [n_seq, n_embd] -> [n_seq, n_embd]
    return x


def gpt2(inputs, wte, wpe, blocks, ln_f, n_head):  # [n_seq] -> [n_seq, n_vocab]
    # token + positional embeddings
    x = wte[inputs] + wpe[range(len(inputs))]  # [n_seq] -> [n_seq, n_embd]
    # wte is now embeddings instead of tokens, add with wpe, which is a number from 0 to len(inputs) repr relative positions

    # forward pass through n_layer transformer blocks
    for block in blocks:
        x = transformer_block(x, **block, n_head=n_head)  # [n_seq, n_embd] -> [n_seq, n_embd]

    # projection to vocab
    x = layer_norm(x, **ln_f)  # [n_seq, n_embd] -> [n_seq, n_embd]
    return x @ wte.T  # [n_seq, n_embd] -> [n_seq, n_vocab] -> RAW LOGITS for numerical stability, flexible usability AND non-redundant np.argmax


def generate(inputs, params, n_head, n_tokens_to_generate):
    from tqdm import tqdm

    for _ in tqdm(range(n_tokens_to_generate), "generating"):  # auto-regressive decode loop
        logits = gpt2(inputs, **params, n_head=n_head)  # model forward pass
        next_id = np.argmax(logits[-1])  # greedy sampling
        inputs.append(int(next_id))  # append prediction to input

    return inputs[len(inputs) - n_tokens_to_generate :]  # only return generated ids


def main(prompt: str, n_tokens_to_generate: int = 40, model_size: str = "124M", models_dir: str = "models"):
    from utils import load_encoder_hparams_and_params

    # load encoder, hparams, and params from the released open-ai gpt-2 files
    encoder, hparams, params = load_encoder_hparams_and_params(model_size, models_dir)

    # encode the input string using the BPE tokenizer
    input_ids = encoder.encode(prompt)

    # make sure we are not surpassing the max sequence length of our model
    assert len(input_ids) + n_tokens_to_generate < hparams["n_ctx"]

    # generate output ids
    output_ids = generate(input_ids, params, hparams["n_head"], n_tokens_to_generate)

    # decode the ids back into a string
    output_text = encoder.decode(output_ids)

    return output_text


if __name__ == "__main__":
    import fire

    fire.Fire(main)