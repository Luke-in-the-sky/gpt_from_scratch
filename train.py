from modules import *

hpars = Hyperparams(
    block_size=16,
    batch_size=64,
    embedding_dim=2**6 * 6,
    num_transf_blocks=2,
    num_heads=6,
    dropout_rate=0.2,
    learning_rate=4e-4,
)

# ----------
# Tokenize and load datasets
# ----------
input_file = "input.txt"

if not os.path.exists(input_file):
    download_data(destination_file=input_file)

# read the file
with open("input.txt", "r") as f:
    text = f.read()

# set up vocab
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"{vocab_size=}")

c2i = {c: i for i, c in enumerate(chars)}
i2c = {i: c for c, i in c2i.items()}
encode = lambda s: [c2i[c] for c in s]
decode = lambda d: "".join([i2c[i] for i in d])

# tokenize and encode the text
data = torch.tensor(encode(text), dtype=torch.long)

# train, eval, test (no shuffle)
n = int(len(data) * 0.9)
train_data = data[:n]
val_data = data[n:]


# ----------
# instantiate model
# ----------

m = MyGPT(
    vocab_size,
    block_size=hpars.block_size,
    embedding_dim=hpars.embedding_dim,
    num_transf_blocks=hpars.num_transf_blocks,
    num_heads=hpars.num_heads,
    dropout_rate=hpars.dropout_rate * 2,
)
logits, loss = m(xb, yb)
print(f"{xb.shape=}")
print(f"{logits.shape=}")

# create optimizer
optim = torch.optim.AdamW(m.parameters(), hpars.learning_rate)
