from modules import *
import logging

debugging = True

debugging = Hyperparams(
    block_size=16,
    batch_size=16,
    embedding_dim=2**3 * 3,
    # transformer params
    num_transf_blocks=1,
    num_heads=3,
    dropout_rate=0.2,
    # trainin params
    learning_rate=4e-4,
    training_steps=500,
    # others
    vocab_size=None,  # will compute this later
)

performing = Hyperparams(
    # input dimensions
    block_size=16,
    batch_size=64,
    embedding_dim=2**6 * 3,
    # transformer params
    num_transf_blocks=1,
    num_heads=3,
    dropout_rate=0.2,
    # trainin params
    learning_rate=4e-4,
    training_steps=500,
    # others
    vocab_size=None,  # will compute this later
)

hpars = debugging
logging.basicConfig(
    level=logging.DEBUG if debugging else logging.INFO,
    format="%(asctime)s-%(levelname)s\t%(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ----------
# Tokenize and load datasets
# ----------
input_file = "input.txt"

if not os.path.exists(input_file):
    print("Downloading text..")
    download_data(destination_file=input_file)

# read the file
logger.info("Reading the text..")
with open("input.txt", "r") as f:
    text = f.read()

# set up vocab
chars = sorted(list(set(text)))
hpars.vocab_size = len(chars)
logger.info(f"{hpars.vocab_size=}")

c2i = {c: i for i, c in enumerate(chars)}
i2c = {i: c for c, i in c2i.items()}
encode = lambda s: [c2i[c] for c in s]
decode = lambda d: "".join([i2c[i] for i in d])

# tokenize and encode the text
data = torch.tensor(encode(text), dtype=torch.long)


# create dataset and dataloader
n = int(len(data) * 0.9)
dataset = DataSet(
    train=data[:n],
    validation=data[n:],
)

loader = DataLoader(
    batch_size=hpars.batch_size,
    block_size=hpars.block_size,
    data=dataset,
)

# ----------
# instantiate model
# ----------

m = MyGPT(hypers=hpars)
logging.info(f'Model size: {sum([p.numel() for p in m.parameters()])}')
if debugging:
    xb_debug, yb_debug = loader.get_batch(split="train")
    logits, loss = m(xb_debug, yb_debug)
    logger.debug(f"{xb_debug.shape=}")
    logger.debug(f"{logits.shape=}")
    evaled_loss = evaluate_loss(m, loader, num_evals=10)
    logger.debug(f"{evaled_loss=}")


# create optimizer
optim = torch.optim.AdamW(m.parameters(), hpars.learning_rate)

# allocate tracker for losses
losses = []

# train
for i in range(hpars.training_steps):
    # forward
    xb, yb = loader.get_batch(split="train")

    # compute loss
    logits, loss = m(xb, targets=yb)
    if i % 100 == 0:
        out = evaluate_loss(m, loader)
        logger.info(f"{i}/{hpars.training_steps} {out=}")
        losses.append(out)

    # back
    optim.zero_grad(set_to_none=True)
    loss.backward()
    optim.step()
