import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)
# hyperparameters adjusted for cpu-only training
batch_sz = 8
block_sz = 64
max_iters = 3000
eval_interval = 300
learning_rate = 3e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2

with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()


chars = sorted(list(set(text)))
vocab_sz = len(chars)

# creating a simple tokenizer
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [
    stoi[c] for c in s
]  # encoder: take a string, output a list of integers
decode = lambda l: "".join(
    [itos[i] for i in l]
)  # decoder: take a list of integers, output a string


# 90/10 data split
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train = data[:n]
val = data[n:]


def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train if split == "train" else val
    ix = torch.randint(len(data) - block_sz, (batch_sz,))
    x = torch.stack([data[i : i + block_sz] for i in ix])
    y = torch.stack([data[i + 1 : i + block_sz + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, y = get_batch(split)
            logits, loss = model(X, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):
    """implementation of one head of self-attention"""

    def __init__(self, head_sz):
        super().__init__()
        self.key = nn.Linear(n_embd, head_sz, bias=False)
        self.query = nn.Linear(n_embd, head_sz, bias=False)
        self.value = nn.Linear(n_embd, head_sz, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_sz, block_sz)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape

        k = self.key(x)
        q = self.query(x)
        W = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        W = W.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        W = F.softmax(W, dim=-1)
        W = self.dropout(W)

        v = self.value(x)
        out = W @ v
        return out


class MultiHeadAttention(nn.Module):
    """multiple heads of self-attention in parallel"""

    def __init__(self, head_sz, num_heads):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_sz) for _ in range(num_heads)])
        self.proj = nn.Linear(head_sz * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(x))
        return out


class FeedForward(nn.Module):
    """MLP layer with dropout"""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(
                n_embd, 4 * n_embd
            ),  # attention paper uses 1:4 input to inner-layer dimensionality ratio
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block: communication followed by computation"""

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_sz = n_embd // n_head
        self.s_attn = MultiHeadAttention(head_sz, n_head)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.s_attn(self.ln1(x))  # skip connections as our nn gets more dense
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    """Putting it together"""

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_sz, n_embd)
        self.position_embedding_table = nn.Embedding(block_sz, n_embd)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head=n_head) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)  # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_sz)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C)
        x = self.blocks(x)  # (B,T,C)
        x = self.ln_f(x)  # (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_sz:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


model = GPTLanguageModel()
m = model.to(device)
print(sum(p.numel() for p in m.parameters()) / 1e6, "M parameters")


# optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

# training loop
for iter in range(max_iters):
    if (
        iter % eval_interval == 0 or iter == max_iters - 1
    ):  # at intervals of 300 or at 2999
        losses = estimate_loss()

        print(
            f"step {iter} : training loss {losses['train']:.4f}, validation loss: {losses['val']:.4f}"
        )

    xb, yb = get_batch("train")

    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1,1), dtype= torch.long, device=device)
open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=1000)[0].tolist()))