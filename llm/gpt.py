from minbpe import RegexTokenizer as Tokenizer
import torch
import torch.nn as nn
from torch.nn import functional as F

device = "cuda"

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # (vocab_size,vocab_size) tensor which maps id to embedding vector
        self.embedding = nn.Embedding(vocab_size, vocab_size)

    # idx and targets are (batch_size, context) tensor
    def forward(self, idx, targets=None):
        # This returns a (batch_size, context, embedding) tensor where each
        # token id was now replaced by an embedding vector
        logits = self.embedding(idx)

        if not targets is None:
            B, T, E = logits.shape # T (context) is called the time dimension
            # reshape to be matrix of batch_size * context rows with each row being the
            # embedding vectors that we will compute loss with against the targets
            logits = logits.view(B*T, E)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        else:
            loss = None

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T)
        for i in range(max_new_tokens):
            logits, loss = self(idx)

            # get the last item in the context/time dimension
            logits = logits[:, -1, :]
            # get probs
            probs = F.softmax(logits, dim=-1) # (B, E)
            # sample from the probs
            next_idx = torch.multinomial(probs, num_samples=1) # (B, 1)
            idx = torch.cat((idx, next_idx), dim=1) # (B, T+1)
        return idx

# Get a random batch
def get_batch(data, context, batch_size):
    batch_idxs = torch.randint(len(data) - context, (batch_size,))
    x = torch.stack([data[i:i+context] for i in batch_idxs])
    y = torch.stack([data[i+1:i+context+1] for i in batch_idxs])
    return x, y

def main():
    with open("data/shakespeare.txt") as f:
        text = f.read()

    vocab_size = 1000
    tokenizer = Tokenizer()
    try:
        tokenizer.load("tokenizer.model")
    except:
        print("Training tokenizer")
        tokenizer.train(text, vocab_size)
        tokenizer.save("tokenizer")
    tokens = tokenizer.encode(text[:1000])

    data = torch.tensor(tokens)
    data = data.to(device)

    train_split = int(0.9*len(data))
    train_data = data[:train_split]
    val_data = data[train_split:]

    # Max context length for predictions
    model = BigramLanguageModel(vocab_size).to(device)

    context = 8
    batch_size = 32
    learn_rate = 0.003
    optimizer = torch.optim.AdamW(model.parameters(), lr=learn_rate)

    epochs = 10_000
    for i in range(epochs):
        xb, yb = get_batch(train_data, batch_size, context)

        logits, loss = model(xb, yb)
        if i % 10 == 0: print(f"{i}: Loss = {loss.item()}")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    lf_token = tokenizer.encode('\n')
    idx = torch.tensor([lf_token], dtype=torch.long).to(device)
    generated_tokens = model.generate(idx, max_new_tokens=200)[0].tolist()
    print(tokenizer.decode(generated_tokens))

main()
