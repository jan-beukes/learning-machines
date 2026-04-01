import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader

def main():
    with open("./data/the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    vocab_size = 50257
    output_dim = 256
    # Creates a 2d tensor with vocab_size number of output_dim vectors which maps the token ids in
    # the vocab to an embedding of size output_dim
    token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

    max_length = 4
    dataloader = create_dataloader(raw_text, batch_size=8, max_length=max_length, stride=4, shuffle=False)
    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)

    context_length = max_length
    token_embeddings = token_embedding_layer(inputs)
    pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
    # torch.arange gives seq from 0..context_length-1
    pos_embeddings = pos_embedding_layer(torch.arange(context_length))

    # This is the input data that we will use to train the llm
    input_embeddings = token_embeddings + pos_embeddings



def create_dataloader(txt, batch_size=4, max_length=256, stride=128,
                      shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDataset(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                drop_last=drop_last,
                num_workers=num_workers,
            )
    return dataloader

class GPTDataset(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.output_ids = []

        token_ids = tokenizer.encode(txt)

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i+max_length]
            target_chunk = token_ids[i+1:i+1+max_length]
            self.input_ids.append(torch.tensor(input_chunk))
            self.output_ids.append(torch.tensor(target_chunk))

        # These methods need to be implemented by a dataset when we use it in a DataLoader
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.output_ids[idx]

if __name__ == "__main__":
    main()
