import tiktoken

tokenizer = tiktoken.get_encoding('gpt2')
with open('the-verdict.txt') as file:
    text = file.read()

tok_ids = tokenizer.encode(text)
tok_ids = tok_ids[50:]

context_size = 4
for i in range(1, context_size+1):
    context = tokenizer.decode(tok_ids[:i])
    target = tokenizer.decode([tok_ids[i]])
    print(f"{context} --> {target}")
