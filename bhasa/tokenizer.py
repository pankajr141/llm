import re
import torch
import tiktoken

class SimpleTokenizer:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        preprocessed = [item if item in self.str_to_int else "<|unk|>" for item in preprocessed]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids]) 
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())

def get_tokenizer():
    tokenizer = tiktoken.get_encoding("gpt2")
    return tokenizer

def load_vocab(filepath):
    import re

    with open(filepath, "r", encoding="utf-8") as f:
        raw_text = f.read()

    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]
    
    all_tokens = sorted(set(preprocessed))
    all_tokens.extend(["<|endoftext|>", "<|unk|>"])
    
    vocab = {token:integer for integer,token in enumerate(all_tokens)}
    return vocab
    
if __name__ == "__main__":
    import os
    import sys
    path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    sys.path.append(path)

    # working with sample data
    from bhasa import data
    filepath = data.download_sample_text()
    vocab = load_vocab(filepath)
    tokenizer = SimpleTokenizer(vocab)

    text1 = "Hello, do you like tea?"
    text2 = "In the sunlit terraces of the palace."
    text = " <|endoftext|> ".join((text1, text2))
    print(f"Text (original) : {text}")
    print(f"Text (encoded)  : {tokenizer.encode(text)}")
    print(f"Text (decoded)  : {tokenizer.decode(tokenizer.encode(text))}")