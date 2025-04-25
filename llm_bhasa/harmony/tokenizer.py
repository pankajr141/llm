import re
import torch
import tiktoken

class SimpleTokenizer:
    """
    A simple tokenizer that splits text into tokens based on punctuation and whitespace.

    This tokenizer is designed for basic text processing and is not as sophisticated
    as the BPE tokenizer used in GPT models. It splits text based on common
    punctuation marks and whitespace, and it handles unknown tokens by replacing
    them with a special "<|unk|>" token.

    Attributes:
        str_to_int (dict): A dictionary mapping tokens (strings) to their integer IDs.
        int_to_str (dict): A dictionary mapping integer IDs to their corresponding tokens (strings).
    """
    def __init__(self, vocab):
        """
        Initializes the SimpleTokenizer with a given vocabulary.

        Args:
            vocab (dict): A dictionary mapping tokens (strings) to their integer IDs.
        """
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}

    def encode(self, text):
        """
        Encodes a text string into a list of integer IDs.

        The text is preprocessed by splitting it into tokens based on punctuation
        and whitespace. Unknown tokens are replaced with the "<|unk|>" token.

        Args:
            text (str): The input text string.

        Returns:
            list: A list of integer IDs representing the encoded text.
        """
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        preprocessed = [item if item in self.str_to_int else "<|unk|>" for item in preprocessed]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        """
        Decodes a list of integer IDs back into a text string.

        The integer IDs are converted back to their corresponding tokens, and
        the tokens are joined together with spaces. Punctuation marks are
        adjusted to remove extra whitespace.

        Args:
            ids (list): A list of integer IDs.

        Returns:
            str: The decoded text string.
        """
        text = " ".join([self.int_to_str[i] for i in ids]) 
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text

def text_to_token_ids(text, tokenizer):
    """
    Converts a text string to a tensor of token IDs.

    This function uses the provided tokenizer to encode the text into a list of
    token IDs and then converts it into a PyTorch tensor.

    Args:
        text (str): The input text string.
        tokenizer (tiktoken.Encoding): The tokenizer to use for encoding.

    Returns:
        torch.Tensor: A tensor of token IDs with shape (1, num_tokens).
    """
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    """
    Converts a tensor of token IDs back to a text string.

    This function takes a tensor of token IDs, flattens it, and then uses the
    provided tokenizer to decode it back into a text string.

    Args:
        token_ids (torch.Tensor): A tensor of token IDs.
        tokenizer (tiktoken.Encoding): The tokenizer to use for decoding.

    Returns:
        str: The decoded text string.
    """
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())

def get_tokenizer():
    """
    Returns the GPT-2 tokenizer from the tiktoken library.

    This function initializes and returns the GPT-2 tokenizer, which is a
    Byte Pair Encoding (BPE) tokenizer.

    Returns:
        tiktoken.Encoding: The GPT-2 tokenizer.
    """
    tokenizer = tiktoken.get_encoding("gpt2")
    return tokenizer

def load_vocab(filepath):
    """
    Loads a vocabulary from a text file.

    This function reads a text file, preprocesses it to extract unique tokens,
    and creates a vocabulary mapping each token to a unique integer ID.

    Args:
        filepath (str): The path to the text file.

    Returns:
        dict: A dictionary mapping tokens (strings) to their integer IDs.
    """
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