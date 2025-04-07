import torch
import tiktoken
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    """
    Custom dataset for preparing text data for language model training.

    This dataset tokenizes the input text and creates input-target pairs
    using a sliding window approach. Each input sequence is a fixed-length
    chunk of tokens, and the corresponding target sequence is the same chunk
    shifted by one token to the right.

    Args:
        txt (str): The input text string.
        tokenizer (tiktoken.Encoding): The tokenizer used to convert text to tokens.
        max_length (int): The maximum length of each input and target sequence.
        stride (int): The step size for the sliding window.
    """
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt)  # Tokenize the text
        """
        Sliding window on length (max_length) and jump (stride) to
        store input and target to be used later.
        Note: Since all data is loaded into memory with large dataset
        this can cause implementation issues
        """
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        """
        Returns the total number of rows in the dataset.

        Returns:
            int: The number of input-target pairs in the dataset.
        """
        return len(self.input_ids)

    def __getitem__(self, idx):
        """
        Returns a single row (input-target pair) from the dataset.

        Args:
            idx (int): The index of the desired row.

        Returns:
            tuple: A tuple containing the input tensor and the target tensor.
                   Both tensors have shape (max_length,).
        """
        return self.input_ids[idx], self.target_ids[idx]

def create_dataloader(txt, batch_size=4, max_length=256, stride=128, shuffle=True,
                      drop_last=True, num_workers=0):
    """
    Creates a DataLoader for training a language model.

    This function takes raw text data, tokenizes it, and prepares it for
    training by creating a DataLoader that yields batches of input-target
    pairs.

    Args:
        txt (str): The input text string.
        batch_size (int, optional): The number of samples per batch. Defaults to 4.
        max_length (int, optional): The maximum length of each input and target sequence. Defaults to 256.
        stride (int, optional): The step size for the sliding window. Defaults to 128.
        shuffle (bool, optional): Whether to shuffle the data. Defaults to True.
        drop_last (bool, optional): Whether to drop the last incomplete batch. Defaults to True.
        num_workers (int, optional): The number of subprocesses to use for data loading. Defaults to 0.

    Returns:
        torch.utils.data.DataLoader: A DataLoader that yields batches of input-target pairs.
    """

    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Creates dataset
    dataset = CustomDataset(txt, tokenizer, max_length, stride)

    # drop_last=True drops the last batch if it is shorter than the specified batch_size
    # to prevent loss spikes during training.
    # num_workers - The number of CPU processes to use for preprocessing
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
    return dataloader

if __name__ == "__main__":
    
    # working with sample data
    from llm.bhasa import data
    filepath = data.download_sample_text()
    with open(filepath, "r", encoding="utf-8") as f:
        raw_text = f.read()
        
    dataloader = create_dataloader(raw_text, batch_size=1, max_length=4, stride=1, shuffle=False)
    data_iter = iter(dataloader)
    first_batch = next(data_iter)
    print(first_batch)
