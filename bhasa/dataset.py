import torch
import tiktoken
from torch.utils.data import Dataset, DataLoader

class CustomDataset_V1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt) # Tokenize the text
        """ sliding window on length (max_length) and jump (stride) to 
        store input and target to be used later.
        Note: Since all data is loaded into memory with large dataset 
        this can cause implementation issues """
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        """Returns the total number of rows in the dataset"""
        return len(self.input_ids)

    def __getitem__(self, idx):
        """Returns a single row from the dataset"""
        return self.input_ids[idx], self.target_ids[idx]

def create_dataloader(txt, batch_size=4, max_length=256, stride=128, shuffle=True, 
                      drop_last=True, num_workers=0):

    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Creates dataset
    dataset = CustomDataset_V1(txt, tokenizer, max_length, stride)


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
    from llm.bhasa import sample_data
    filepath = data.download_sample_text()
    with open(filepath, "r", encoding="utf-8") as f:
        raw_text = f.read()
        
    dataloader = create_dataloader(raw_text, batch_size=1, max_length=4, stride=1, shuffle=False)
    data_iter = iter(dataloader)
    first_batch = next(data_iter)
    print(first_batch)