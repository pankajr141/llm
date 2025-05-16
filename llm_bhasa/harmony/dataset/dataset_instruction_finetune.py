import json
import torch
import random
import tiktoken
from functools import partial
from torch.utils.data import IterableDataset, Dataset, DataLoader
from llm_bhasa.harmony import data


class InstructionFineTuneDataset(IterableDataset):
    """
    Custom dataset for preparing text data for language model training.

    This dataset processes text data from multiple files, tokenizes it, and
    creates input-target pairs using a sliding window approach. Each input
    sequence is a fixed-length chunk of tokens, and the corresponding target
    sequence is the same chunk shifted by one token to the right.

    This class inherits from `torch.utils.data.IterableDataset`, making it
    suitable for handling large datasets that may not fit entirely in memory.

    Args:
        filepaths (list): A list of filepaths to text files.
        tokenizer (tiktoken.Encoding): The tokenizer used to convert text to tokens.
        max_length (int): The maximum length of each input and target sequence.
        stride (int): The step size for the sliding window.

    Yields:
        tuple: A tuple containing two tensors:
            - input_sequence (torch.Tensor): A tensor of token IDs representing the input sequence.
            - target_sequence (torch.Tensor): A tensor of token IDs representing the target sequence.

    Raises:
        FileNotFoundError: If any of the specified filepaths do not exist.
        IOError: If there is an error reading any of the files.
    """
    def __init__(self, filepaths, tokenizer, shuffle, style="alpaca"):
        """
        Initializes the InstructionFineTuneDataset.

        Args:
            filepaths (list): A list of filepaths to text files.
            tokenizer (tiktoken.Encoding): The tokenizer used to convert text to tokens.
            max_length (int): The maximum length of each input and target sequence.
            stride (int): The step size for the sliding window.
        """
        self.filepaths = filepaths
        self.tokenizer = tokenizer
        self.shuffle = shuffle
        self.style = style

    def __iter__(self):
        """
        Iterates through the dataset, yielding input-target pairs.

        This method reads each file, tokenizes its content, and then applies
        a sliding window to create input-target pairs.

        Yields:
            tuple: A tuple containing two tensors:
                - input_sequence (torch.Tensor): A tensor of token IDs representing the input sequence.
                - target_sequence (torch.Tensor): A tensor of token IDs representing the target sequence.
        """

        # Shuffle the filepaths / URIs
        filepaths = self.filepaths
        if self.shuffle:
            random.shuffle(filepaths)

        for filepath in filepaths:
            try:
                # In case filepath is an URI directly read text from URI                
                status, text = data.read_from_url(filepath) if filepath.startswith("http") else data.read_from_file(filepath)

                if not status:
                    continue

                if filepath.endswith('json'):
                    text = json.loads(text)
                    if self.shuffle:
                        random.shuffle(text)

                for instruction in text:
                    instruction_plus_input = self.format_prompt(instruction, style=self.style)
                    response_text = f"\n\n### Response:\n{instruction['output']}"
                    instruction_complete = instruction_plus_input + response_text
                    encoded_text = self.tokenizer.encode(instruction_complete)
                    yield encoded_text

            except Exception as err:
                print(f"Error processing {filepath}: {err}")

    def format_prompt(self, entry, style="alpaca"):
        if style == "alpaca":
            instruction_text = (
                f"Below is an instruction that describes a task. "
                f"Write a response that appropriately completes the request."
                f"\n\n### Instruction:\n{entry['instruction']}"
            )

            input_text = (
                f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
            )
            return instruction_text + input_text

def create_dataloader_instruction_finetune(filepaths, batch_size=4, max_length=1024, shuffle=False, drop_last=True, num_workers=0, device="cpu"):
    """
    Creates a DataLoader for training a language model.

    This function takes a list of filepaths, tokenizes the text data in each file,
    and prepares it for training by creating a DataLoader that yields batches
    of input-target pairs.

    Args:
        filepaths (list): A list of filepaths to text files.
        batch_size (int, optional): The number of samples per batch. Defaults to 4.
        max_length (int, optional): The maximum length of each input and target sequence. Defaults to 1024.
        drop_last (bool, optional): Whether to drop the last incomplete batch. Defaults to True.
        num_workers (int, optional): The number of subprocesses to use for data loading. Defaults to 0.

    Returns:
        torch.utils.data.DataLoader: A DataLoader that yields batches of input-target pairs.
    """

    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    customized_collate_fn = partial(custom_collate_fn, device=device, allowed_max_length=max_length)

    # Creates dataset
    dataset = InstructionFineTuneDataset(filepaths, tokenizer, shuffle)

    # drop_last=True drops the last batch if it is shorter than the specified batch_size
    # to prevent loss spikes during training.
    # num_workers - The number of CPU processes to use for preprocessing
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        drop_last=drop_last,
        num_workers=num_workers
    )
    return dataloader

def custom_collate_fn(batch, pad_token_id=50256, ignore_index=-100, allowed_max_length=None, device="cpu"):

    # Finds the longest sequence in the batch
    batch_max_length = max(len(item) + 1 for item in batch)
    inputs_lst, targets_lst = [], []

    # Pads and prepares inputs
    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]

        # Pads sequences to max_length
        padded = (
            new_item + [pad_token_id] *
            (batch_max_length - len(new_item))
        )

        inputs = torch.tensor(padded[:-1]) # Truncates last token for inputs
        targets = torch.tensor(padded[1:]) # Shift +1 to the right for targets

        # Replaces all but the first padding tokens in targets by ignore_index
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        # Optionally truncates to the maximum sequence length
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    # Converts the list of inputs to a tensor and transfers it to the target device
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor, targets_tensor

if __name__ == "__main__":
    base_urls = ["https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch07/01_main-chapter-code/instruction-data.json", 
                "https://raw.githubusercontent.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM/refs/heads/main/data/alpaca_gpt4_data.json"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader = create_dataloader_instruction_finetune(base_urls, batch_size=2, max_length=1024, shuffle=True, device=device)
    for i, (inputs, targets) in enumerate(dataloader):
        #print(inputs.shape, targets.shape)
        #break
        if i % 100 == 0:
            print(i)