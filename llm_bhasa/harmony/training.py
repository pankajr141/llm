import os
# import sys
# path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
# sys.path.append(path)

import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from torch.nn.parallel.data_parallel import DataParallel

from llm_bhasa.harmony import config
from llm_bhasa.harmony import model
from llm_bhasa.harmony import generator
from llm_bhasa.harmony import data, dataset
from llm_bhasa.harmony import tokenizer as tokenizer_lib


def calc_loss_batch(input_batch, target_batch, model, device):
    """
    Calculates the cross-entropy loss for a single batch.

    This function computes the cross-entropy loss between the model's
    predictions (logits) and the target values for a given batch of data.

    Args:
        input_batch (torch.Tensor): Input batch tensor of shape (batch_size, sequence_length).
        target_batch (torch.Tensor): Target batch tensor of shape (batch_size, sequence_length).
        model (torch.nn.Module): The language model.
        device (torch.device): The device (CPU or GPU) to use.

    Returns:
        torch.Tensor: The calculated cross-entropy loss (a scalar tensor).
    """
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)      
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )
    return loss

def calc_loss_loader(data_loader, model, device, num_batches=None):
    """
    Calculates the average loss over a data loader.

    This function iterates through a data loader, computes the loss for each
    batch, and returns the average loss across all processed batches.

    Args:
        data_loader (torch.utils.data.DataLoader): The data loader.
        model (torch.nn.Module): The language model.
        device (torch.device): The device (CPU or GPU) to use.
        num_batches (int, optional): The number of batches to process. If None,
            processes all batches. Defaults to None.

    Returns:
        float: The average loss. Returns NaN if the data loader is empty.
    """
    total_loss = 0.
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i > num_batches:
            return total_loss / num_batches

        loss = calc_loss_batch(
            input_batch, target_batch, model, device
        )
        total_loss += loss.item()
    return total_loss / i

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    """
    Evaluates the model on the training and validation sets.

    This function calculates the average loss of the model on both the training
    and validation data loaders. It sets the model to evaluation mode during
    the process and then returns it to training mode.

    Args:
        model (torch.nn.Module): The language model.
        train_loader (torch.utils.data.DataLoader): The training data loader.
        val_loader (torch.utils.data.DataLoader): The validation data loader.
        device (torch.device): The device (CPU or GPU) to use.
        eval_iter (int): The number of batches to use for evaluation.

    Returns:
        tuple: A tuple containing the training loss and validation loss.
    """
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_loss = calc_loss_loader(
            val_loader, model, device, num_batches=eval_iter
        )
    model.train()
    return train_loss, val_loss

def train_model(model, train_loader, val_loader, optimizer, device, num_epochs,
                eval_freq, eval_iter, start_context, tokenizer):
    """
    Trains the language model.

    This function trains the provided language model using the given training
    data loader and optimizer. It also evaluates the model on the validation
    data loader at specified intervals.

    Args:
        model (torch.nn.Module): The language model.
        train_loader (torch.utils.data.DataLoader): The training data loader.
        val_loader (torch.utils.data.DataLoader): The validation data loader.
        optimizer (torch.optim.Optimizer): The optimizer.
        device (torch.device): The device (CPU or GPU) to use.
        num_epochs (int): The number of training epochs.
        eval_freq (int): The frequency of evaluation (in steps).
        eval_iter (int): The number of batches to use for evaluation.
        start_context (str): The starting context for text generation.
        tokenizer (tiktoken.Encoding): The tokenizer.

    Returns:
        tuple: A tuple containing lists of training losses, validation losses,
            and the number of tokens seen at each evaluation step.
    """
    
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(
                input_batch, target_batch, model, device
            )
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)

                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, "
                      f"Val loss {val_loss:.3f}"
                )

        generate_and_print_sample(
            model, device, tokenizer, start_context
        )
    return train_losses, val_losses, track_tokens_seen

def generate_and_print_sample(model, device, tokenizer, start_context):
    """
    Generates and prints a sample text sequence from the model.

    This function generates a sample text sequence using the provided model,
    starting from the given context. It then prints the generated text to the
    console.

    Args:
        model (torch.nn.Module): The language model.
        device (torch.device): The device (CPU or GPU) to use.
        tokenizer (tiktoken.Encoding): The tokenizer.
        start_context (str): The starting context for text generation.
    """
    model.eval()
    if isinstance(model, DataParallel):
        context_size = model.module.pos_emb.weight.shape[0] # config_train["context_length"]
    else:
        context_size = model.pos_emb.weight.shape[0] # config_train["context_length"]

    encoded = tokenizer_lib.text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generator.generate(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size
        )
    decoded_text = tokenizer_lib.token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))
    model.train()

def split_data(filepaths, train_ratio=0.90):
    """
    Splits the filepaths into training and validation sets.

    This function divides the input filepaths into two parts: a training set
    and a validation set, based on the specified ratio.

    Args:
        filepaths (list): A list of filepaths to text files.
        train_ratio (float, optional): The ratio of data to use for training.
            Defaults to 0.90.

    Returns:
        tuple: A tuple containing the training data and validation data.
    """
    split_idx = int(train_ratio * len(filepaths))
    train_data = filepaths[:split_idx]
    val_data = filepaths[split_idx:]
    return train_data, val_data

def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    """
    Plots the training and validation losses.

    This function creates a plot showing the training and validation losses
    over the course of training.

    Args:
        epochs_seen (torch.Tensor): Tensor of epochs seen.
        tokens_seen (list): List of tokens seen.
        train_losses (list): List of training losses.
        val_losses (list): List of validation losses.
    """
    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(
        epochs_seen, val_losses, linestyle="-.", label="Validation loss"
    )
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
 
def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device

def train(tokenizer=tokenizer_lib.get_tokenizer(), config_train=config.GPT_CONFIG_124M, 
          num_epochs=10, eval_freq=5, eval_iter=5, model_filepath="model_and_optimizer.pth"):
    """
    Main training function to orchestrate the model training process.

    This function manages the entire training pipeline for the language model.
    It performs the following steps:
    1. Initializes the training environment, including setting the random seed and determining the device (CPU or GPU).
    2. Loads the language model, either from scratch or from a previously saved checkpoint.
    3. Prepares the training and validation datasets using the specified tokenizer and configuration.
    4. Sets up the optimizer for training.
    5. Iterates through the specified number of training epochs, performing forward and backward passes, and updating the model's weights.
    6. Evaluates the model's performance on the validation set at regular intervals.
    7. Generates and prints sample text from the model during training to monitor progress.
    8. Plots the training and validation loss curves.
    9. Saves the trained model to a file, allowing for later resumption of training or inference.

    Args:
        tokenizer (tiktoken.Encoding, optional): The tokenizer to use for text processing.
            Defaults to tokenizer_lib.get_tokenizer().
        config_train (dict, optional): The configuration dictionary for the model.
            Defaults to config.GPT_CONFIG_124M.
        num_epochs (int, optional): The number of training epochs to run. Defaults to 10.
        eval_freq (int, optional): The frequency of evaluation (in steps).
            Defaults to 5.
        eval_iter (int, optional): The number of batches to use for each
            evaluation. Defaults to 5.
        model_filepath (str, optional): The file path to load/save the model checkpoint.
            Defaults to "model_and_optimizer.pth".
    """
    torch.manual_seed(123)
    device = get_device()

    model_llm = model.LLMModel(config_train)
    model_llm = model.load_model(model_llm, model_filepath) # Resuming training by loading previously trained model
    model_llm.to(device)                                    # Assigning GPU/CPU to model

    print(f"Device: {device}")

    model.print_model_information(model_llm)

    filepaths = data.download_sample_text()         # Download sample data for model training
    print(f"Total filepaths: len{filepaths}")

    train_data, val_data = split_data(filepaths, train_ratio=0.70)
    
    context_len = config_train['context_length']

    # Creating data loader for both train and validation
    train_loader = dataset.create_dataloader(train_data, batch_size=2, max_length=context_len, stride=context_len,
                                             drop_last=True, num_workers=0)
    
    val_loader = dataset.create_dataloader(val_data, batch_size=2, max_length=context_len, stride=context_len,
                                            drop_last=False, num_workers=0)

    # Defining optimizer
    optimizer = torch.optim.AdamW(model_llm.parameters(), lr=0.0004, weight_decay=0.1)

    # Training LLM model from scratch
    train_losses, val_losses, tokens_seen = train_model(model_llm, train_loader, val_loader, optimizer, device,
                                                        num_epochs=num_epochs, eval_freq=eval_freq, eval_iter=eval_iter,
                                                        start_context="Every effort moves you", tokenizer=tokenizer)
    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)

    # Saving the model, so that we can resume training later or use for inference
    model.save_model(model_llm, optimizer, model_filepath)

if __name__ == "__main__":
    train()
