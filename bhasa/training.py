import os
import sys
path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(path)

import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from bhasa import config
from bhasa import model
from bhasa import generator
from bhasa import data, dataset
from bhasa import tokenizer as tokenizer_lib

tokenizer = None
config_train = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)      
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )
    return loss

def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(
                input_batch, target_batch, model, device
            )
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
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
    model.eval()
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

def split_data(textdata, train_ratio=0.90):
    split_idx = int(train_ratio * len(textdata))
    train_data = textdata[:split_idx]
    val_data = textdata[split_idx:]
    return train_data, val_data

def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(
        epochs_seen, val_losses, linestyle="-.", label="Validation loss"
    )
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
 

def train(tokenizer):
    torch.manual_seed(123)
    model_llm = model.LLMModel(config_train)
    model_llm = model.load_model(model_llm)         # Resuming training by loading previously trained model
    model_llm.to(device)                            # Assigning GPU/CPU to model

    print(f"Device: {device}")

    model.print_model_information(model_llm)

    filepaths = data.download_sample_text()         # Download sample data for model training
    textdata = data.read_filepaths(filepaths)       # Read content of sample file

    total_characters = len(textdata)                # Number of characters in textdata
    total_tokens = len(tokenizer.encode(textdata))  # Convert/Encode textdata -> tokens to be passed to LLM
    print(f"Characters: {total_characters}\nTokens: {total_tokens}")

    train_data, val_data = split_data(textdata, train_ratio=0.70)
    
    context_len = config_train['context_length']

    # Creating data loader for both train and validation
    train_loader = dataset.create_dataloader(train_data, batch_size=2, max_length=context_len, stride=context_len,
                                             drop_last=True, shuffle=True, num_workers=0)
    
    val_loader = dataset.create_dataloader(val_data, batch_size=2, max_length=context_len, stride=context_len,
                                            drop_last=False, shuffle=False, num_workers=0)
    # print("Train loader:")
    # for x, y in train_loader:
    #     print(x.shape, y.shape)

    # print("\nValidation loader:")
    # for x, y in val_loader:
    #     print(x.shape, y.shape)

    # for input_batch, target_batch in train_loader:
    #     print(input_batch.shape, target_batch.shape)
        
    # return
    # Defining optimizer
    optimizer = torch.optim.AdamW(model_llm.parameters(), lr=0.0004, weight_decay=0.1)

    # Training LLM model from scratch
    num_epochs = 10
    train_losses, val_losses, tokens_seen = train_model(model_llm, train_loader, val_loader, optimizer, device,
                                                        num_epochs=num_epochs, eval_freq=5, eval_iter=5,
                                                        start_context="Every effort moves you", tokenizer=tokenizer)
    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)

    # Saving the model, so that we can resume training later or use for inference
    model.save_model(model_llm)

if __name__ == "__main__":
    config_train = config.GPT_CONFIG_124M
    tokenizer = tokenizer_lib.get_tokenizer()
    train(tokenizer)