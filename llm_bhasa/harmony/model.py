import os
import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin
from llm_bhasa.harmony.attention import MultiHeadAttention

class GELU(nn.Module):
    """
    Gaussian Error Linear Unit (GELU) activation function.

    GELU is a smooth, non-monotonic activation function that approximates
    the expected transformation of a neuron's input by randomly applying
    either the identity or zero transformation, depending on the input's value.
    It's known for its strong performance in transformer models.

    The GELU activation is defined as:
    GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

    Attributes:
        None
    """
    def __init__(self):
        """
        Initializes the GELU activation function.
        """
        super().__init__()

    def forward(self, x):
        """
        Applies the GELU activation function to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying the GELU activation.
        """
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * 
            (x + 0.044715 * torch.pow(x, 3))
        ))

class LayerNorm(nn.Module):
    """
    Layer Normalization module.

    Layer normalization normalizes the activations of a layer across the
    feature dimension. It stabilizes and accelerates the training of deep
    neural networks.

    Args:
        emb_dim (int): The embedding dimension (number of features).

    Attributes:
        eps (float): A small constant added to the variance to prevent
            division by zero (default: 1e-5).
        scale (nn.Parameter): A trainable scaling parameter.
        shift (nn.Parameter): A trainable shifting parameter.
    """
    def __init__(self, emb_dim):
        """
        Initializes the LayerNorm module.

        Args:
            emb_dim (int): The embedding dimension.
        """
        super().__init__()
        
        # eps is a small constant (epsilon) added to the variance to prevent division by zero during normalization
        self.eps = 1e-5

        # Trainable parameters
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        """
        Applies layer normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after layer normalization.
        """

        # Normalization operates on the last dimension of the input tensor x
        # which represents the embedding dimension (emb_dim)
        
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)

        return self.scale * norm_x + self.shift

class FeedForward(nn.Module):
    """
    Feed-forward neural network module.

    This module implements a feed-forward neural network with two linear
    layers and a GELU activation function in between. It's a standard
    component in transformer blocks.

    Args:
        cfg (dict): A dictionary containing configuration parameters, including
            "emb_dim" (embedding dimension).

    Attributes:
        layers (nn.Sequential): A sequential container of the linear layers
            and GELU activation.
    """
    def __init__(self, cfg):
        """
        Initializes the FeedForward module.

        Args:
            cfg (dict): Configuration dictionary.
        """
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        """
        Forward pass of the FeedForward module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        return self.layers(x)

class TransformerBlock(nn.Module):
    """
    Transformer block module.

    This module implements a single transformer block, which consists of a
    multi-head attention layer, a feed-forward layer, layer normalization,
    and residual connections.

    Args:
        cfg (dict): A dictionary containing configuration parameters, including
            "emb_dim" (embedding dimension), "context_length" (maximum sequence
            length), "n_heads" (number of attention heads), "drop_rate"
            (dropout rate), and "qkv_bias" (whether to use bias in query, key,
            value projections).

    Attributes:
        att (MultiHeadAttention): The multi-head attention layer.
        ff (FeedForward): The feed-forward layer.
        norm1 (LayerNorm): The first layer normalization layer.
        norm2 (LayerNorm): The second layer normalization layer.
        drop_shortcut (nn.Dropout): Dropout layer for residual connections.
    """
    def __init__(self, cfg):
        """
        Initializes the TransformerBlock module.

        Args:
            cfg (dict): Configuration dictionary.
        """
        super().__init__()
        
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"], 
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        """
        Forward pass of the TransformerBlock module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """

        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x

class LLMModel(nn.Module, PyTorchModelHubMixin, repo_url="pankajr141/llm_bhasa_v1", 
               pipeline_tag="text-generation", license="mit"):
    """
    Large Language Model (LLM) module.

    This module implements a large language model based on the transformer
    architecture. It consists of token and positional embeddings, multiple
    transformer blocks, layer normalization, and an output head.

    Args:
        cfg (dict): A dictionary containing configuration parameters, including
            "vocab_size" (vocabulary size), "emb_dim" (embedding dimension),
            "context_length" (maximum sequence length), "n_heads" (number of
            attention heads), "n_layers" (number of transformer blocks), and
            "drop_rate" (dropout rate).

    Attributes:
        tok_emb (nn.Embedding): Token embedding layer.
        pos_emb (nn.Embedding): Positional embedding layer.
        drop_emb (nn.Dropout): Dropout layer for embeddings.
        trf_blocks (nn.Sequential): Sequential container of transformer blocks.
        final_norm (LayerNorm): Final layer normalization layer.
        out_head (nn.Linear): Output linear layer.
    """
    def __init__(self, cfg):
        """
        Initializes the LLMModel module.

        Args:
            cfg (dict): Configuration dictionary.
        """
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        """
        Forward pass of the LLMModel module.

        Args:
            in_idx (torch.Tensor): Input tensor of token indices.

        Returns:
            torch.Tensor: Output tensor of logits.
        """
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)

        pos_embeds = self.pos_emb(
            torch.arange(seq_len, device=in_idx.device)
        )
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

def print_model_information(model):
    """
    Prints information about the model.

    This function prints the total number of parameters, the shape of the
    token embedding layer, the shape of the output layer, the number of
    trainable parameters considering weight tying, and the total size of the
    model in MB.

    Weight tying refers to sharing the weights between the token embedding
    layer and the output layer.

    Args:
        model (nn.Module): The model to print information about.
    """
    print("##============== Model Summary =========================##")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"# Total number of parameters: {total_params:,}")

    print("# Token embedding layer shape:", model.tok_emb.weight.shape)
    print("# Output layer shape:", model.out_head.weight.shape)

    total_params_gpt2 = ( total_params - sum(p.numel() for p in model.out_head.parameters()))
    print(f"# Number of trainable parameters considering weight tying: {total_params_gpt2:,}")


    # Assuming each parameter is a 32-bit float taking up 4 bytes
    total_size_bytes = total_params * 4
    total_size_mb = total_size_bytes / (1024 * 1024)
    print(f"# Total size of the model: {total_size_mb:.2f} MB")
    print("##======================================================##")

def save_model(model, optimizer, modelfile="model_and_optimizer.pth"):
    """
    Saves the model and optimizer state to a file.

    Args:
        model (nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer to save.
        modelfile (str, optional): The file path to save the model to.
            Defaults to "model_and_optimizer.pth".
    """
    torch.save({
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                modelfile
    )
    print(f"Model saved to {modelfile}")

def load_model(model, modelfile="model_and_optimizer.pth"):
    """
    Loads the model and optimizer state from a file.

    Args:
        model (nn.Module): The model to load the state into.
        modelfile (str, optional): The file path to load the model from.
            Defaults to "model_and_optimizer.pth".

    Returns:
        nn.Module: The model with loaded state, or the original model if the
            file doesn't exist.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(modelfile):
        print(f"Model cannot be loaded as {modelfile} doesnt exist")
        return model

    print(f"Loading Model weights from {modelfile}")

    checkpoint = torch.load("model_and_optimizer.pth", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    model.train()
    return model

if __name__ == "__main__":
    from bhasa import config
    torch.manual_seed(123)
    model = LLMModel(config.GPT_CONFIG_124M)
    
    print_model_information(model)
    batch = torch.randint(0, 100, (2, 5))

    out = model(batch)
    print("Input batch:\n", batch)
    print("\nOutput shape:", out.shape)
    print(out)
