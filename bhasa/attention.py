import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention module.

    Args:
        d_in (int): Input dimension.
        d_out (int): Output dimension.
        context_length (int): Maximum length of the context.
        dropout (float): Dropout probability.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional): Whether to use bias in the query, key, and value projections. Defaults to False.

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, num_tokens, d_out).
    """
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        
        # Check if d_out is divisible by num_heads
        assert (d_out % num_heads == 0),  "d_out must be divisible by num_heads"

        self.d_out = d_out # Output dimension
        self.num_heads = num_heads # Number of attention heads
        
        # Reduces the projection dim to match the desired output dim
        self.head_dim = d_out // num_heads 

        # Attention weights
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        # Uses a Linear layer to combine head outputs
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), # Create a mask for causal attention
                       diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        # Tensor shape: (b, num_tokens, d_out)
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # Reshape to add num_heads dimension: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim) 
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose to shape (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2) 
        queries = queries.transpose(1, 2) 
        values = values.transpose(1, 2) 

        attn_scores = queries @ keys.transpose(2, 3)               # Compute attention scores (cross product)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]     # Truncate mask to the number of tokens
        attn_scores.masked_fill_(mask_bool, -torch.inf)            # Uses the mask to fill attention scores
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)    # Tensor shape: (b, num_tokens, num_heads, head_dim)

        # Combines heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(
            b, num_tokens, self.d_out
        )
        context_vec = self.out_proj(context_vec) # Apply output projection
        return context_vec

if __name__ == "__main__":
    torch.manual_seed(123)
    batch = torch.randn(2, 5, 4)
    batch_size, context_length, d_in = batch.shape
    d_out = 4
    mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)
    context_vecs = mha(batch)
    print(context_vecs)
    print("context_vecs.shape:", context_vecs.shape)
