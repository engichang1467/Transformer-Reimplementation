import torch
import math
from torch.nn import Module, Embedding, Dropout, Parameter, Linear, ModuleList, init


class InputEmbeddings(Module):
    """Class for input embeddings.

    Args:
        d_model (int): The dimensionality of the model.
        vocab_size (int): The size of the vocabulary.

    Attributes:
        d_model (int): The dimensionality of the model.
        vocab_size (int): The size of the vocabulary.
        embedding (torch.nn.Embedding): The embedding layer.

    """

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = Embedding(vocab_size, d_model)

    def forward(self, x):
        """Forward pass of the input embeddings.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length).

        Returns:
            torch.Tensor: Embedded input tensor of shape (batch_size, sequence_length, d_model).

        """
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(Module):
    """Class for positional encoding.

    Args:
        d_model (int): The dimensionality of the model.
        seq_len (int): The length of the sequence.
        dropout (float): The dropout probability.

    Attributes:
        d_model (int): The dimensionality of the model.
        seq_len (int): The length of the sequence.
        dropout (torch.nn.Dropout): The dropout layer.
        pe (torch.Tensor): The positional encoding tensor.

    """

    def __init__(self, d_model: int, seq_len: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = Dropout(dropout)

        # Create a matrix of shape (seqlen, d_model)
        pe = torch.zeros(seq_len, d_model)
        # Create a vector of shape(seqlen, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        # Apply the sin to even positions
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply the cos to odd positions
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, seq_len, d_model)

        self.register_buffer("pe", pe)

    def forward(self, x):
        """Forward pass of the positional encoding.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Tensor with positional encoding added.

        """
        x += (self.pe[:, : x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)


class LayerNormalization(Module):
    """Class for layer normalization.

    Args:
        eps (float): The epsilon value.

    Attributes:
        eps (float): The epsilon value.
        alpha (torch.nn.Parameter): Scaling parameter.
        bias (torch.nn.Parameter): Bias parameter.

    """

    def __init__(self, feature: int, eps: float = 10**6):
        super().__init__()
        self.eps = eps
        self.alpha = Parameter(torch.ones(feature))  # Multiplied
        self.bias = Parameter(torch.zeros(feature))  # Added

    def forward(self, x):
        """Forward pass of layer normalization.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Normalized tensor.

        """
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class FeedForwardBlock(Module):
    """Class for feed forward block.

    Args:
        d_model (int): The dimensionality of the model.
        d_ff (int): The dimensionality of the feed forward layer.
        dropout (float): The dropout probability.

    Attributes:
        linear_1 (torch.nn.Linear): First linear layer.
        dropout (torch.nn.Dropout): Dropout layer.
        linear_2 (torch.nn.Linear): Second linear layer.

    """

    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.linear_1 = Linear(d_model, d_ff)  # W1 and b1
        self.dropout = Dropout(dropout)
        self.linear_2 = Linear(d_ff, d_model)  # W2 and B2

    def forward(self, x):
        """Forward pass of the feed forward block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.

        """
        # (Batch, Seq_Len, d_model) --> (Batch, Seq_Len, d_ff) --> (Batch, Seq_Len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))


class MultiHeadAttentionBlock(Module):
    """Multi-head attention block.

    Args:
        d_model (int): The dimensionality of the model.
        h (int): The number of heads.
        dropout (float): The dropout probability.

    Attributes:
        d_model (int): The dimensionality of the model.
        h (int): The number of heads.
        d_k (int): The dimensionality of keys and queries for each head.
        w_q (torch.nn.Linear): Linear layer for queries.
        w_k (torch.nn.Linear): Linear layer for keys.
        w_v (torch.nn.Linear): Linear layer for values.
        w_o (torch.nn.Linear): Linear layer for output.
        dropout (torch.nn.Dropout): Dropout layer.

    """

    def __init__(self, d_model: int, h: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_k = d_model // h
        self.w_q = Linear(d_model, d_model)  # Wq
        self.w_k = Linear(d_model, d_model)  # Wk
        self.w_v = Linear(d_model, d_model)  # Wv
        self.w_o = Linear(d_model, d_model)  # Wo
        self.dropout = Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: Dropout):
        """Compute scaled dot-product attention.

        Args:
            query (torch.Tensor): Query tensor.
            key (torch.Tensor): Key tensor.
            value (torch.Tensor): Value tensor.
            mask (torch.Tensor): Mask tensor.
            dropout (torch.nn.Dropout): Dropout layer.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple containing attended value tensor and attention scores.

        """
        d_k = query.shape[-1]

        # (Batch, h, Seq_len, d_k) --> (Batch, h, Seq_Len, Seq_Len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(
            dim=-1
        )  # (Batch, h, seq_len, seq_len)
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        """Forward pass of the multi-head attention block.

        Args:
            q (torch.Tensor): Query tensor.
            k (torch.Tensor): Key tensor.
            v (torch.Tensor): Value tensor.
            mask (torch.Tensor): Mask tensor.

        Returns:
            torch.Tensor: Output tensor.

        """
        query = self.w_q(q)  # (Batch, Seq_Len, d_model) --> (Batch, Seq_Len, d_model)
        key = self.w_k(k)  # (Batch, Seq_Len, d_model) --> (Batch, Seq_Len, d_model)
        value = self.w_v(v)  # (Batch, Seq_Len, d_model) --> (Batch, Seq_Len, d_model)

        # (Batch, Seq_Len, d_model) --> (Batch, Seq_Len, h, d_k) --> (Batch, h, Seq_Len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(
            1, 2
        )
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(
            1, 2
        )

        x, self.attention_scores = MultiHeadAttentionBlock.attention(
            query, key, value, mask, self.dropout
        )

        # (Batch, h, seq_len, d_k) --> (Batch, h, seq_len, d_k) --> (Batch, Seq_Len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # (Batch, Seq_Len, d_model) --> (Batch, Seq_Len, d_model)
        return self.w_o(x)


class ResidualConnection(Module):
    """Residual connection module.

    This module adds a residual connection to the output of a sublayer.

    Args:
        dropout (float): The dropout probability.

    Attributes:
        dropout (torch.nn.Dropout): Dropout layer.
        norm (LayerNormalization): Layer normalization module.

    """

    def __init__(self, features: int, dropout: float):
        super().__init__()
        self.dropout = Dropout(dropout)
        self.norm = LayerNormalization(features)

    def forward(self, x, sublayer):
        """Forward pass of the residual connection module.

        Args:
            x (torch.Tensor): Input tensor.
            sublayer (Callable): Sublayer function.

        Returns:
            torch.Tensor: Output tensor.

        """
        return x + self.dropout(sublayer(self.norm(x)))


# Encoder
class EncoderBlock(Module):
    """Encoder block module.

    This module represents a single block in the encoder of a transformer.

    Args:
        self_attention_block (MultiHeadAttentionBlock): Self-attention block.
        feed_forward_block (FeedForwardBlock): Feedforward block.
        dropout (float): The dropout probability.

    Attributes:
        self_attention_block (MultiHeadAttentionBlock): Self-attention block.
        feed_forward_block (FeedForwardBlock): Feedforward block.
        residual_connections (ModuleList): List of residual connections.

    """

    def __init__(
        self,
        features: int,
        self_attention_block: MultiHeadAttentionBlock,
        feed_forward_block: FeedForwardBlock,
        dropout: float,
    ):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = ModuleList(
            [ResidualConnection(features, dropout) for _ in range(2)]
        )

    def forward(self, x, src_mask):
        """Forward pass of the encoder block module.

        Args:
            x (torch.Tensor): Input tensor.
            src_mask (torch.Tensor): Source mask tensor.

        Returns:
            torch.Tensor: Output tensor.

        """
        x = self.residual_connections[0](
            x, lambda x: self.self_attention_block(x, x, x, src_mask)
        )
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x


class Encoder(Module):
    """Encoder module.

    This module represents the encoder of a transformer.

    Args:
        layers (ModuleList): List of encoder layers.

    Attributes:
        layers (ModuleList): List of encoder layers.
        norm (LayerNormalization): Layer normalization module.

    """

    def __init__(self, features: int, layers: ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, mask):
        """Forward pass of the encoder module.

        Args:
            x (torch.Tensor): Input tensor.
            mask (torch.Tensor): Mask tensor.

        Returns:
            torch.Tensor: Output tensor.

        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


# Decoder
class DecoderBlock(Module):
    """Decoder block module.

    This module represents a single block in the decoder of a transformer.

    Args:
        features (int): Number of features.
        self_attention_block (MultiHeadAttentionBlock): Self-attention block.
        cross_attention_block (MultiHeadAttentionBlock): Cross-attention block.
        feed_forward_block (FeedForwardBlock): Feedforward block.
        dropout (float): The dropout probability.

    Attributes:
        self_attention_block (MultiHeadAttentionBlock): Self-attention block.
        cross_attention_block (MultiHeadAttentionBlock): Cross-attention block.
        feed_forward_block (FeedForwardBlock): Feedforward block.
        residual_connections (ModuleList): List of residual connections.

    """

    def __init__(
        self,
        features: int,
        self_attention_block: MultiHeadAttentionBlock,
        cross_attention_block: MultiHeadAttentionBlock,
        feed_forward_block: FeedForwardBlock,
        dropout: float,
    ):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = ModuleList(
            [ResidualConnection(features, dropout) for _ in range(3)]
        )

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        """Forward pass of the decoder block module.

        Args:
            x (torch.Tensor): Input tensor.
            encoder_output (torch.Tensor): Encoder output tensor.
            src_mask (torch.Tensor): Source mask tensor.
            tgt_mask (torch.Tensor): Target mask tensor.

        Returns:
            torch.Tensor: Output tensor.

        """
        x = self.residual_connections[0](
            x, lambda x: self.self_attention_block(x, x, x, tgt_mask)
        )
        x = self.residual_connections[1](
            x,
            lambda x: self.cross_attention_block(
                x, encoder_output, encoder_output, src_mask
            ),
        )
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x


class Decoder(Module):
    """Decoder module.

    This module represents the decoder of a transformer.

    Args:
        layers (ModuleList): List of decoder layers.

    Attributes:
        layers (ModuleList): List of decoder layers.
        norm (LayerNormalization): Layer normalization module.

    """

    def __init__(self, features: int, layers: ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        """Forward pass of the decoder module.

        Args:
            x (torch.Tensor): Input tensor.
            encoder_output (torch.Tensor): Encoder output tensor.
            src_mask (torch.Tensor): Source mask tensor.
            tgt_mask (torch.Tensor): Target mask tensor.

        Returns:
            torch.Tensor: Output tensor.

        """
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)


# Linear Layer
class ProjectionLayer(Module):
    """Projection layer module.

    This module represents the final projection layer of the transformer.

    Args:
        d_model (int): The dimensionality of the model.
        vocab_size (int): The size of the vocabulary.

    Attributes:
        proj (torch.nn.Linear): Linear projection layer.

    """

    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.proj = Linear(d_model, vocab_size)

    def forward(self, x):
        """Forward pass of the projection layer module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.

        """
        # (Batch, Seq_Len, d_model) --> (Batch, Seq_Len, Vocab_Size)
        return self.proj(x)


# Transformer
class Transformer(Module):
    """Transformer module.

    This module represents the Transformer model.

    Args:
        encoder (Encoder): Encoder module.
        decoder (Decoder): Decoder module.
        src_embed (InputEmbeddings): Source input embeddings module.
        tgt_embed (InputEmbeddings): Target input embeddings module.
        src_pos (PositionalEncoding): Source positional encoding module.
        tgt_pos (PositionalEncoding): Target positional encoding module.
        projection_layer (ProjectionLayer): Projection layer module.

    Attributes:
        encoder (Encoder): Encoder module.
        decoder (Decoder): Decoder module.
        src_embed (InputEmbeddings): Source input embeddings module.
        tgt_embed (InputEmbeddings): Target input embeddings module.
        src_pos (PositionalEncoding): Source positional encoding module.
        tgt_pos (PositionalEncoding): Target positional encoding module.
        projection_layer (ProjectionLayer): Projection layer module.

    """

    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        src_embed: InputEmbeddings,
        tgt_embed: InputEmbeddings,
        src_pos: PositionalEncoding,
        tgt_pos: PositionalEncoding,
        projection_layer: ProjectionLayer,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        """Encode input sequence.

        Args:
            src (torch.Tensor): Source input tensor.
            src_mask (torch.Tensor): Source mask tensor.

        Returns:
            torch.Tensor: Encoded tensor.

        """
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor):
        """Decode target sequence.

        Args:
            encoder_output (torch.Tensor): Encoder output tensor.
            src_mask (torch.Tensor): Source mask tensor.
            tgt (torch.Tensor): Target input tensor.
            tgt_mask (torch.Tensor): Target mask tensor.

        Returns:
            torch.Tensor: Decoded tensor.

        """
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x):
        """Project tensor to vocabulary space.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Projected tensor.

        """
        return self.projection_layer(x)


def build_transformer(
    src_vocab_size: int,
    tgt_vocab_size: int,
    src_seq_len: int,
    tgt_seq_len: int,
    d_model: int = 512,
    N: int = 6,
    h: int = 8,
    dropout: float = 0.1,
    d_ff: int = 2048,
) -> Transformer:
    """Builds a Transformer model.

    Args:
        src_vocab_size (int): Size of the source vocabulary.
        tgt_vocab_size (int): Size of the target vocabulary.
        src_seq_len (int): Length of the source sequence.
        tgt_seq_len (int): Length of the target sequence.
        d_model (int, optional): Dimensionality of the model. Defaults to 512.
        N (int, optional): Number of encoder and decoder blocks. Defaults to 6.
        h (int, optional): Number of attention heads. Defaults to 8.
        dropout (float, optional): Dropout rate. Defaults to 0.1.
        d_ff (int, optional): Dimensionality of the feed-forward layer. Defaults to 2048.

    Returns:
        Transformer: The constructed Transformer model.

    """
    # Create the embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # Create the positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(
            d_model, encoder_self_attention_block, feed_forward_block, dropout
        )
        encoder_blocks.append(encoder_block)

    # create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(
            d_model,
            decoder_self_attention_block,
            decoder_cross_attention_block,
            feed_forward_block,
            dropout,
        )
        decoder_blocks.append(decoder_block)

    encoder = Encoder(d_model, ModuleList(encoder_blocks))
    decoder = Decoder(d_model, ModuleList(decoder_blocks))

    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    transformer = Transformer(
        encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer
    )

    # Initialize the parameters
    for parameter in transformer.parameters():
        if parameter.dim() > 1:
            init.xavier_uniform(parameter)

    return transformer
