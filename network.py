import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10000):
        super(PositionalEncoding, self).__init__()

        # Init matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)

        # calc sinosudial values
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # Apply sine to even indices and cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape (1, max_len, d_model)

        # Add batch dimension
        self.register_buffer("pe", pe)

    def forward(self, x):
        # Add positional encoding
        x + self.pe[:, : x.size(1)]
        return x


class EncoderOnly(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=512,
        nhead=8,
        num_layers=3,
        dim_feedforward=2048,
        dropout=0.1,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True
        )
        norm = nn.LayerNorm(d_model)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers, norm=norm)

    def forward(self, src_ids, src_key_padding_mask=None):
        x = self.embedding(src_ids) * math.sqrt(self.embedding.embedding_dim)
        x = self.pos_encoder(x)
        memory = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        return memory


class Seq2SeqTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=512,
        nhead=8,
        num_encoder_layers=3,
        num_decoder_layers=3,
        dim_feedforward=2048,
        dropout=0.1,
    ):
        super().__init__()
        self.model_type = "Transformer"
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.d_model = d_model

    def forward(self, src, tgt, src_key_padding_mask, tgt_key_padding_mask):
        # src, tgt: (batch, seq) long tensors of token IDs
        src_emb = self.embedding(src) * math.sqrt(self.d_model)
        tgt_emb = self.embedding(tgt) * math.sqrt(self.d_model)
        # Generate causal mask for target (square subsequent mask)
        tgt_seq_len = tgt.size(1)
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt_seq_len).to(
            src.device
        )
        tgt_mask = (
            tgt_mask.masked_fill(tgt_mask == float("-inf"), True)
            .masked_fill(tgt_mask == 0, False)
            .bool()
        )
        output = self.transformer(
            src_emb,
            tgt_emb,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
            tgt_mask=tgt_mask,
        )
        output = self.fc_out(output)  # (batch, tgt_seq, vocab_size)
        return output


class TransformerAutoencoder(nn.Module):
    def __init__(
        self, vocab_size, d_model=512, nhead=8, num_layers=6, dim_ff=2048, dropout=0.1
    ):
        super(TransformerAutoencoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_ff, dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_ff, dropou)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.out_proj = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        src,
        tgt,
        src_key_padding_mask=None,
        tgt_mask=None,
        tgt_key_padding_mask=None,
    ):
        # src, tgt: (batch, seq_len, d_model)
        src_emb = self.pos_encoder(self.embedding(src)).transpose(
            0, 1
        )  # (seq_len, batch, d_model)
        tgt_emb = self.pos_encoder(self.embedding(tgt)).transpose(0, 1)
        memory = self.encoder(
            src_emb, src_key_padding_mask=src_key_padding_mask
        )  # encode

        out = self.decoder(
            tgt_emb,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )
        out = out.transpose(0, 1)  # (batch, seq_len, d_model)
        return self.out_proj(out)  # logits
