import torch
import torch.nn as nn
import torch.nn.functional as F


class LuongAttention(nn.Module):
    """Luong multiplicative attention"""

    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, decoder_hidden, encoder_outputs, src_mask=None):
        """
        Args:
            decoder_hidden: (batch_size, hidden_dim)
            encoder_outputs: (batch_size, src_len, hidden_dim)
            src_mask: (batch_size, src_len) - 1 for real tokens, 0 for padding

        Returns:
            context: (batch_size, hidden_dim)
            attn_weights: (batch_size, src_len)
        """
        # Align encoder outputs
        # (batch_size, src_len, hidden_dim)
        aligned = self.attn(encoder_outputs)

        # Compute attention scores
        # decoder_hidden: (batch_size, hidden_dim) -> (batch_size, hidden_dim, 1)
        decoder_hidden = decoder_hidden.unsqueeze(2)

        # scores: (batch_size, src_len, hidden_dim) x (batch_size, hidden_dim, 1)
        #       = (batch_size, src_len, 1)
        scores = torch.bmm(aligned, decoder_hidden).squeeze(2)  # (batch_size, src_len)

        # Mask padding positions
        if src_mask is not None:
            scores = scores.masked_fill(src_mask == 0, -1e10)

        # Normalize to get attention weights
        attn_weights = F.softmax(scores, dim=1)  # (batch_size, src_len)

        # Compute context vector (weighted sum of encoder outputs)
        # attn_weights: (batch_size, 1, src_len)
        # encoder_outputs: (batch_size, src_len, hidden_dim)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)  # (batch_size, 1, hidden_dim)
        context = context.squeeze(1)  # (batch_size, hidden_dim)

        return context, attn_weights


class Seq2Seq(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, emb_dim=300, hid_dim=256):
        super().__init__()

        self.hid_dim = hid_dim

        self.src_emb = nn.Embedding(src_vocab, emb_dim)
        self.tgt_emb = nn.Embedding(tgt_vocab, emb_dim)

        self.encoder = nn.LSTM(emb_dim, hid_dim, batch_first=True)

        # Attention module
        self.attention = LuongAttention(hid_dim)

        # Decoder now takes [embedding; context]
        self.decoder = nn.LSTM(emb_dim + hid_dim, hid_dim, batch_first=True)

        # Output layer takes [decoder_output; context]
        self.fc = nn.Linear(hid_dim + hid_dim, tgt_vocab)

    def forward(self, src, tgt, src_mask=None):
        """
        Args:
            src: (batch_size, src_len)
            tgt: (batch_size, tgt_len)
            src_mask: (batch_size, src_len)

        Returns:
            output: (batch_size, tgt_len, vocab_size)
        """
        # Encode
        src_emb = self.src_emb(src)
        encoder_outputs, (h, c) = self.encoder(src_emb)
        # encoder_outputs: (batch_size, src_len, hid_dim)

        # Decode with attention
        tgt_emb = self.tgt_emb(tgt)  # (batch_size, tgt_len, emb_dim)

        batch_size = tgt.size(0)
        tgt_len = tgt.size(1)

        # We'll process decoder step-by-step to apply attention
        outputs = []

        for t in range(tgt_len):
            # Current target embedding
            input_emb = tgt_emb[:, t:t + 1, :]  # (batch_size, 1, emb_dim)

            # Compute attention using current hidden state
            # h: (1, batch_size, hid_dim) -> squeeze -> (batch_size, hid_dim)
            context, attn_weights = self.attention(h.squeeze(0), encoder_outputs, src_mask)

            # Concatenate embedding with context
            # context: (batch_size, hid_dim) -> (batch_size, 1, hid_dim)
            decoder_input = torch.cat([input_emb, context.unsqueeze(1)], dim=2)
            # decoder_input: (batch_size, 1, emb_dim + hid_dim)

            # LSTM step
            decoder_output, (h, c) = self.decoder(decoder_input, (h, c))
            # decoder_output: (batch_size, 1, hid_dim)

            # Concatenate decoder output with context for prediction
            prediction_input = torch.cat([decoder_output.squeeze(1), context], dim=1)
            # prediction_input: (batch_size, hid_dim + hid_dim)

            output = self.fc(prediction_input)  # (batch_size, vocab_size)
            outputs.append(output)

        # Stack all outputs
        outputs = torch.stack(outputs, dim=1)  # (batch_size, tgt_len, vocab_size)

        return outputs

    def decode_step(self, input_token, h, c, encoder_outputs, src_mask=None):
        """
        Single decoding step for inference (beam search)

        Args:
            input_token: (batch_size, 1) - single token
            h, c: decoder states
            encoder_outputs: (batch_size, src_len, hid_dim)
            src_mask: (batch_size, src_len)

        Returns:
            logits: (batch_size, vocab_size)
            h, c: updated states
            attn_weights: (batch_size, src_len)
        """
        # Embed input
        input_emb = self.tgt_emb(input_token)  # (batch_size, 1, emb_dim)

        # Compute attention
        context, attn_weights = self.attention(h.squeeze(0), encoder_outputs, src_mask)

        # Concatenate embedding with context
        decoder_input = torch.cat([input_emb, context.unsqueeze(1)], dim=2)

        # LSTM step
        decoder_output, (h, c) = self.decoder(decoder_input, (h, c))

        # Prediction
        prediction_input = torch.cat([decoder_output.squeeze(1), context], dim=1)
        logits = self.fc(prediction_input)

        return logits, h, c, attn_weights