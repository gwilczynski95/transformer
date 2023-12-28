import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence


class LinearLayer(nn.Module):
    def __init__(self, in_dimension, out_dimension, weights_initialization="glorot_uniform",
                 bias_initialization="zeros", use_bias=True, scaling_factor=1, freeze=False):
        super().__init__()

        self.in_dimension = in_dimension
        self.out_dimension = out_dimension
        self.scaling_factor = scaling_factor
        self.use_bias = use_bias
        self.freeze = freeze

        # initialize weights
        assert weights_initialization in ["he", "glorot_normal", "normal", "ones", "glorot_uniform"]
        std = 1
        if weights_initialization == "glorot_normal":
            fan_avg = (self.in_dimension + self.out_dimension) / 2
            std = np.sqrt(1 / fan_avg)
        elif weights_initialization == "he":
            std = np.sqrt(2 / in_dimension)
        elif weights_initialization == "normal":
            std = 1
        weights = np.random.normal(0, std, [in_dimension, out_dimension]).astype(np.float32)
        if weights_initialization == "ones":
            weights = np.ones([in_dimension, out_dimension], dtype=np.float32)
        elif weights_initialization == "glorot_uniform":
            fan_avg = (self.in_dimension + self.out_dimension) / 2
            r = np.sqrt(3 / fan_avg)
            weights = np.random.uniform(low=-r, high=r, size=[in_dimension, out_dimension]).astype(np.float32)

        self.weights = torch.from_numpy(weights)
        self.weights = nn.Parameter(self.weights)
        self.weights.requires_grad = not freeze

        if use_bias:
            assert bias_initialization in ["zeros"]
            if bias_initialization == "zeros":
                bias = np.zeros(out_dimension, dtype=np.float32)

            self.bias = torch.from_numpy(bias)
            self.bias = nn.Parameter(self.bias)
            self.bias.requires_grad = not freeze

    def forward(self, x):
        # x.shape = [batch_size, in_dimension]
        # weights.shape = [in_dimension, out_dimension]
        _out = torch.matmul(x, self.weights * self.scaling_factor)
        if self.use_bias:
            _out = torch.add(_out, self.bias)
        return _out


class ScaledEmbedding(nn.Module):
    def __init__(self, embed_size, model_dim, padding_idx):
        super().__init__()
        self.embed_size = embed_size
        self.model_dim = model_dim
        self.padding_idx = padding_idx

        self.embedding = nn.Embedding(
            embed_size,
            model_dim,
            padding_idx=padding_idx
        )
        for p in self.embedding.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        self.model_dim = torch.tensor(self.model_dim, dtype=torch.float32)

    def forward(self, x):
        return self.embedding(x) * torch.sqrt(self.model_dim).to(x.device)


class PositionWiseFeedForward(nn.Module):
    def __init__(self, in_dim, mid_dim, out_dim, weights_initialization="glorot_uniform",
                 bias_initialization="zeros"):
        super().__init__()
        self.layer1 = LinearLayer(
            in_dim,
            mid_dim,
            weights_initialization=weights_initialization,
            bias_initialization=bias_initialization,
            use_bias=True
        )
        self.layer2 = LinearLayer(
            mid_dim,
            out_dim,
            weights_initialization=weights_initialization,
            bias_initialization=bias_initialization,
            use_bias=True
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        _out = self.layer1(x)
        _out = self.relu(_out)
        return self.layer2(_out)


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        query_key = torch.matmul(
            query,  # [BS, AH, NOW, DK]
            key.transpose(-2, -1)  # [BS, AH, NOW, DK] -> [BS, AH, DK, NOW]
        )  # [BS, AH, NOW, NOW]
        query_key /= query.shape[-1]  # divide by DK
        if mask is not None:
            query_key = query_key.masked_fill(
                mask == 0, -1e9
            )
        scores = self.softmax(query_key)
        scores = self.dropout(scores)
        return torch.matmul(scores, value)

    def legacy_forward_1(self, query, key, value, timestep=None):
        # inp [batch_size, num_of_tokens, inp_dim]
        query_pad_mask = torch.any(query, dim=2, keepdim=True)
        query_pad_mask = torch.broadcast_to(query_pad_mask, query.shape).float()
        key_pad_mask = torch.any(key, dim=2, keepdim=True)
        key_pad_mask = torch.broadcast_to(key_pad_mask, key.shape).float()
        pad_mask_mul = torch.bmm(query_pad_mask, torch.transpose(key_pad_mask, 2, 1))
        Q = self.layer_Q(query)
        K = self.layer_K(key)
        V = self.layer_V(value)
        _big_mul = torch.bmm(Q, torch.transpose(K, 2, 1))
        _big_mul_scaled = _big_mul / torch.sqrt(self.keys_dim)
        if timestep is not None:
            # inf_mask = torch.full(_big_mul_scaled.shape, float("-inf"))
            # inf_mask[:, :timestep + 1, :timestep + 1] = 0
            # _big_mul_scaled = _big_mul_scaled + inf_mask
            # scores = scores.masked_fill(mask == 0, -1e9)
            inf_mask = torch.zeros(_big_mul_scaled.shape, device=query.device)
            inf_mask[:, :timestep + 1, :timestep + 1] = 1
            _big_mul_scaled = _big_mul_scaled.detach().masked_fill(inf_mask == 0, -1e9)
        # _big_mul_scaled[pad_mask_mul == 0] = torch.Tensor([float("-inf")])
        _big_mul_scaled = _big_mul_scaled.detach().masked_fill(pad_mask_mul == 0, -1e9)
        _big_mul_softed = nn.functional.softmax(_big_mul_scaled, dim=-1)
        if timestep is not None:
            _big_mul_softed[:, timestep + 1:, :] = 0.
        # _big_mul_softed[pad_mask_mul == 0] = 0.
        _big_mul_softed = _big_mul_softed.detach().masked_fill(pad_mask_mul == 0, 0.)
        _big_tokens = torch.bmm(_big_mul_softed, V)
        return _big_tokens

    def legacy_forward_2(self, query, key, value):
        # inp [batch_size, num_of_tokens, inp_dim]
        out = []
        big_Q = self.layer_Q(query)
        big_K = self.layer_K(key)
        big_V = self.layer_V(value)

        _big_mul = torch.bmm(big_Q, torch.transpose(big_K, 2, 1))
        _big_mul_scaled = _big_mul / torch.sqrt(self.keys_dim)
        _big_mul_softed = nn.functional.softmax(_big_mul_scaled, dim=-1)
        _big_tokens = torch.bmm(_big_mul_softed, big_V)

        for batch_idx in range(query.shape[0]):
            Q = self.layer_Q(query[batch_idx, :, :])
            K = self.layer_K(key[batch_idx, :, :])
            V = self.layer_V(value[batch_idx, :, :])

            assert torch.min(Q == big_Q[batch_idx]).item()
            assert torch.min(K == big_K[batch_idx]).item()
            assert torch.min(V == big_V[batch_idx]).item()

            _mul = torch.matmul(Q, K.T)
            assert torch.min(_mul == _big_mul[batch_idx]).item()
            _mul_scaled = _mul / torch.sqrt(self.keys_dim)
            assert torch.min(_mul_scaled == _big_mul_scaled[batch_idx]).item()
            _mul_softed = nn.functional.softmax(_mul_scaled, dim=-1)
            assert torch.min(_mul_softed == _big_mul_softed[batch_idx]).item()
            tokens = torch.matmul(_mul_softed, V)
            assert torch.min(tokens == _big_tokens[batch_idx]).item()

            out.append(tokens[None, :, :])
        out = torch.cat(out, dim=0)
        assert torch.min(out == _big_tokens).item()
        return _big_tokens


class MultiHeadAttention(nn.Module):
    def __init__(self, num_attention_heads, model_dim, dropout, weights_initialization="glorot_uniform"):
        super().__init__()
        assert num_attention_heads in [1, 2, 4, 8]
        self.num_attention_heads = num_attention_heads
        self.model_dim = model_dim
        self.dim_per_head = int(self.model_dim / self.num_attention_heads)
        self.head_dim = model_dim // num_attention_heads
        self.dropout = dropout

        self.linears = nn.ModuleList([
            LinearLayer(
                model_dim, model_dim, weights_initialization=weights_initialization, use_bias=False
            ) for _ in range(4)
        ])

        self.attn = ScaledDotProductAttention(self.dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # to handle attention heads dimension
            # NOW = number of words
            # src_mask: [BS, 1, NOW] -> [BS, 1, 1, NOW]
            # tgt_mask: [BS, NOW, NOW] -> [BS, 1, NOW, NOW]
            mask = mask.unsqueeze(1)

        # out of every layer is [BS, NOW, MD]
        # after view and transpose [BS, NOW, MD] -> [BS, AH, NOW, DK]
        _bs = query.shape[0]
        query, key, value = [
            layer(x).view(_bs, -1, self.num_attention_heads, self.head_dim).transpose(1, 2)
            for layer, x in zip(self.linears[:-1], (query, key, value))
        ]

        x = self.attn(query, key, value, mask)
        # [BS, AH, NOW, DK] -> [BS, NOW, MD]
        x = x.transpose(1, 2).contiguous().view(
            _bs, -1, self.num_attention_heads * self.head_dim
        )
        return self.linears[-1](x)


class PositionalEncodings:
    def __init__(self, out_dimension, device):
        self.out_dimension = out_dimension

        # generate positional encodings
        pos_vector = np.arange(0, out_dimension, 1)
        embeddings = np.zeros([out_dimension, out_dimension], dtype=np.float32)
        for i in range(out_dimension // 2):
            denom = 10000 ** (2 * i / out_dimension)
            embeddings[:, 2 * i] = np.sin(pos_vector / denom)
            embeddings[:, 2 * i + 1] = np.cos(pos_vector / denom)
        self.embeddings = torch.from_numpy(embeddings).to(device)

    def get_positional_encodings(self, sequence_lengths):
        """
        :param sequence_lengths: list of integers that represents len of sequences for the whole batch
        :return: list of torch.Tensor
        """
        out = []
        for sequence_length in sequence_lengths:
            out.append(
                self.embeddings[:sequence_length, :]
            )
        return out


class LayerNormalization(nn.Module):
    def __init__(self, model_dim, eps=1e-6):
        super().__init__()
        self.model_dim = model_dim
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(self.model_dim, dtype=torch.float32))
        self.beta = nn.Parameter(torch.zeros(self.model_dim, dtype=torch.float32))

    def forward(self, x):
        # [BS, NUM_OF_SEQ, MODEL_DIM]
        # mean.shape = [bs, num_of_seq]
        mean = torch.mean(x, -1)
        # std.shape = [bs, num_of_seq]
        std = torch.std(x, -1) + self.eps
        norm_x = (x - mean[:, :, None]) / std[:, :, None]  # expanding dims to enable proper broadcast
        out_x = self.gamma * norm_x + self.beta
        return out_x


class EncoderBlock(nn.Module):
    def __init__(self, model_dim=512, attention_heads=8, pwff_mid_dim=2048, dropout_rate=0.1):
        """
        Encoder block for the Transformer model
        :param model_dim: Dimension of the embeddings and tokens
        :param attention_heads: Number of attention heads
        :param pwff_mid_dim: Dimension of the middle layer of Position-Wise Feed Forward layer
        :param dropout_rate: Dropout rate
        """
        super().__init__()

        self.model_dim = model_dim
        self.attention_heads = attention_heads
        self.pwff_mid_dim = pwff_mid_dim
        self.dropout_rate = dropout_rate

        self.dropout_layer = nn.Dropout(p=dropout_rate)
        self.ln1 = LayerNormalization(model_dim)
        self.ln2 = LayerNormalization(model_dim)

        self.mha = MultiHeadAttention(
            num_attention_heads=attention_heads,
            model_dim=model_dim,
            dropout=dropout_rate
        )
        self.pwff = PositionWiseFeedForward(
            in_dim=model_dim,
            mid_dim=pwff_mid_dim,
            out_dim=model_dim
        )

    def forward(self, x, src_mask):
        x_skip = x
        x = self.mha(x, x, x, src_mask)
        x = self.dropout_layer(x)
        x = x + x_skip
        # todo: in harvard transformer normalization is after creating the skip connection not before it
        x = self.ln1(x)
        x_skip = x

        x = self.pwff(x)
        x = self.dropout_layer(x)
        x = x + x_skip

        x = self.ln2(x)

        return x


class Encoder(nn.Module):  # TODO: WHY I HAVE THOSE SINUSOIDAL PATTERNS
    def __init__(self, embed_size, padding_idx, encoder_blocks=6, model_dim=512, attention_heads=8, pwff_mid_dim=2048,
                 dropout_rate=0.1, device=None):
        """
        Encoder for the Transformer model
        :param embedding_layer: Embedding layer
        :param encoder_blocks: Number of encoder blocks
        :param model_dim: Dimension of the embeddings and tokens
        :param attention_heads: Number of attention heads
        :param pwff_mid_dim: Dimension of the middle layer of Position-Wise Feed Forward layer
        :param dropout_rate: Dropout rate
        :param device: Torch device
        """
        super().__init__()
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        self.embed_layer = ScaledEmbedding(
            embed_size,
            model_dim,
            padding_idx
        )
        self.model_dim = model_dim
        self.attention_heads = attention_heads
        self.pwff_mid_dim = pwff_mid_dim
        self.dropout_rate = dropout_rate

        self.enc_blocks = []
        for _ in range(encoder_blocks):
            self.enc_blocks.append(
                EncoderBlock(
                    model_dim=model_dim,
                    attention_heads=attention_heads,
                    pwff_mid_dim=pwff_mid_dim,
                    dropout_rate=dropout_rate,
                ).to(self.device)
            )
        self.enc_blocks = nn.ModuleList(self.enc_blocks)
        self.positional_encoder = PositionalEncodings(model_dim, self.device)
        self.dropout_layer = nn.Dropout(p=dropout_rate).to(self.device)

    def forward(self, x, src_lens, src_mask):
        """
        :param x: Input tokens to Encoder layer
        :param src_lens: Len of every input sentence (for Positional Encoding's sake)
        :return:
        """
        x = self.embed_layer(x)
        pos_encodings = self.positional_encoder.get_positional_encodings(src_lens)
        pos_encodings = torch.transpose(pad_sequence(pos_encodings, padding_value=0.), 1, 0)

        x = x + pos_encodings
        x = self.dropout_layer(x)

        for enc_block in self.enc_blocks:
            x = enc_block(x, src_mask)

        return x


class DecoderBlock(nn.Module):
    def __init__(self, model_dim=512, attention_heads=8, pwff_mid_dim=2048, dropout_rate=0.1):
        """
        Decoder block for the Transformer model
        :param model_dim: Dimension of the embeddings and tokens
        :param attention_heads: Number of attention heads
        :param pwff_mid_dim: Dimension of the middle layer of Position-Wise Feed Forward layer
        :param dropout_rate: Dropout rate
        """
        super().__init__()
        self.model_dim = model_dim
        self.attention_heads = attention_heads
        self.pwff_mid_dim = pwff_mid_dim
        self.dropout_rate = dropout_rate

        self.dropout_layer = nn.Dropout(p=dropout_rate)
        self.ln1 = LayerNormalization(model_dim)
        self.ln2 = LayerNormalization(model_dim)
        self.ln3 = LayerNormalization(model_dim)

        self.d_mha = MultiHeadAttention(
            self.attention_heads,
            self.model_dim,
            self.dropout_rate
        )

        self.mha = MultiHeadAttention(
            self.attention_heads,
            self.model_dim,
            self.dropout_rate
        )

        self.pwff = PositionWiseFeedForward(
            in_dim=model_dim,
            mid_dim=pwff_mid_dim,
            out_dim=model_dim
        )

    def forward(self, x, enc_x, tgt_mask, src_mask):
        skip_x = x

        # add mask to skip_x
        x = self.d_mha(x, x, x, tgt_mask)
        x = self.dropout_layer(x)
        x = skip_x + x
        x = self.ln1(x)

        skip_x = x
        x = self.mha(
            x,
            enc_x,
            enc_x,
            src_mask
        )
        x = self.dropout_layer(x)
        x = skip_x + x
        x = self.ln2(x)

        skip_x = x
        x = self.pwff(x)
        x = self.dropout_layer(x)
        x = skip_x + x
        x = self.ln3(x)

        return x


class Decoder(nn.Module):
    def __init__(self, embed_size, padding_idx, decoder_blocks=6, model_dim=512, attention_heads=8,
                 pwff_mid_dim=2048, dropout_rate=0.1, device=None):
        super().__init__()
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        self.embed_layer = ScaledEmbedding(
            embed_size,
            model_dim,
            padding_idx
        )
        self.model_dim = model_dim
        self.attention_heads = attention_heads
        self.pwff_mid_dim = pwff_mid_dim
        self.dropout_rate = dropout_rate

        self.dec_blocks = []
        for _ in range(decoder_blocks):
            self.dec_blocks.append(
                DecoderBlock(
                    model_dim=model_dim,
                    attention_heads=attention_heads,
                    pwff_mid_dim=pwff_mid_dim,
                    dropout_rate=dropout_rate
                ).to(self.device)
            )
        self.out_linear = LinearLayer(model_dim, embed_size)
        self.dec_blocks = nn.ModuleList(self.dec_blocks)
        self.positional_encoder = PositionalEncodings(model_dim, self.device)
        self.dropout_layer = nn.Dropout(p=dropout_rate).to(self.device)

    def forward(self, x, tgt_lens, enc_x, src_mask, tgt_mask):
        """
        :param x: Input tokens to Decoder layer
        :param tgt_lens: Len of every input sentence (for Positionel Encoding's sake)
        :param enc_x: Output from Embedding block
        :param timestep: Which timestep should the decoder output now? (Masking)
        :return:
        """
        x = self.embed_layer(x)
        pos_encodings = self.positional_encoder.get_positional_encodings(tgt_lens)
        pos_encodings = torch.transpose(pad_sequence(pos_encodings, padding_value=0.), 1, 0)

        x = x + pos_encodings
        x = self.dropout_layer(x)

        for dec_block in self.dec_blocks:
            x = dec_block(x, enc_x, tgt_mask, src_mask)

        # now do the linear layer but with embedding weights
        x = self.out_linear(x)

        # now softmax but probably it's better to do it outside the decoder
        return x


class TransformerModel(nn.Module):
    def __init__(self, dec_embed_size, enc_embed_size, padding_idx, enc_dec_blocks=6, model_dim=512, attention_heads=8,
                 pwff_mid_dim=2048, dropout_rate=0.1, device=None):
        super().__init__()
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        self.dec_embed_size = dec_embed_size
        self.enc_embed_size = enc_embed_size
        self.enc_dec_blocks = enc_dec_blocks
        self.model_dim = model_dim
        self.attention_heads = attention_heads
        self.pwff_mid_dim = pwff_mid_dim
        self.dropout_rate = dropout_rate

        self.encoder = Encoder(
            enc_embed_size,
            padding_idx,
            encoder_blocks=enc_dec_blocks,
            model_dim=model_dim,
            attention_heads=attention_heads,
            pwff_mid_dim=pwff_mid_dim,
            dropout_rate=dropout_rate,
            device=self.device
        ).to(self.device)
        self.decoder = Decoder(
            dec_embed_size,
            padding_idx,
            decoder_blocks=enc_dec_blocks,
            model_dim=model_dim,
            attention_heads=attention_heads,
            pwff_mid_dim=pwff_mid_dim,
            dropout_rate=dropout_rate,
            device=self.device
        ).to(self.device)

    def forward(self, x, y, src_lens, tgt_lens, src_mask, tgt_mask):
        tgt_input_lens = [x - 1 for x in tgt_lens]
        enc_x = self.encoder(x, src_lens, src_mask)  # todo: test this
        out = self.decoder(y, tgt_input_lens, enc_x, src_mask, tgt_mask)
        return out

    def forward_gen(self, x, src_lens, src_mask, max_len, bos_idx, temperature):
        enc_x = self.encoder(x, src_lens, src_mask)
        out_tokens = torch.full([x.shape[0], 1], bos_idx, dtype=torch.int64, device=x.device)
        out_lens = [1] * x.shape[0]
        out_probas = None
        for i in range(max_len):
            tgt_mask = torch.ones((1, i + 1, i + 1), dtype=torch.bool).to(self.device)
            dec_probas = self.decoder(out_tokens, out_lens, enc_x, src_mask, tgt_mask)
            if out_probas is None:
                out_probas = dec_probas
            else:
                out_probas = torch.cat([
                    out_probas,
                    dec_probas[:, -1:, :]
                ],
                    dim=1
                )
            _p = torch.softmax(dec_probas / temperature, dim=-1).cpu().detach().numpy()[:, -1, :]
            dec_tokens = np.array(
                [
                    [np.random.choice(np.arange(_p.shape[-1]), p=_p[x])] for x in range(x.shape[0])
                ]
            )
            out_tokens = torch.cat([out_tokens, torch.tensor(dec_tokens, device=x.device)], dim=-1)
            out_lens = [x + 1 for x in out_lens]
        return out_tokens, out_probas
