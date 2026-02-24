import torch
import torch.nn as nn
import torch.nn.functional as F
from models.Mamba_EncDec import Encoder, EncoderLayer
from models.Embed import DataEmbedding_inverted

from mamba_ssm import Mamba

from Attention.SKAttention import SKAttention

class DS_Mamba(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs):
        super(DS_Mamba, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        # self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.dropout)
        # self.class_strategy = configs.class_strategy
        # Encoder-only architecture
        self.fe = PPM(configs)
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=configs.c_out, out_channels=configs.d_model, kernel_size=3, padding='same'),
            # output: (batch_size, d_model, seq_len-1)
            nn.ReLU(),
            # nn.MaxPool1d(kernel_size=3, padding=1),  # output: (batch_size, d_model, (seq_len-1)-1)
        )

        self.encoder = Encoder(
            [
                EncoderLayer(
                    Mamba(
                        d_model=configs.d_model,  # Model dimension d_model
                        d_state=configs.d_state,  # SSM state expansion factor
                        d_conv=2,  # Local convolution width
                        expand=1,  # Block expansion factor)
                    ),
                    Mamba(
                        d_model=configs.d_model,  # Model dimension d_model
                        d_state=configs.d_state,  # SSM state expansion factor
                        d_conv=2,  # Local convolution width
                        expand=1,  # Block expansion factor)
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)
        SOH_output_size = configs.d_model * configs.seq_len
        self.mlp = MLP(in_channel=SOH_output_size, out_channel=configs.c_out, configs=configs)

    def forecast(self, x_enc):
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape  # B L N

        # enc_out = self.enc_embedding(x_enc)  # covariates (e.g timestamp) can be also embedded as tokens

        enc_out = x_enc.permute(0, 2, 1)
        enc_out = self.fe(enc_out)
        # enc_out = self.conv1(enc_out)
        enc_out = enc_out.permute(0, 2, 1)

        # B N E -> B N E                (B L E -> B L E in the vanilla Transformer)
        # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules

        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # B N E -> B N S -> B S N
        # enc_out = enc_out.contiguous().view(enc_out.size(0), -1)
        # dec_out = self.mlp(enc_out)
        # dec_out = dec_out.view(dec_out.shape[0], 1, -1)
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N]  # filter the covariates

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out

    def forward(self, x_enc):
        dec_out = self.forecast(x_enc)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]

class MLP(nn.Module):
    def __init__(self, in_channel, out_channel, configs):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_channel, configs.d_mlp)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(configs.dropout)
        self.fc2 = nn.Linear(configs.d_mlp, out_channel)
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class PPM(nn.Module):  # 1D version of PPM for time series
    def __init__(self, configs):
        super(PPM, self).__init__()
        self.down_conv = nn.Sequential(
            nn.Conv1d(1, configs.d_model, 1, padding='same'),  # 输入通道改为1，输出为down_dim
            # nn.BatchNorm1d(configs.d_model),
            nn.ReLU(),
        )
        self.average_pool = nn.AvgPool1d(kernel_size=2, stride=2)
        # 修改为1D池化和卷积
        self.conv1 = nn.Sequential(
            nn.Conv1d(configs.d_model//2, configs.d_model//2, kernel_size=3, padding='same'),
            # nn.BatchNorm1d(configs.d_model//2),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(configs.d_model//2, configs.d_model//2, kernel_size=5, padding='same'),
            # nn.BatchNorm1d(configs.d_model//2),
            nn.ReLU()
        )

        self.fuse = nn.Sequential(
            nn.Conv1d(2 * configs.d_model, configs.d_model, kernel_size=3,padding='same'),
            # nn.BatchNorm1d(configs.d_model),
            nn.ReLU()
        )

    def forward(self, x):
        # x shape: [batch_size, 1, seq_len]
        x = self.down_conv(x)

        x_out = x.permute(0, 2, 1)
        x_out = self.average_pool(x_out)
        x_out = x_out.permute(0, 2, 1)

        conv1_up = self.conv1(x_out)
        conv2_up = self.conv2(x_out)

        conv_all = torch.cat([x , conv1_up, conv2_up], dim=1)

        return self.fuse(conv_all)