import torch
import torch.nn as nn
import torch.nn.functional as F
from model.modules import FCBlock, MixerLayer,BasicBlock

class CompositionalEncoder(nn.Module):
    def __init__(self,k,d,h,m,dropout_ratio=0.0):
        super().__init__()
        self.linear1 = nn.Linear(d,h)
        self.mixer = MixerLayer(hidden_dim=h,hidden_inter_dim=h,token_inter_dim=k,token_dim=k,dropout_ratio=dropout_ratio)
        self.linear2 = nn.Linear(k,m)
    def forward(self,x):
        x = self.linear1(x)
        x = self.mixer(x)
        x = x.transpose(1,2)
        x = self.linear2(x)
        x = x.transpose(1,2)
        return x

# utilizes MSE
class VectorQuantizer(nn.Module):
    def __init__(self, v, h, commitment_cost):
        super().__init__()
        self.num_codes = v
        self.code_dim = h
        self.commitment_cost = commitment_cost
        # randomly initialized codebook (v x h) which can be learned
        self.codebook = nn.Parameter(torch.randn(v, h))

    def forward(self, x):
        # x: (batch_size, M tokens, code_dim)
        b, m, h = x.shape
        flat_x = x.reshape(-1, h)  # (b*m, h)

        # compute distances
        distances = (
            flat_x.pow(2).sum(1, keepdim=True)
            - 2 * flat_x @ self.codebook.t()
            + self.codebook.pow(2).sum(1)
        )  # (b*m, v)

        encoding_indices = torch.argmin(distances, dim=1)  # (b*m)

        quantized = self.codebook[encoding_indices]  # (b*m, h)

        # compute loss
        commitment_loss = F.mse_loss(quantized.detach(), flat_x)
        codebook_loss = F.mse_loss(quantized, flat_x.detach())
        loss = self.commitment_cost * commitment_loss + codebook_loss

        return quantized.view(b, m, h), loss, encoding_indices
 
class Decoder(nn.Module):
    def __init__(self,k,d,h,m,dropout_ratio=0.0):
        super().__init__()
        self.k = k
        self.d = d
        self.mixer  = MixerLayer(
            hidden_dim=h,
            hidden_inter_dim=h,
            token_inter_dim=m,
            token_dim=m,
            dropout_ratio=dropout_ratio
        )
        self.linear = nn.Linear(m*h,k*d)
    def forward(self,x):
        # x: b x m x h
        x = self.mixer(x) # b x m x h
        x = x.flatten(1) # b x mh
        x = self.linear(x) # b x kd
        x = x.view(x.shape[0],self.k,self.d) # b x k x d
        return x

class ClassificationHead(nn.Module):
    def __init__(self, in_channels, num_tokens, token_dim, codebook_size):
        """
        Args:
            in_channels: C from backbone features (e.g., 1024 from Swin-V2)
            num_tokens: M, number of compositional tokens (e.g., 16)
            token_dim: N, embedding dimension per token (e.g., 256)
            codebook_size: V, number of discrete codebook entries (e.g., 512)
        """
        super().__init__()
        self.modulator = nn.Sequential(
            BasicBlock(inplanes=in_channels, planes=in_channels),
            BasicBlock(inplanes=in_channels, planes=in_channels),
        )

        self.flatten_proj = nn.Linear(in_channels * 8 * 8, num_tokens * token_dim)

        # MLP Mixer Blocks
        self.mixers = nn.Sequential(*[
            MixerLayer(
                hidden_dim=token_dim,
                hidden_inter_dim=token_dim * 2,
                token_dim=num_tokens,
                token_inter_dim=num_tokens * 2,
                dropout_ratio=0.0
            ) for i in range(4)
        ])

        # Final classification layer: M tokens × V classes
        self.classifier = nn.Linear(token_dim, codebook_size)

        self.num_tokens = num_tokens
        self.token_dim = token_dim

    def forward(self, x):
        # x shape: [B, 8, 8, C] => permute to [B, C, 8, 8]
        x = x.permute(0, 3, 1, 2)  # [B, C, H, W]
        x = self.modulator(x)

        # Flatten and project: [B, C, H, W] -> [B, M * token_dim]
        x = x.reshape(x.size(0), -1)
        x = self.flatten_proj(x)

        # Reshape to [B, M, N]
        x = x.view(x.size(0), self.num_tokens, self.token_dim)

        # Apply 4 Mixer Blocks
        x = self.mixers(x)

        # Final logits: [B, M, V]
        logits = self.classifier(x)
        return logits