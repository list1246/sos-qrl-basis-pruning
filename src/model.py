import torch
import torch.nn as nn


class TwoStreamQNet(nn.Module):
    def __init__(self, coeff_dim, mask_dim, base_dim=256, embed_dim=8, dropout=0.0):
        super(TwoStreamQNet, self).__init__()
        dim_layer_1 = base_dim
        dim_layer_2 = base_dim // 2
        dim_fusion = base_dim // 4

        self.coeff_stream = nn.Sequential(
            nn.Linear(coeff_dim, dim_layer_1), nn.LayerNorm(dim_layer_1), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(dim_layer_1, dim_layer_2), nn.ReLU(), nn.Dropout(dropout)
        )
        self.mask_embedding = nn.Embedding(2, embed_dim)
        mask_flat_dim = mask_dim * embed_dim
        self.mask_stream = nn.Sequential(
            nn.Linear(mask_flat_dim, dim_layer_1), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(dim_layer_1, dim_layer_2), nn.ReLU(), nn.Dropout(dropout)
        )
        self.fusion = nn.Sequential(
            nn.Linear(dim_layer_2 * 2, dim_fusion),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_fusion, mask_dim + 1)
        )

    def forward(self, coeffs, masks):
        c_feat = self.coeff_stream(coeffs)
        m_emb = self.mask_embedding(masks)
        m_flat = m_emb.view(m_emb.size(0), -1)
        m_feat = self.mask_stream(m_flat)
        combined = torch.cat([c_feat, m_feat], dim=1)
        q_values = self.fusion(combined)
        return q_values
