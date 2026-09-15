"""
Architettura Neurale Multi-Task Learning (MTL) per l'Analisi Clinica e del Rischio.

Combina:
  - Backbone Transformer (RoBERTa / ClinicalBERT) per estrarre rappresentazioni testuali dense (dim 768)
  - Multi-Modal Feature Fusion con i vettori numerici estratti dagli agenti a monte:
      * Sentiment ed Emozioni (dim 10) dall'Agente 3
      * Topic probability e Topic features dall'Agente 4
      * Segnali ontologici dall'Agente 2
  - Heads di Classificazione Multi-Task:
      * Primary Head (Mental Health Status): 4 classi (Normal, Depression, Anxiety, Suicidal)
      * Auxiliary Head (Risk Conditions): Multi-label (es. Insomnia Risk, Substance/SUD Risk)
"""

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel


class MultiTaskClinicalTransformer(nn.Module):
    """
    Rete Neurale Multi-Task con Feature Fusion:
    Combina l'output del Transformer Backbone con feature numeriche tabulari.
    """

    def __init__(
        self,
        backbone_model_name: str = "FacebookAI/roberta-base",
        num_primary_classes: int = 4,
        num_auxiliary_classes: int = 2,
        tabular_feature_dim: int = 12,  # 3 sentiment + 7 emotions + 1 entities count + 1 topic prob
        fusion_hidden_dim: int = 256,
        dropout_rate: float = 0.2,
        freeze_backbone: bool = True,
    ):
        super().__init__()

        self.backbone_model_name = backbone_model_name
        self.num_primary_classes = num_primary_classes
        self.num_auxiliary_classes = num_auxiliary_classes
        self.tabular_feature_dim = tabular_feature_dim
        self.freeze_backbone = freeze_backbone

        # 1. Transformer Backbone
        self.config = AutoConfig.from_pretrained(backbone_model_name)
        self.config.attention_probs_dropout_prob = 0.0
        self.config.hidden_dropout_prob = 0.0
        self.transformer = AutoModel.from_pretrained(backbone_model_name, config=self.config)
        hidden_size = self.config.hidden_size  # tipicamente 768 per roberta-base

        # Freeze backbone weights per transfer learning
        if freeze_backbone:
            for param in self.transformer.parameters():
                param.requires_grad = False

        # 2. MLP Proiezione Feature Tabulari
        self.tabular_mlp = nn.Sequential(
            nn.Linear(tabular_feature_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        # 3. Layer di Feature Fusion (Testo + Tabulare)
        total_features_dim = hidden_size + 64
        self.fusion_layer = nn.Sequential(
            nn.Linear(total_features_dim, fusion_hidden_dim),
            nn.LayerNorm(fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        # 4. Heads di Classificazione
        # Head Primaria: 4 Classi di Salute Mentale (Normal, Depression, Anxiety, Suicidal)
        self.primary_head = nn.Sequential(
            nn.Linear(fusion_hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_primary_classes),
        )

        # Head Ausiliaria: Rischio Comportamentale / Clinico (Insonnia, Sostanze) - Multi-Label Sigmoid
        self.auxiliary_head = nn.Sequential(
            nn.Linear(fusion_hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_auxiliary_classes),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        tabular_features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass della rete.

        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            tabular_features: [batch_size, tabular_feature_dim]

        Returns:
            Dict con 'primary_logits' e 'auxiliary_logits'
        """
        # Transformer forward
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)

        # Estrazione della rappresentazione CLS / Mean pooling
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            text_features = outputs.pooler_output
        else:
            # Mean pooling ponderato su attention_mask per modelli senza pooler
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            text_features = sum_embeddings / sum_mask

        # Proiezione feature tabulari
        tab_emb = self.tabular_mlp(tabular_features)

        # Concatenazione e Fusion
        combined = torch.cat([text_features, tab_emb], dim=1)
        fused = self.fusion_layer(combined)

        # Calcolo Logits delle heads
        primary_logits = self.primary_head(fused)
        auxiliary_logits = self.auxiliary_head(fused)

        return {
            "primary_logits": primary_logits,
            "auxiliary_logits": auxiliary_logits,
            "fused_features": fused,
        }
