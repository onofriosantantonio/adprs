#!/usr/bin/env python3
"""
Script di Training e Fine-Tuning della Rete Neurale Multi-Task (Agente 5).

Legge:   data/processed/train_topics.csv
Addestra: MultiTaskClinicalTransformer (RoBERTa + Tabular Fusion)
Salva:   models/decision_multitask/
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from src.agents.agent5_decision_multitask import AUXILIARY_CLASSES, PRIMARY_CLASSES, MultiTaskDecisionAgent
from src.models.multitask_transformer import MultiTaskClinicalTransformer
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger("Train-MultiTask")

LABEL2ID = {label: i for i, label in enumerate(PRIMARY_CLASSES)}


class ClinicalDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer: AutoTokenizer, max_len: int = 256):
        self.texts = df["clean_text"].fillna("").astype(str).tolist()
        self.labels = [LABEL2ID.get(lbl, 0) for lbl in df["status"]]

        # Costruzione target ausiliari euristici dai dati elaborati
        # 1. Insonnia: basato su topic/sintomi o testo correlato al sonno
        insomnia_labels = []
        # 2. Sostanze: basato sul conteggio entità ontologiche da DAO
        substance_labels = []

        for _, row in df.iterrows():
            txt = str(row.get("clean_text", "")).lower()
            is_insomnia = 1.0 if ("sleep" in txt or "insomnia" in txt or "restless" in txt) else 0.0
            is_substance = 1.0 if float(row.get("entities_detected_count", 0.0)) > 0 else 0.0
            insomnia_labels.append(is_insomnia)
            substance_labels.append(is_substance)

        self.aux_targets = np.column_stack([insomnia_labels, substance_labels]).astype(np.float32)

        # Costruzione feature tabulari (dim 12)
        tab_cols = [
            "sent_neg", "sent_neu", "sent_pos",
            "emo_joy", "emo_sadness", "emo_anger", "emo_fear", "emo_love", "emo_surprise", "emo_gratitude",
            "entities_detected_count", "topic_probability"
        ]
        # Riempi eventuali NaN con 0
        df_tab = df[tab_cols].fillna(0.0).copy()
        df_tab["entities_detected_count"] = df_tab["entities_detected_count"].clip(upper=10.0) / 10.0
        self.tabular_features = df_tab.values.astype(np.float32)

        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "tabular_features": torch.tensor(self.tabular_features[idx], dtype=torch.float32),
            "primary_label": torch.tensor(self.labels[idx], dtype=torch.long),
            "aux_label": torch.tensor(self.aux_targets[idx], dtype=torch.float32),
        }


def train_epoch(model, dataloader, optimizer, scheduler, criterion_primary, criterion_aux, device):
    model.train()
    total_loss = 0.0

    for batch in tqdm(dataloader, desc="Training Batch", leave=False):
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        tab_feat = batch["tabular_features"].to(device)
        primary_labels = batch["primary_label"].to(device)
        aux_labels = batch["aux_label"].to(device)

        outputs = model(input_ids, attention_mask, tab_feat)

        loss_p = criterion_primary(outputs["primary_logits"], primary_labels)
        loss_a = criterion_aux(outputs["auxiliary_logits"], aux_labels)
        loss = loss_p + (0.5 * loss_a)

        loss.backward()
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def eval_model(model, dataloader, criterion_primary, criterion_aux, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_trues = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation Batch", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            tab_feat = batch["tabular_features"].to(device)
            primary_labels = batch["primary_label"].to(device)
            aux_labels = batch["aux_label"].to(device)

            outputs = model(input_ids, attention_mask, tab_feat)

            loss_p = criterion_primary(outputs["primary_logits"], primary_labels)
            loss_a = criterion_aux(outputs["auxiliary_logits"], aux_labels)
            loss = loss_p + (0.5 * loss_a)
            total_loss += loss.item()

            preds = torch.argmax(outputs["primary_logits"], dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_trues.extend(primary_labels.cpu().numpy())

    macro_f1 = f1_score(all_trues, all_preds, average="macro")
    return total_loss / len(dataloader), macro_f1, all_trues, all_preds


def main():
    parser = argparse.ArgumentParser(description="Training Multi-Task Transformer.")
    parser.add_argument("--data", type=str, default="data/processed/train_topics.csv")
    parser.add_argument("--save-dir", type=str, default="models/decision_multitask/")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--sample-size", type=int, default=0, help="Limita righe per debug veloce (0 = full)")
    parser.add_argument("--freeze-backbone", action="store_true", help="Congela il backbone RoBERTa per transfer learning")
    args = parser.parse_args()

    config = load_config()
    device_str = config.get("general", {}).get("device", "cpu")
    if device_str == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif device_str == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    logger.info(f"Avvio training su device: {device}")

    # Carica dati
    df = pd.read_csv(args.data)
    if args.sample_size > 0 and args.sample_size < len(df):
        logger.info(f"Campionamento di {args.sample_size} righe per il training...")
        df = df.sample(n=args.sample_size, random_state=42).reset_index(drop=True)

    logger.info(f"Distribuzione classi:\n{df['status'].value_counts()}")

    # Split train/val stratificato
    train_df, val_df = train_test_split(df, test_size=0.15, stratify=df["status"], random_state=42)

    backbone_name = config.get("agent6_decision_multitask", {}).get("backbone_model", "FacebookAI/roberta-base")
    tokenizer = AutoTokenizer.from_pretrained(backbone_name)

    train_dataset = ClinicalDataset(train_df, tokenizer)
    val_dataset = ClinicalDataset(val_df, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Inizializza modello
    model = MultiTaskClinicalTransformer(
        backbone_model_name=backbone_name,
        num_primary_classes=len(PRIMARY_CLASSES),
        num_auxiliary_classes=len(AUXILIARY_CLASSES),
        tabular_feature_dim=12,
        freeze_backbone=args.freeze_backbone,
    ).to(device)

    # Class weights per gestire lo sbilanciamento
    class_counts = train_df["status"].value_counts()
    weights = [len(train_df) / (len(PRIMARY_CLASSES) * class_counts[cls]) for cls in PRIMARY_CLASSES]
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)

    criterion_primary = nn.CrossEntropyLoss(weight=class_weights)
    criterion_aux = nn.BCEWithLogitsLoss()

    if args.freeze_backbone:
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        backbone_params = sum(p.numel() for p in model.transformer.parameters())
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Backbone congelato: {backbone_params:,}/{total_params:,} parametri frozen")
        logger.info(f"Parametri addestrabili: {sum(p.numel() for p in trainable_params):,}")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=0.01,
    )
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)

    best_f1 = 0.0
    save_path = Path(args.save_dir)

    for epoch in range(1, args.epochs + 1):
        logger.info(f"\n--- Epoca {epoch}/{args.epochs} ---")
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, criterion_primary, criterion_aux, device)
        val_loss, val_f1, trues, preds = eval_model(model, val_loader, criterion_primary, criterion_aux, device)

        logger.info(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1-Macro: {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            logger.info(f"Nuovo miglior F1-Macro: {best_f1:.4f}! Salvataggio modello in {save_path}...")
            agent = MultiTaskDecisionAgent(config=config, model=model, tokenizer=tokenizer)
            agent.save_pretrained(save_path)

    logger.info(f"\nAddestramento completato! Miglior F1-Macro ottenuto: {best_f1:.4f}")


if __name__ == "__main__":
    main()
