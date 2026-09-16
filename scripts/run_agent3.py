#!/usr/bin/env python3
"""
Script per eseguire l'Agente 3 (Sentiment & Emotion Analysis)
sui dati elaborati dall'Agente 2.

Legge:   data/processed/train_masked.csv
Scrive:  data/processed/train_emotions.csv
Salva:   models/sentiment_emotion/

Le colonne aggiunte al CSV di output:
  - sentiment_label        : negative / neutral / positive
  - sentiment_score        : confidenza (float)
  - sent_neg / sent_neu / sent_pos : vettore sentiment (3 float)
  - emo_joy, emo_sadness, emo_anger, emo_fear,
    emo_love, emo_surprise, emo_gratitude : score emozioni (7 float)
"""

import argparse
import ast
import sys
from pathlib import Path

# Assicura che la directory radice sia in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from tqdm import tqdm

from src.agents.agent3_sentiment_emotion import SentimentEmotionAgent
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger("Agent3-Runner")


def run_agent3(
    input_csv: str = "data/processed/train_masked.csv",
    save_model_dir: str = "models/sentiment_emotion/",
    batch_size: int = 64,
) -> None:
    """
    Esegue l'Agente 3 su tutto il dataset e salva i risultati.

    Args:
        input_csv     : percorso al CSV prodotto dall'Agente 2
        save_model_dir: directory dove salvare la configurazione dell'agente
        batch_size    : numero di testi per batch (ottimizza la memoria)
    """
    input_path = Path(input_csv)
    if not input_path.exists():
        logger.error(f"File di input non trovato: {input_path}")
        logger.info(
            "Assicurati di aver eseguito prima l'Agente 2: "
            "python scripts/run_agent2.py"
        )
        sys.exit(1)

    logger.info(f"Caricamento dati da: {input_path}")
    df = pd.read_csv(input_path)
    logger.info(f"Righe caricate: {len(df)}")

    # Sceglie la colonna testo: preferisce clean_text, altrimenti text
    text_col = "clean_text" if "clean_text" in df.columns else "text"
    logger.info(f"Colonna testo utilizzata: '{text_col}'")

    config = load_config()
    logger.info("Inizializzazione Agente 3 (SentimentEmotionAgent)...")
    agent3 = SentimentEmotionAgent(config=config)

    # --- Inference -------------------------------------------------------
    sentiment_labels = []
    sentiment_scores = []
    sent_neg_list, sent_neu_list, sent_pos_list = [], [], []
    emotion_cols = {f"emo_{e}": [] for e in agent3.target_emotions}

    texts = df[text_col].fillna("").tolist()
    n = len(texts)
    logger.info(
        f"Avvio inference su {n} testi "
        f"(batch_size={batch_size}, modelli: sentiment + emotion)..."
    )

    # Carica i modelli una sola volta prima del loop
    agent3._load_models()

    for i in tqdm(range(0, n, batch_size), desc="Agent 3 - Sentiment & Emotion"):
        batch = texts[i : i + batch_size]

        for text in batch:
            text_str = str(text) if text else ""

            label, score, s_vec = agent3.analyze_sentiment(text_str)
            e_scores, _ = agent3.analyze_emotions(text_str)

            sentiment_labels.append(label)
            sentiment_scores.append(round(score, 6))
            sent_neg_list.append(round(s_vec[0], 6))
            sent_neu_list.append(round(s_vec[1], 6))
            sent_pos_list.append(round(s_vec[2], 6))

            for emo in agent3.target_emotions:
                emotion_cols[f"emo_{emo}"].append(round(e_scores[emo], 6))

    # --- Aggiunta colonne al dataframe -----------------------------------
    df["sentiment_label"] = sentiment_labels
    df["sentiment_score"] = sentiment_scores
    df["sent_neg"] = sent_neg_list
    df["sent_neu"] = sent_neu_list
    df["sent_pos"] = sent_pos_list

    for col_name, values in emotion_cols.items():
        df[col_name] = values

    # --- Statistiche sommarie --------------------------------------------
    logger.info("=== Distribuzione Sentiment ===")
    for lbl, cnt in df["sentiment_label"].value_counts().items():
        logger.info(f"  {lbl}: {cnt} ({cnt/len(df)*100:.1f}%)")

    logger.info("=== Media Score Emozioni ===")
    for emo in agent3.target_emotions:
        col = f"emo_{emo}"
        logger.info(f"  {emo}: {df[col].mean():.4f}")

    # --- Salvataggio CSV -------------------------------------------------
    output_path = input_path.parent / (
        input_path.stem.replace("_masked", "") + "_emotions.csv"
    )
    df.to_csv(output_path, index=False)
    logger.info(f"Dataset arricchito salvato in: {output_path}")

    # --- Salvataggio modello ---------------------------------------------
    if save_model_dir:
        model_dir = Path(save_model_dir)
        logger.info(f"Salvataggio configurazione Agente 3 in: {model_dir}")
        agent3.save_pretrained(model_dir)
        logger.info(f"Configurazione salvata in: {model_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Esecuzione Agente 3 (Sentiment & Emotion Analysis)."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/processed/train_masked.csv",
        help=(
            "Percorso al CSV prodotto dall'Agente 2 "
            "(default: data/processed/train_masked.csv)."
        ),
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="models/sentiment_emotion/",
        help=(
            "Directory dove salvare la configurazione dell'Agente 3 "
            "(default: models/sentiment_emotion/)."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Numero di testi per batch durante l'inference (default: 64).",
    )

    args = parser.parse_args()
    run_agent3(
        input_csv=args.input,
        save_model_dir=args.save_dir,
        batch_size=args.batch_size,
    )
