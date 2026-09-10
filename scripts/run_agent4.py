#!/usr/bin/env python3
"""
Script per eseguire l'Agente 4 (Clinical Topic Modeling con BERTopic)
sui dati arricchiti dall'Agente 3.

Flusso:
1. Legge data/processed/train_emotions.csv (o train_masked.csv come fallback)
2. Addestra la pipeline BERTopic su `clean_text`
3. Assegna topic_id, topic_probability e topic_keywords a ciascun record
4. Salva il dataset arricchito in data/processed/train_topics.csv
5. Salva il modello BERTopic addestrato e le info topic in models/topic_modeling/
"""

import argparse
import sys
from pathlib import Path

# Assicura che la radice del progetto sia in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from tqdm import tqdm

from src.agents.agent4_topic_modeling import ClinicalTopicModelingAgent
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger("Agent4-Runner")


def run_agent4(
    input_csv: str = "data/processed/train_emotions.csv",
    save_model_dir: str = "models/topic_modeling/",
    sample_size: int = 0,
) -> None:
    input_path = Path(input_csv)
    if not input_path.exists():
        fallback_path = Path("data/processed/train_masked.csv")
        if fallback_path.exists():
            logger.warning(f"File {input_path} non trovato. Uso il fallback: {fallback_path}")
            input_path = fallback_path
        else:
            logger.error(f"Nessun file di input trovato in {input_path} o {fallback_path}")
            sys.exit(1)

    logger.info(f"Caricamento dati da: {input_path}...")
    df = pd.read_csv(input_path)
    logger.info(f"Record totali: {len(df)}")

    text_col = "clean_text" if "clean_text" in df.columns else "text"
    texts = df[text_col].fillna("").astype(str).tolist()

    config = load_config()
    logger.info("Inizializzazione Agente 4 (ClinicalTopicModelingAgent)...")
    agent4 = ClinicalTopicModelingAgent(config=config)

    # Addestramento BERTopic
    if sample_size > 0 and sample_size < len(texts):
        logger.info(f"Addestramento BERTopic su un campione di {sample_size} testi...")
        train_sample = df[text_col].dropna().sample(n=sample_size, random_state=42).tolist()
        agent4.fit(train_sample)
    else:
        logger.info("Addestramento BERTopic sull'intero dataset...")
        agent4.fit(texts)

    # Inferenza sui testi per associare i topic
    logger.info("Calcolo topic e probabilità per tutti i record...")
    topics, probs = agent4.transform(texts)

    topic_ids = [int(t) for t in topics]
    topic_probs = []
    topic_keywords_list = []

    for i, t_id in enumerate(tqdm(topic_ids, desc="Agent 4 - Topic Extraction")):
        if probs is not None and len(probs) > 0:
            if hasattr(probs[i], "__iter__"):
                prob_val = float(max(probs[i]))
            else:
                prob_val = float(probs[i])
        else:
            prob_val = 0.0

        topic_probs.append(round(prob_val, 6))
        kw = agent4.get_topic_keywords(t_id, top_n=5)
        topic_keywords_list.append(", ".join(kw) if kw else "outlier")

    df["topic_id"] = topic_ids
    df["topic_probability"] = topic_probs
    df["topic_keywords"] = topic_keywords_list

    # Distribuzione dei topic estratti
    unique_topics = set(topic_ids)
    num_topics_found = len(unique_topics) - (1 if -1 in unique_topics else 0)
    outliers = sum(1 for t in topic_ids if t == -1)
    logger.info(f"Topic unici identificati: {num_topics_found} (Outliers / rumore: {outliers})")

    # Salvataggio dataset con topic
    output_path = input_path.parent / (input_path.stem.replace("_emotions", "").replace("_masked", "") + "_topics.csv")
    df.to_csv(output_path, index=False)
    logger.info(f"Dataset con feature tematiche salvato in: {output_path}")

    # Salvataggio modello
    if save_model_dir:
        model_dir = Path(save_model_dir)
        logger.info(f"Salvataggio modello Agente 4 in: {model_dir}...")
        agent4.save_pretrained(model_dir)
        logger.info(f"Modello Agente 4 salvato con successo in: {model_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Esecuzione Agente 4 (Topic Modeling Clinico).")
    parser.add_argument(
        "--input",
        type=str,
        default="data/processed/train_emotions.csv",
        help="File di input prodotto dall'Agente 3 (default: data/processed/train_emotions.csv).",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="models/topic_modeling/",
        help="Directory dove salvare il modello Agente 4 (default: models/topic_modeling/).",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=0,
        help="Dimensione opzionale del campione per il fitting veloce (default: 0 = tutto il dataset).",
    )
    args = parser.parse_args()

    run_agent4(
        input_csv=args.input,
        save_model_dir=args.save_dir,
        sample_size=args.sample_size,
    )
