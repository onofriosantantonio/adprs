#!/usr/bin/env python3
"""
Script per eseguire l'Agente 2 (Ontological NER & Entity Masking)
sui dati pre-elaborati dall'Agente 1.
Salva:
1. Il modello dell'Agente 2 in 'models/ner_ontology/'.
2. I dataset con testo mascherato in 'data/processed/train_masked.csv'.
"""

import argparse
import sys
from pathlib import Path

# Assicura che la directory radice sia in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from tqdm import tqdm

from src.agents.agent2_ner_ontology import OntologicalNERAgent
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger("Agent2-Runner")


def run_agent2(input_csv: str = "data/processed/train_preprocessed.csv", save_model_dir: str = "models/ner_ontology/"):
    input_path = Path(input_csv)
    if not input_path.exists():
        logger.error(f"File di input non trovato in: {input_path}")
        logger.info("Assicurati di aver eseguito prima l'Agente 1: python scripts/prepare_dataset.py")
        sys.exit(1)

    logger.info(f"Caricamento dati pre-elaborati da: {input_path}...")
    df = pd.read_csv(input_path)

    config = load_config()
    logger.info("Inizializzazione Agente 2 (OntologicalNERAgent)...")
    agent2 = OntologicalNERAgent(config=config)

    logger.info("Esecuzione Named Entity Recognition ed Entity Masking sui testi...")
    masked_texts = []
    total_entities_counts = []

    text_col = "clean_text" if "clean_text" in df.columns else "text"

    for text in tqdm(df[text_col], desc="Agent 2 - Entity Masking"):
        masked, entities = agent2.mask_and_extract_entities(str(text))
        masked_texts.append(masked)
        total_entities_counts.append(len(entities))

    df["masked_text"] = masked_texts
    df["entities_detected_count"] = total_entities_counts

    # Statistiche delle entità trovate
    total_detected = sum(total_entities_counts)
    logger.info(f"Completato! Rilevate {total_detected} entità clinico-sostanziali totali.")

    # Salvataggio dataset con testo mascherato
    output_csv = input_path.parent / (input_path.stem.replace("_preprocessed", "") + "_masked.csv")
    df.to_csv(output_csv, index=False)
    logger.info(f"Dataset mascherato salvato in: {output_csv}")

    # Salvataggio del modello dell'Agente 2
    if save_model_dir:
        model_dir = Path(save_model_dir)
        logger.info(f"Salvataggio del modello Agente 2 in: {model_dir}...")
        agent2.save_pretrained(model_dir)
        logger.info(f"Modello salvato con successo in {model_dir}!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Esecuzione Agente 2 (NER Ontologico & Entity Masking).")
    parser.add_argument(
        "--input",
        type=str,
        default="data/processed/train_preprocessed.csv",
        help="Percorso al CSV pre-elaborato da Agente 1 (default: data/processed/train_preprocessed.csv)."
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="models/ner_ontology/",
        help="Directory dove salvare gli artefatti del modello Agente 2 (default: models/ner_ontology/)."
    )
    args = parser.parse_args()

    run_agent2(input_csv=args.input, save_model_dir=args.save_dir)
