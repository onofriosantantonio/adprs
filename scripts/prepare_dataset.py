#!/usr/bin/env python3
"""
Script per scaricare, validare e pre-elaborare il dataset
'ourafla/Mental-Health_Text-Classification_Dataset' tramite l'Agente 1.
Salva:
1. Il modello di preprocessing in 'models/preprocessor/'.
2. Il dataset pre-elaborato in 'data/processed/'.
"""

import argparse
import sys
from pathlib import Path

# Assicura che la root del progetto sia in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

from src.agents.agent1_preprocessor import PreprocessingAgent
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger("DatasetPreparation")

def prepare_dataset(sample_size: int = None, save_model_dir: str = "models/preprocessor/"):
    config = load_config()
    ds_cfg = config.get("dataset", {})
    hf_repo = ds_cfg.get("hf_repo", "ourafla/Mental-Health_Text-Classification_Dataset")
    train_file = ds_cfg.get("train_file", "mental_heath_unbanlanced.csv")
    test_file = ds_cfg.get("test_file", "mental_health_combined_test.csv")
    text_col = ds_cfg.get("text_column", "text")
    label_col = ds_cfg.get("label_column", "status")

    logger.info(f"Scaricamento dataset da Hugging Face: {hf_repo}...")
    dataset = load_dataset(
        hf_repo,
        data_files={"train": train_file, "test": test_file}
    )

    train_data = dataset["train"]
    test_data = dataset["test"]

    logger.info(f"Dataset caricato: Train={len(train_data)} campioni, Test={len(test_data)} campioni.")

    if sample_size and sample_size > 0:
        logger.info(f"Estrazione subset rapido di {sample_size} campioni per il test...")
        train_df = pd.DataFrame(train_data.select(range(min(sample_size, len(train_data)))))
        test_df = pd.DataFrame(test_data.select(range(min(max(100, sample_size // 5), len(test_data)))))
    else:
        train_df = pd.DataFrame(train_data)
        test_df = pd.DataFrame(test_data)

    logger.info("Inizializzazione Agente 1 (PreprocessingAgent)...")
    preprocessor = PreprocessingAgent(config=config)

    # Elaborazione del set di addestramento
    logger.info("Elaborazione e normalizzazione testi di Train...")
    clean_train_texts = []
    lemmatized_train_texts = []

    for text in tqdm(train_df[text_col], desc="Agent 1 - Preprocessing Train"):
        clean = preprocessor.clean_text(text)
        lemmas = preprocessor.lemmatize_and_filter(clean)["lemmatized_text"]
        clean_train_texts.append(clean)
        lemmatized_train_texts.append(lemmas)

    train_df["clean_text"] = clean_train_texts
    train_df["lemmatized_text"] = lemmatized_train_texts

    # Elaborazione del set di test
    logger.info("Elaborazione e normalizzazione testi di Test...")
    clean_test_texts = []
    lemmatized_test_texts = []

    for text in tqdm(test_df[text_col], desc="Agent 1 - Preprocessing Test"):
        clean = preprocessor.clean_text(text)
        lemmas = preprocessor.lemmatize_and_filter(clean)["lemmatized_text"]
        clean_test_texts.append(clean)
        lemmatized_test_texts.append(lemmas)

    test_df["clean_text"] = clean_test_texts
    test_df["lemmatized_text"] = lemmatized_test_texts

    # Salvataggio dati processati
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    train_out = processed_dir / "train_preprocessed.csv"
    test_out = processed_dir / "test_preprocessed.csv"

    train_df.to_csv(train_out, index=False)
    test_df.to_csv(test_out, index=False)
    logger.info(f"Dataset pre-elaborati salvati con successo in:\n - {train_out}\n - {test_out}")

    # Salvataggio artefatti del modello di preprocessing
    if save_model_dir:
        model_dir = Path(save_model_dir)
        logger.info(f"Salvataggio del modello di preprocessing in: {model_dir}...")
        preprocessor.save_pretrained(model_dir)

    logger.info("Pipeline di preparazione completata con successo!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download e preprocessing del dataset con Agente 1.")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=500,
        help="Numero di campioni da processare per un test rapido (default: 500, usa 0 per l'intero dataset)."
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="models/preprocessor/",
        help="Directory dove salvare il modello di preprocessing."
    )
    args = parser.parse_args()

    sample = None if args.sample_size <= 0 else args.sample_size
    prepare_dataset(sample_size=sample, save_model_dir=args.save_dir)
