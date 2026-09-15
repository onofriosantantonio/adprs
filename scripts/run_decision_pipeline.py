#!/usr/bin/env python3
"""
Pipeline Decisionale Finale (Agente 5).

Carica il modello salvato in models/decision_multitask/
ed esegue l'inferenza completa generando:
- predicted_condition (Normal, Depression, Anxiety, Suicidal)
- primary_confidence
- p_insomnia_risk e p_substance_risk
- sleep_dep_score
- clinical_risk_level (NORMAL, WARNING, CRITICAL)
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from tqdm import tqdm

from src.agents.agent5_decision_multitask import MultiTaskDecisionAgent
from src.utils.logger import get_logger

logger = get_logger("DecisionPipeline")


def run_pipeline(
    input_csv: str = "data/processed/train_topics.csv",
    model_dir: str = "models/decision_multitask/",
    output_csv: str = "data/processed/final_predictions.csv",
):
    input_path = Path(input_csv)
    if not input_path.exists():
        logger.error(f"File di input non trovato: {input_path}")
        sys.exit(1)

    logger.info(f"Caricamento Agente Decisionale da: {model_dir}...")
    agent = MultiTaskDecisionAgent.from_pretrained(model_dir)

    logger.info(f"Caricamento dati da: {input_path}...")
    df = pd.read_csv(input_path)
    logger.info(f"Totale record da valutare: {len(df)}")

    predicted_conditions = []
    primary_confidences = []
    sleep_dep_scores = []
    risk_levels = []
    p_insomnia_list = []
    p_substance_list = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Inference Agente 5"):
        record_dict = row.to_dict()
        pred = agent.predict(record_dict)

        predicted_conditions.append(pred["predicted_condition"])
        primary_confidences.append(pred["primary_confidence"])
        sleep_dep_scores.append(pred["sleep_dep_score"])
        risk_levels.append(pred["clinical_risk_level"])
        p_insomnia_list.append(pred["auxiliary_probabilities"]["p_insomnia_risk"])
        p_substance_list.append(pred["auxiliary_probabilities"]["p_substance_risk"])

    df["predicted_condition"] = predicted_conditions
    df["primary_confidence"] = primary_confidences
    df["p_insomnia_risk"] = p_insomnia_list
    df["p_substance_risk"] = p_substance_list
    df["sleep_dep_score"] = sleep_dep_scores
    df["clinical_risk_level"] = risk_levels

    logger.info("\n=== Distribuzione Condizioni Predette ===")
    for k, v in df["predicted_condition"].value_counts().items():
        logger.info(f"  {k}: {v} ({v/len(df)*100:.1f}%)")

    logger.info("\n=== Distribuzione Livelli di Rischio Clinico (SleepDepScore) ===")
    for k, v in df["clinical_risk_level"].value_counts().items():
        logger.info(f"  {k}: {v} ({v/len(df)*100:.1f}%)")

    df.to_csv(output_csv, index=False)
    logger.info(f"\nPredizioni finali salvate con successo in: {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inferenza Agente Decisionale Multi-Task.")
    parser.add_argument("--input", type=str, default="data/processed/train_topics.csv")
    parser.add_argument("--model-dir", type=str, default="models/decision_multitask/")
    parser.add_argument("--output", type=str, default="data/processed/final_predictions.csv")
    args = parser.parse_args()

    run_pipeline(input_csv=args.input, model_dir=args.model_dir, output_csv=args.output)
