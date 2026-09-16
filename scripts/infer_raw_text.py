#!/usr/bin/env python3
"""
Script di Inferenza End-to-End su testo crudo.

Questo script concatena tutti i 5 agenti del sistema MAS per analizzare
un singolo testo in ingresso, partendo dal testo grezzo fino al punteggio
di rischio finale (SleepDepScore) e alla classificazione multi-task.
"""

import argparse
import json
import sys
from pathlib import Path

# Assicura che la root del progetto sia nel PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.agent1_preprocessor import PreprocessingAgent
from src.agents.agent2_ner_ontology import OntologicalNERAgent
from src.agents.agent3_sentiment_emotion import SentimentEmotionAgent
from src.agents.agent4_topic_modeling import ClinicalTopicModelingAgent
from src.agents.agent5_decision_multitask import MultiTaskDecisionAgent
from src.utils.logger import get_logger

logger = get_logger("EndToEndInference")


def run_inference(text: str, model_dir: str = "models/decision_multitask/"):
    logger.info("Inizializzazione della Pipeline Multiagente in corso...")
    
    # 1. Inizializzazione Agenti
    logger.info("  - Caricamento Agente 1 (Preprocessing)...")
    agent1 = PreprocessingAgent()
    
    logger.info("  - Caricamento Agente 2 (NER & Ontologia)...")
    agent2 = OntologicalNERAgent()
    
    logger.info("  - Caricamento Agente 3 (Sentiment & Emozioni)...")
    agent3 = SentimentEmotionAgent()
    
    logger.info("  - Caricamento Agente 4 (Topic Modeling)...")
    try:
        agent4 = ClinicalTopicModelingAgent.from_pretrained("models/topic_modeling")
    except Exception as e:
        logger.error(f"Impossibile caricare il modello BERTopic (Agente 4). Assicurati che sia stato addestrato. Errore: {e}")
        sys.exit(1)
    
    # Per l'Agente 5, carichiamo i pesi precedentemente addestrati
    logger.info(f"  - Caricamento Agente 5 (Decision Multi-Task) da {model_dir}...")
    try:
        agent5 = MultiTaskDecisionAgent.from_pretrained(model_dir)
    except Exception as e:
        logger.error(f"Impossibile caricare il modello dell'Agente 5. Hai eseguito il training? Errore: {e}")
        sys.exit(1)

    logger.info("\n=== INIZIO ELABORAZIONE PIPELINE ===")
    logger.info(f"Testo Originale: '{text}'")

    # Pipeline di Esecuzione Sincrona
    data = {"text": text}

    # Passaggio 1
    logger.info("\n[Agent 1] Esecuzione Preprocessing...")
    data = agent1.process(data)
    logger.info(f"  Testo pulito: {data.get('clean_text', '')}")

    # Passaggio 2
    logger.info("\n[Agent 2] Esecuzione NER & Masking Ontologico...")
    data = agent2.process(data)
    # Correggiamo la discrepanza del nome della chiave tra Agente 2 e il resto della pipeline
    total_entities = data.get("total_entities_found", len(data.get("extracted_entities", [])))
    data["entities_detected_count"] = total_entities
    logger.info(f"  Entità rilevate: {total_entities}. Testo mascherato: {data.get('masked_text', '')}")

    # Passaggio 3
    logger.info("\n[Agent 3] Estrazione Sentiment ed Emozioni...")
    data = agent3.process(data)
    # Trova l'emozione dominante dal dizionario 'emotion_scores'
    emotion_scores = data.get("emotion_scores", {})
    if emotion_scores:
        top_emotion = max(emotion_scores.items(), key=lambda x: x[1])
        logger.info(f"  Emozione dominante: {top_emotion[0]} ({top_emotion[1]:.4f}) | Scores: {emotion_scores}")
    
    # Appiattiamo le feature per l'Agente 5 (che si aspetta le chiavi del CSV)
    data["sent_neg"] = data.get("sentiment_vector", [0,0,0])[0]
    data["sent_neu"] = data.get("sentiment_vector", [0,0,0])[1]
    data["sent_pos"] = data.get("sentiment_vector", [0,0,0])[2]
    for emo_name, score in emotion_scores.items():
        data[f"emo_{emo_name}"] = score

    # Passaggio 4
    logger.info("\n[Agent 4] Analisi Tematica (Topic Modeling)...")
    data = agent4.process(data)
    logger.info(f"  Topic Prob: {data.get('topic_probability', 0.0):.4f}")

    # Passaggio 5 Finale
    logger.info("\n[Agent 5] Valutazione Clinica Decisionale...")
    data = agent5.process(data)

    logger.info("=============================================")
    logger.info("          REFERTO CLINICO PREVISTO           ")
    logger.info("=============================================")
    logger.info(f"• Condizione Predetta : {data.get('predicted_condition')} (Confidenza: {data.get('primary_confidence'):.2f})")
    logger.info(f"• Livello di Rischio  : {data.get('clinical_risk_level')}")
    logger.info(f"• Sleep Deprivation   : {data.get('sleep_dep_score'):.2f}")
    logger.info(f"• Rischio Insonnia    : {data.get('auxiliary_probabilities', {}).get('p_insomnia_risk', 0.0):.2%}")
    logger.info(f"• Rischio Sostanze    : {data.get('auxiliary_probabilities', {}).get('p_substance_risk', 0.0):.2%}")
    logger.info("=============================================\n")

    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Esegui pipeline end-to-end su testo crudo.")
    parser.add_argument(
        "--text", 
        type=str, 
        default="Ultimamente non riesco a chiudere occhio, mi sveglio con i sudori freddi e mi sento morire. A volte prendo delle gocce di xanax ma non serve a nulla.",
        help="Il testo crudo da analizzare."
    )
    parser.add_argument(
        "--model-dir", 
        type=str, 
        default="models/decision_multitask/",
        help="Directory dove è salvato il modello addestrato (Agente 5)."
    )
    args = parser.parse_args()

    run_inference(args.text, args.model_dir)
