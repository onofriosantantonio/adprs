"""
Agente 3: Analisi del Sentiment e Estrazione del Vettore Emotivo.

Utilizza due modelli open-source pre-addestrati da Hugging Face:
  - Sentiment polarity: cardiffnlp/twitter-roberta-base-sentiment-latest
      → 3 classi: negative / neutral / positive
  - Emotion recognition: SamLowe/roberta-base-go_emotions
      → 28 emozioni GoEmotions

I modelli vengono scaricati automaticamente alla prima esecuzione
e restano in cache locale (nessun fine-tuning necessario).

Output aggiunto al payload della pipeline:
  - sentiment_label       : str  (negative / neutral / positive)
  - sentiment_score       : float  (confidenza della predizione)
  - sentiment_vector      : List[float]  (3 score softmax)
  - emotion_scores        : Dict[str, float]  (28 emozioni GoEmotions)
  - emotion_vector_dense  : List[float]  (31 valori: sentiment 3 + emotion 28)
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from transformers import pipeline as hf_pipeline

from src.agents.base_agent import BaseAgent
from src.utils.config_loader import load_config
from src.utils.constants import ALL_EMOTIONS
from src.utils.logger import get_logger

# Mappa label raw → label normalizzata per il modello sentiment Cardiff
_SENTIMENT_LABEL_MAP = {
    "LABEL_0": "negative",
    "LABEL_1": "neutral",
    "LABEL_2": "positive",
    "negative": "negative",
    "neutral": "neutral",
    "positive": "positive",
}

# Ordine fisso delle label sentiment nel vettore (garantisce consistenza)
SENTIMENT_CLASSES = ["negative", "neutral", "positive"]

# Dimensione del vettore denso: 3 sentiment + 28 emozioni
FEATURE_VECTOR_DIM = len(SENTIMENT_CLASSES) + len(ALL_EMOTIONS)


def _unwrap_pipeline_output(raw_output) -> list:
    """
    Normalizza l'output di transformers.pipeline con return_all_scores=True
    per un singolo testo, gestendo i due formati possibili:
      - Lista annidata: [[{"label": ..., "score": ...}, ...]]  → restituisce raw_output[0]
      - Lista piatta:   [{"label": ..., "score": ...}, ...]    → restituisce raw_output
    """
    if not raw_output:
        return []
    first = raw_output[0]
    if isinstance(first, list):
        return first
    return raw_output


def _resolve_device(device_str: str) -> int:
    """
    Converte la stringa di configurazione del device nel formato
    atteso da transformers.pipeline:
      - 'cuda' → 0
      - 'mps'  → -1 (la pipeline usa MPS automaticamente se disponibile)
      - 'cpu'  → -1
    """
    if device_str == "cuda" and torch.cuda.is_available():
        return 0
    return -1


class SentimentEmotionAgent(BaseAgent):
    """
    Agente 3: Sentiment Polarity + Emotion Vector Extraction.

    Carica due pipeline Hugging Face in modalità inference-only
    (nessun addestramento richiesto), entrambe basate su RoBERTa.
    Restituisce tutte e 28 le emozioni GoEmotions.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
    ):
        if config is None:
            config = load_config()

        super().__init__(
            agent_id="agent_3",
            name="SentimentEmotionAgent",
            config=config,
        )

        agent_cfg = self.config.get("agent3_sentiment_emotion", {})
        general_cfg = self.config.get("general", {})

        self.sentiment_model_name: str = agent_cfg.get(
            "sentiment_model",
            "cardiffnlp/twitter-roberta-base-sentiment-latest",
        )
        self.emotion_model_name: str = agent_cfg.get(
            "emotion_model",
            "SamLowe/roberta-base-go_emotions",
        )
        self.emotions: List[str] = ALL_EMOTIONS

        device_str = general_cfg.get("device", "cpu")
        self._device_id = _resolve_device(device_str)
        self.logger.info(
            f"Device selezionato: '{device_str}' "
            f"(pipeline device_id={self._device_id})"
        )

        self._sentiment_pipeline = None
        self._emotion_pipeline = None

    def _load_models(self) -> None:
        """Carica entrambe le pipeline HuggingFace (una sola volta)."""
        if self._sentiment_pipeline is not None:
            return

        self.logger.info(
            f"Caricamento modello sentiment: {self.sentiment_model_name}"
        )
        self._sentiment_pipeline = hf_pipeline(
            task="text-classification",
            model=self.sentiment_model_name,
            return_all_scores=True,
            truncation=True,
            max_length=512,
            device=self._device_id,
        )

        self.logger.info(
            f"Caricamento modello emotion: {self.emotion_model_name}"
        )
        self._emotion_pipeline = hf_pipeline(
            task="text-classification",
            model=self.emotion_model_name,
            return_all_scores=True,
            truncation=True,
            max_length=512,
            device=self._device_id,
        )
        self.logger.info("Entrambe le pipeline caricate con successo.")

    def analyze_sentiment(self, text: str) -> Tuple[str, float, List[float]]:
        """
        Predice il sentiment del testo.

        Returns:
            label  : classe dominante (negative / neutral / positive)
            score  : confidenza della classe dominante
            vector : lista di 3 score in ordine [negative, neutral, positive]
        """
        self._load_models()

        if not text or not text.strip():
            return "neutral", 0.0, [0.0, 1.0, 0.0]

        raw_output = self._sentiment_pipeline(text)
        raw_results: List[Dict] = _unwrap_pipeline_output(raw_output)

        score_map: Dict[str, float] = {}
        for item in raw_results:
            normalized = _SENTIMENT_LABEL_MAP.get(
                item["label"].lower(), item["label"].lower()
            )
            score_map[normalized] = item["score"]

        vector = [score_map.get(cls, 0.0) for cls in SENTIMENT_CLASSES]

        dominant_label = max(score_map, key=score_map.get)
        dominant_score = score_map[dominant_label]

        return dominant_label, dominant_score, vector

    def analyze_emotions(self, text: str) -> Tuple[Dict[str, float], List[float]]:
        """
        Predice il vettore delle 28 emozioni GoEmotions dal modello.

        Returns:
            emotion_scores : dict {emotion_name: score} per tutte le 28 emozioni
            emotion_vector : lista ordinata dei 28 score (stesso ordine di ALL_EMOTIONS)
        """
        self._load_models()

        if not text or not text.strip():
            return (
                {e: 0.0 for e in self.emotions},
                [0.0] * len(self.emotions),
            )

        raw_output = self._emotion_pipeline(text)
        raw_results: List[Dict] = _unwrap_pipeline_output(raw_output)

        all_scores: Dict[str, float] = {
            item["label"].lower(): item["score"] for item in raw_results
        }

        emotion_scores = {
            emotion: all_scores.get(emotion, 0.0)
            for emotion in self.emotions
        }
        emotion_vector = [emotion_scores[e] for e in self.emotions]

        return emotion_scores, emotion_vector

    def build_feature_vector(
        self,
        sentiment_vector: List[float],
        emotion_vector: List[float],
    ) -> List[float]:
        """
        Concatena il vettore sentiment (dim=3) e il vettore emozione (dim=28)
        in un unico vettore denso (dim=31) per l'Agente 5.
        """
        return sentiment_vector + emotion_vector

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Elabora un singolo record della pipeline.

        Legge 'clean_text' (preferito) o 'text' e aggiunge al payload:
          - sentiment_label
          - sentiment_score
          - sentiment_vector       (dim=3)
          - emotion_scores         (dict 28 emozioni)
          - emotion_vector_dense   (dim=31: sentiment 3 + emotion 28)
        """
        input_text = data.get("clean_text") or data.get("text", "")

        sentiment_label, sentiment_score, sentiment_vector = self.analyze_sentiment(
            input_text
        )
        emotion_scores, emotion_vector = self.analyze_emotions(input_text)
        feature_vector = self.build_feature_vector(sentiment_vector, emotion_vector)

        result = dict(data)
        result["sentiment_label"] = sentiment_label
        result["sentiment_score"] = round(sentiment_score, 6)
        result["sentiment_vector"] = sentiment_vector
        result["emotion_scores"] = emotion_scores
        result["emotion_vector_dense"] = feature_vector

        return result

    def save_pretrained(self, save_directory: Union[str, Path]) -> None:
        """
        Salva la configurazione dell'Agente 3 su disco.
        I pesi dei modelli rimangono nella cache di Hugging Face.
        """
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)

        config_data = {
            "agent_id": self.agent_id,
            "name": self.name,
            "sentiment_model": self.sentiment_model_name,
            "emotion_model": self.emotion_model_name,
            "emotions": self.emotions,
            "sentiment_classes": SENTIMENT_CLASSES,
            "feature_vector_dim": FEATURE_VECTOR_DIM,
        }

        config_file = save_path / "sentiment_emotion_config.json"
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)

        self.logger.info(
            f"Agente 3 (Sentiment & Emotion) salvato con successo in: {save_path}"
        )

    @classmethod
    def from_pretrained(
        cls, save_directory: Union[str, Path]
    ) -> "SentimentEmotionAgent":
        """
        Ricarica l'Agente 3 dalla configurazione salvata su disco.
        """
        load_path = Path(save_directory)
        cfg_file = load_path / "sentiment_emotion_config.json"

        if not cfg_file.exists():
            raise FileNotFoundError(
                f"File di configurazione non trovato: {cfg_file}"
            )

        with open(cfg_file, "r", encoding="utf-8") as f:
            cfg_data = json.load(f)

        config = {
            "agent3_sentiment_emotion": {
                "sentiment_model": cfg_data["sentiment_model"],
                "emotion_model": cfg_data["emotion_model"],
                "emotions": cfg_data["emotions"],
            },
            "general": {"device": "cpu"},
        }

        agent = cls(config=config)
        logger = get_logger("SentimentEmotionAgent.from_pretrained")
        logger.info(f"Agente 3 ricaricato con successo da: {load_path}")
        return agent
