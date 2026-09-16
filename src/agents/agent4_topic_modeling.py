"""
Agente 4: Topic Modeling Clinico basato su BERTopic.

Compiti dell'Agente:
  - Estrazione tematiche cliniche e psicologiche (cluster UMAP + HDBSCAN)
  - Identificazione del topic associato ad ogni testo e relativa probabilità
  - Generazione di rappresentazioni c-TF-IDF con parole chiave descrittive
  - Estrazione di feature dense per l'Agente Decisionale Multi-Task (Agente 6)
  - Persistenza e caricamento del modello salvato su disco
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from bertopic import BERTopic
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

from src.agents.base_agent import BaseAgent
from src.utils.config_loader import load_config
from src.utils.logger import get_logger


class ClinicalTopicModelingAgent(BaseAgent):
    """
    Agente 4: BERTopic per la scoperta e l'estrazione di tematiche cliniche.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        model: Optional[BERTopic] = None,
    ):
        if config is None:
            config = load_config()

        super().__init__(
            agent_id="agent_4",
            name="ClinicalTopicModelingAgent",
            config=config,
        )

        agent_cfg = self.config.get("agent4_topic_modeling", {})
        general_cfg = self.config.get("general", {})

        self.embedding_model_name: str = agent_cfg.get(
            "embedding_model", "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.n_neighbors: int = int(agent_cfg.get("n_neighbors", 15))
        self.n_components: int = int(agent_cfg.get("n_components", 5))
        self.min_cluster_size: int = int(agent_cfg.get("min_cluster_size", 10))
        self.seed: int = int(general_cfg.get("seed", 42))

        # Modello precaricato o da inizializzare/addestrare
        self.model: Optional[BERTopic] = model
        self.is_fitted: bool = model is not None

        if not self.is_fitted:
            self._init_pipeline()

    def _init_pipeline(self) -> None:
        """
        Inizializza i sottomoduli di BERTopic (Embedding, UMAP, HDBSCAN, Vectorizer).
        """
        self.logger.info(f"Inizializzazione sub-componenti BERTopic con modello: {self.embedding_model_name}")

        embedding_model = SentenceTransformer(self.embedding_model_name)

        umap_model = UMAP(
            n_neighbors=self.n_neighbors,
            n_components=self.n_components,
            min_dist=0.0,
            metric="cosine",
            random_state=self.seed,
        )

        hdbscan_model = HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=True,
        )

        vectorizer_model = CountVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
        )

        self.model = BERTopic(
            embedding_model=embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            calculate_probabilities=True,
            verbose=False,
        )

    def fit(self, texts: List[str]) -> "ClinicalTopicModelingAgent":
        """
        Addestra il modello BERTopic su una collezione di testi.
        """
        self.logger.info(f"Inizio addestramento BERTopic su {len(texts)} testi...")
        if self.model is None:
            self._init_pipeline()

        self.model.fit(texts)
        self.is_fitted = True
        num_topics = len(self.model.get_topic_info()) - 1  # Escludi outlier (-1)
        self.logger.info(f"Addestramento completato! Identificati {num_topics} topic clinici principali.")
        return self

    def transform(self, texts: List[str]) -> Tuple[List[int], np.ndarray]:
        """
        Calcola i topic e le probabilità per una lista di testi.
        """
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Il modello BERTopic non è stato ancora addestrato o caricato. Esegui fit() o from_pretrained().")

        topics, probs = self.model.transform(texts)
        return topics, probs

    def get_topic_keywords(self, topic_id: int, top_n: int = 5) -> List[str]:
        """
        Restituisce le parole chiave più rappresentative per un dato topic.
        """
        if not self.is_fitted or self.model is None:
            return []
        topic_info = self.model.get_topic(topic_id)
        if not topic_info:
            return []
        return [word for word, _ in topic_info[:top_n]]

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Elabora un singolo record conforme a BaseAgent.
        """
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Il modello BERTopic non è stato ancora addestrato o caricato.")

        input_text = data.get("clean_text") or data.get("text", "")
        topics, probs = self.transform([input_text])
        topic_id = int(topics[0])

        if probs is not None and len(probs) > 0:
            if isinstance(probs[0], (np.ndarray, list)):
                topic_prob = float(np.max(probs[0]))
                dense_vector = [float(p) for p in probs[0]]
            else:
                topic_prob = float(probs[0])
                dense_vector = [topic_prob]
        else:
            topic_prob = 0.0
            dense_vector = []

        keywords = self.get_topic_keywords(topic_id, top_n=5)

        result = dict(data)
        result["topic_id"] = topic_id
        result["topic_probability"] = round(topic_prob, 6)
        result["topic_keywords"] = keywords
        result["topic_feature_vector"] = dense_vector

        return result

    def save_pretrained(self, save_directory: Union[str, Path]) -> None:
        """
        Salva sia gli artefatti del modello BERTopic che i metadati di configurazione.
        """
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Impossibile salvare un modello non addestrato.")

        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)

        model_dir = save_path / "bertopic_model"
        self.logger.info(f"Salvataggio artefatti BERTopic in: {model_dir}")
        self.model.save(str(model_dir), serialization="safetensors", save_embedding_model=True)

        # Salva metadati e info sui topic estratti
        topic_info_df = self.model.get_topic_info()
        topic_info_path = save_path / "topics_info.csv"
        topic_info_df.to_csv(topic_info_path, index=False)

        config_data = {
            "agent_id": self.agent_id,
            "name": self.name,
            "embedding_model": self.embedding_model_name,
            "n_neighbors": self.n_neighbors,
            "n_components": self.n_components,
            "min_cluster_size": self.min_cluster_size,
            "num_topics": len(topic_info_df) - 1,
        }

        with open(save_path / "topic_modeling_config.json", "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Agente 4 (Topic Modeling) salvato con successo in: {save_path}")

    @classmethod
    def from_pretrained(cls, save_directory: Union[str, Path]) -> "ClinicalTopicModelingAgent":
        """
        Ricarica l'Agente 4 da disco locale.
        """
        load_path = Path(save_directory)
        cfg_file = load_path / "topic_modeling_config.json"
        model_dir = load_path / "bertopic_model"

        if not cfg_file.exists() or not model_dir.exists():
            raise FileNotFoundError(f"File o cartella modello mancante in: {load_path}")

        with open(cfg_file, "r", encoding="utf-8") as f:
            cfg_data = json.load(f)

        config = {
            "agent4_topic_modeling": {
                "embedding_model": cfg_data.get("embedding_model"),
                "n_neighbors": cfg_data.get("n_neighbors"),
                "n_components": cfg_data.get("n_components"),
                "min_cluster_size": cfg_data.get("min_cluster_size"),
            }
        }

        logger = get_logger("ClinicalTopicModelingAgent.from_pretrained")
        embedding_model_name = cfg_data.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
        logger.info(f"Caricamento modello BERTopic da: {model_dir} con embedding model: {embedding_model_name}")
        embedding_model = SentenceTransformer(embedding_model_name)
        loaded_bertopic = BERTopic.load(str(model_dir), embedding_model=embedding_model)

        agent = cls(config=config, model=loaded_bertopic)
        agent.is_fitted = True
        logger.info(f"Agente 4 ricaricato con successo con {len(loaded_bertopic.get_topic_info()) - 1} topic.")
        return agent
