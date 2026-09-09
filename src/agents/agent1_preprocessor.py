import os
import re
import json
import html
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import spacy
from transformers import AutoTokenizer

from src.agents.base_agent import BaseAgent
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

# Regex per pulizia mirata del rumore testuale
RE_URL = re.compile(r"https?://\S+|www\.\S+")
RE_REDDIT_TAGS = re.compile(r"\b[ru]/[A-Za-z0-9_-]+\b|\[deleted\]|\[removed\]")
RE_HTML_TAGS = re.compile(r"<.*?>")
RE_EMOJIS = re.compile(
    r"[\U00010000-\U0010ffff]|[\u2600-\u26FF]|[\u2700-\u27BF]",
    flags=re.UNICODE
)
RE_CHAR_REPETITIONS = re.compile(r"(.)\1{2,}") # 'sooo' -> 'soo'
RE_PUNCT_REPETITIONS = re.compile(r"([!?,.:;])\1+") # '???!!' -> '?'


class PreprocessingAgent(BaseAgent):
    """
    Agente 1: Preprocessing e Normalizzazione del Testo.
    Implementa:
    1. Pulizia deterministica (URL, emoji, rumore social Reddit/Twitter).
    2. Lemmatizzazione spaCy con protezione delle stopword clinico-psicologiche.
    3. Tokenizzazione Transformer Hugging Face (WordPiece/BPE) con padding e troncamento.
    4. Serializzazione / caricamento (save_pretrained / from_pretrained).
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        tokenizer: Optional[AutoTokenizer] = None,
        nlp: Optional[spacy.language.Language] = None
    ):
        if config is None:
            config = load_config()

        super().__init__(
            agent_id="agent_1",
            name="PreprocessingAgent",
            config=config
        )

        agent_cfg = self.config.get("agent1_preprocessing", {})
        self.max_seq_length = agent_cfg.get("max_seq_length", 512)
        self.spacy_model_name = agent_cfg.get("spacy_model", "en_core_web_sm")
        
        # Parole sentinella cliniche da non filtrare mai come stopword
        preserved = agent_cfg.get("clinical_preserved_keywords", [])
        self.clinical_preserved_keywords = set(k.lower() for k in preserved)

        # Inizializzazione spaCy
        if nlp is not None:
            self.nlp = nlp
        else:
            self.logger.info(f"Caricamento pipeline spaCy: {self.spacy_model_name}")
            try:
                self.nlp = spacy.load(self.spacy_model_name)
            except OSError:
                self.logger.warning(f"Download automatico del modello spaCy {self.spacy_model_name}...")
                spacy.cli.download(self.spacy_model_name)
                self.nlp = spacy.load(self.spacy_model_name)

        # Configurazione personalizzata delle stopword in spaCy:
        # Assicura che le parole cliniche sentinella NON siano considerate stopword
        for kw in self.clinical_preserved_keywords:
            self.nlp.vocab[kw].is_stop = False

        # Inizializzazione Hugging Face AutoTokenizer
        backbone_name = self.config.get("agent6_decision_multitask", {}).get(
            "backbone_model", "FacebookAI/roberta-base"
        )
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.logger.info(f"Caricamento AutoTokenizer Hugging Face: {backbone_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(backbone_name)

    def clean_text(self, text: str) -> str:
        """
        Esegue la pulizia deterministica preliminare del testo grezzo.
        """
        if not isinstance(text, str):
            text = str(text) if text is not None else ""

        # Decodifica entità HTML (&amp;, &gt;, etc.)
        text = html.unescape(text)

        # Rimozione URL
        text = RE_URL.sub(" ", text)

        # Rimozione tag tipici di Reddit/forum (r/sub, u/user, [deleted])
        text = RE_REDDIT_TAGS.sub(" ", text)

        # Rimozione tag HTML
        text = RE_HTML_TAGS.sub(" ", text)

        # Rimozione emoji
        text = RE_EMOJIS.sub(" ", text)


        # Riduzione sequenze di caratteri ripetuti (es. 'depreeeessed' -> 'depressed')
        text = RE_CHAR_REPETITIONS.sub(r"\1\1", text)

        # Riduzione sequenze di punteggiatura multipla
        text = RE_PUNCT_REPETITIONS.sub(r"\1", text)

        # Normalizzazione spazi bianchi multipli
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def lemmatize_and_filter(self, text: str) -> Dict[str, Any]:
        """
        Esegue la lemmatizzazione preservando le parole sentinella cliniche
        ed eliminando il rumore lessicale / stopword non rilevanti.
        """
        doc = self.nlp(text)
        filtered_lemmas = []
        tokens_info = []

        for token in doc:
            token_lower = token.text.lower()
            lemma = token.lemma_.lower()

            # Se è una parola clinica sentinella, la preserviamo sempre
            is_clinical = (token_lower in self.clinical_preserved_keywords) or (lemma in self.clinical_preserved_keywords)

            # Rimuoviamo punteggiatura generica, spazi o stopword NON cliniche
            if (token.is_stop and not is_clinical) or token.is_punct or token.is_space:
                continue

            chosen_lemma = lemma if lemma != "-pron-" else token_lower
            filtered_lemmas.append(chosen_lemma)
            tokens_info.append({
                "text": token.text,
                "lemma": chosen_lemma,
                "pos": token.pos_,
                "is_clinical_preserved": is_clinical
            })

        lemmatized_text = " ".join(filtered_lemmas)
        return {
            "lemmatized_text": lemmatized_text,
            "tokens_info": tokens_info
        }

    def tokenize_transformer(
        self,
        text: Union[str, List[str]],
        padding: bool = True,
        truncation: bool = True
    ) -> Dict[str, Any]:
        """
        Converte il testo pulito in tensori compatibili con il Transformer (Agente 6).
        """
        return self.tokenizer(
            text,
            padding=padding,
            truncation=truncation,
            max_length=self.max_seq_length,
            return_tensors="pt"
        )

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Punto di ingresso principale della pipeline MAS per l'Agente 1.
        
        Args:
            data: Dizionario contenente almeno la chiave 'text'.
            
        Returns:
            Dizionario arricchito con 'clean_text', 'lemmatized_text',
            'tokens_info', 'input_ids', 'attention_mask'.
        """
        raw_text = data.get("text", "")
        clean_text = self.clean_text(raw_text)
        lemma_result = self.lemmatize_and_filter(clean_text)
        
        encoded = self.tokenize_transformer(clean_text)

        # Arricchimento payload per gli agenti a valle
        result = dict(data)
        result["clean_text"] = clean_text
        result["lemmatized_text"] = lemma_result["lemmatized_text"]
        result["tokens_info"] = lemma_result["tokens_info"]
        result["input_ids"] = encoded["input_ids"]
        result["attention_mask"] = encoded["attention_mask"]

        return result

    def save_pretrained(self, save_directory: Union[str, Path]) -> None:
        """
        Serializza l'Agente di Preprocessing (configurazioni, stopword, tokenizer HF)
        nella cartella di output specificata.
        """
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)

        # 1. Salva la configurazione e i metadati
        metadata = {
            "agent_id": self.agent_id,
            "name": self.name,
            "spacy_model": self.spacy_model_name,
            "max_seq_length": self.max_seq_length,
            "clinical_preserved_keywords": sorted(list(self.clinical_preserved_keywords))
        }
        with open(save_path / "preprocessor_config.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        # 2. Salva il tokenizer Hugging Face
        self.tokenizer.save_pretrained(save_path)
        self.logger.info(f"Modello di preprocessing salvato con successo in: {save_path}")

    @classmethod
    def from_pretrained(cls, save_directory: Union[str, Path]) -> "PreprocessingAgent":
        """
        Carica un'istanza dell'Agente di Preprocessing da una cartella salvata localmente.
        """
        load_path = Path(save_directory)
        config_file = load_path / "preprocessor_config.json"

        if not config_file.exists():
            raise FileNotFoundError(f"File di configurazione non trovato in: {config_file}")

        with open(config_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        # Carica il tokenizer salvato localmente
        tokenizer = AutoTokenizer.from_pretrained(load_path)

        # Ricostruisce la configurazione necessaria
        config = {
            "agent1_preprocessing": {
                "spacy_model": metadata.get("spacy_model", "en_core_web_sm"),
                "max_seq_length": metadata.get("max_seq_length", 512),
                "clinical_preserved_keywords": metadata.get("clinical_preserved_keywords", [])
            }
        }

        agent = cls(config=config, tokenizer=tokenizer)
        agent.logger.info(f"Agente di preprocessing ricaricato con successo da: {load_path}")
        return agent
