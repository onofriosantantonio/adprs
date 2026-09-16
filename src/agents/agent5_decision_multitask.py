"""
Agente Decisionale Multi-Task e Risk Scoring (Agente 5 / Decision Agent).

Compiti dell'Agente:
  - Riceve il payload completo arricchito da tutti gli agenti a monte
  - Esegue la classificazione congiunta multi-task (Stato di salute mentale + Rischio correlato)
  - Calcola l'indice clinico SleepDepScore: alpha * P(insomnia) + beta * P(depression)
  - Determina le soglie di rischio (NORMAL, WARNING, CRITICAL)
  - Fornisce un report diagnostico/decisionale interpretabile
  - Salva e ricarica i pesi del modello addestrato (save_pretrained / from_pretrained)
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from src.agents.base_agent import BaseAgent
from src.models.multitask_transformer import MultiTaskClinicalTransformer
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

PRIMARY_CLASSES = ["Normal", "Depression", "Anxiety", "Suicidal"]
AUXILIARY_CLASSES = ["Insomnia_Risk", "Substance_Abuse_Risk"]


class MultiTaskDecisionAgent(BaseAgent):
    """
    Agente 5: Agente Decisionale Multi-Task & Risk Scoring.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        model: Optional[MultiTaskClinicalTransformer] = None,
        tokenizer: Optional[AutoTokenizer] = None,
        freeze_backbone: bool = True,
    ):
        if config is None:
            config = load_config()

        super().__init__(
            agent_id="agent_5",
            name="MultiTaskDecisionAgent",
            config=config,
        )

        agent_cfg = self.config.get("agent6_decision_multitask", {})
        risk_cfg = self.config.get("risk_scoring", {})
        general_cfg = self.config.get("general", {})

        self.backbone_name: str = agent_cfg.get("backbone_model", "FacebookAI/roberta-base")
        self.alpha: float = float(risk_cfg.get("alpha", 0.5))
        self.beta: float = float(risk_cfg.get("beta", 0.5))
        self.critical_threshold: float = float(risk_cfg.get("critical_threshold", 0.75))
        self.warning_threshold: float = float(risk_cfg.get("warning_threshold", 0.50))
        self.freeze_backbone: bool = freeze_backbone

        # Configurazione Device
        device_str = general_cfg.get("device", "cpu")
        if device_str == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif device_str == "mps" and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.logger.info(f"MultiTaskDecisionAgent inizializzato sul device: {self.device}")

        # Inizializza o assegna Tokenizer e Modello
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.backbone_name)

        if model is not None:
            self.model = model.to(self.device)
            self.is_trained = True
        else:
            self.model = MultiTaskClinicalTransformer(
                backbone_model_name=self.backbone_name,
                num_primary_classes=len(PRIMARY_CLASSES),
                num_auxiliary_classes=len(AUXILIARY_CLASSES),
                tabular_feature_dim=12,
                freeze_backbone=freeze_backbone,
            ).to(self.device)
            self.is_trained = False

        self.model.eval()

    def extract_tabular_vector(self, data: Dict[str, Any]) -> torch.Tensor:
        """
        Estrae e normalizza il vettore tabulare (dim=12) dai campi prodotti dagli agenti a monte:
        - 3 sentiment scores (neg, neu, pos)
        - 7 emotion scores (joy, sadness, anger, fear, love, surprise, gratitude)
        - 1 entities detected count (normalizzato)
        - 1 topic probability
        """
        sent_neg = float(data.get("sent_neg", 0.0))
        sent_neu = float(data.get("sent_neu", 0.0))
        sent_pos = float(data.get("sent_pos", 0.0))

        emo_joy = float(data.get("emo_joy", 0.0))
        emo_sad = float(data.get("emo_sadness", 0.0))
        emo_ang = float(data.get("emo_anger", 0.0))
        emo_fea = float(data.get("emo_fear", 0.0))
        emo_lov = float(data.get("emo_love", 0.0))
        emo_sur = float(data.get("emo_surprise", 0.0))
        emo_gra = float(data.get("emo_gratitude", 0.0))

        # Feature ontologica e tematica
        entities_cnt = min(float(data.get("entities_detected_count", 0.0)) / 10.0, 1.0)
        topic_prob = float(data.get("topic_probability", 0.0))

        vec = [
            sent_neg, sent_neu, sent_pos,
            emo_joy, emo_sad, emo_ang, emo_fea, emo_lov, emo_sur, emo_gra,
            entities_cnt, topic_prob,
        ]
        return torch.tensor([vec], dtype=torch.float32, device=self.device)

    def calculate_sleep_dep_score(self, p_depression: float, p_insomnia: float) -> tuple[float, str]:
        """
        Calcola lo SleepDepScore combinato e assegna il livello di allarme.
        SleepDepScore = alpha * P(insomnia) + beta * P(depression)
        """
        score = (self.alpha * p_insomnia) + (self.beta * p_depression)
        score = float(np.clip(score, 0.0, 1.0))

        if score >= self.critical_threshold:
            risk_level = "CRITICAL"
        elif score >= self.warning_threshold:
            risk_level = "WARNING"
        else:
            risk_level = "NORMAL"

        return round(score, 4), risk_level

    def predict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Esegue la forward pass e calcola tutte le metriche decisionali per un record.
        """
        text = str(data.get("clean_text") or data.get("text", ""))

        # Tokenizzazione del testo
        tokens = self.tokenizer(
            text,
            max_length=256,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = tokens["input_ids"].to(self.device)
        attention_mask = tokens["attention_mask"].to(self.device)
        tabular_vec = self.extract_tabular_vector(data)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask, tabular_vec)
            p_logits = outputs["primary_logits"]
            a_logits = outputs["auxiliary_logits"]

            primary_probs = F.softmax(p_logits, dim=-1)[0].cpu().numpy()
            auxiliary_probs = torch.sigmoid(a_logits)[0].cpu().numpy()

        # Risultati Head Primaria
        pred_idx = int(np.argmax(primary_probs))
        pred_label = PRIMARY_CLASSES[pred_idx]
        primary_confidence = float(primary_probs[pred_idx])

        primary_prob_dict = {
            cls_name: round(float(prob), 4)
            for cls_name, prob in zip(PRIMARY_CLASSES, primary_probs)
        }

        # Risultati Head Ausiliaria (Insonnia e Sostanze)
        p_insomnia = float(auxiliary_probs[0])
        p_substance = float(auxiliary_probs[1])

        auxiliary_prob_dict = {
            "p_insomnia_risk": round(p_insomnia, 4),
            "p_substance_risk": round(p_substance, 4),
        }

        # Calcolo SleepDepScore
        p_depression = float(primary_prob_dict.get("Depression", 0.0))
        sleep_dep_score, risk_level = self.calculate_sleep_dep_score(p_depression, p_insomnia)

        return {
            "predicted_condition": pred_label,
            "primary_confidence": round(primary_confidence, 4),
            "primary_probabilities": primary_prob_dict,
            "auxiliary_probabilities": auxiliary_prob_dict,
            "sleep_dep_score": sleep_dep_score,
            "clinical_risk_level": risk_level,
        }

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Interfaccia conforme a BaseAgent.
        """
        predictions = self.predict(data)
        result = dict(data)
        result.update(predictions)
        return result

    def save_pretrained(self, save_directory: Union[str, Path]) -> None:
        """
        Salva i pesi del modello PyTorch, del tokenizer e la configurazione dell'Agente.
        """
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)

        # Salva pesi PyTorch
        weights_file = save_path / "pytorch_model.bin"
        torch.save(self.model.state_dict(), weights_file)

        # Salva tokenizer
        self.tokenizer.save_pretrained(str(save_path))

        # Salva configurazione agent & risk parameters
        config_data = {
            "agent_id": self.agent_id,
            "name": self.name,
            "backbone_model": self.backbone_name,
            "alpha": self.alpha,
            "beta": self.beta,
            "critical_threshold": self.critical_threshold,
            "warning_threshold": self.warning_threshold,
            "freeze_backbone": self.freeze_backbone,
            "primary_classes": PRIMARY_CLASSES,
            "auxiliary_classes": AUXILIARY_CLASSES,
        }

        with open(save_path / "decision_agent_config.json", "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Agente 5 (Decisione & MTL) salvato con successo in: {save_path}")

    @classmethod
    def from_pretrained(cls, save_directory: Union[str, Path]) -> "MultiTaskDecisionAgent":
        """
        Ricarica l'Agente 5 con pesi allenati da disco locale.
        """
        load_path = Path(save_directory)
        weights_file = load_path / "pytorch_model.bin"
        config_file = load_path / "decision_agent_config.json"

        if not weights_file.exists() or not config_file.exists():
            raise FileNotFoundError(f"File modello mancanti in: {load_path}")

        with open(config_file, "r", encoding="utf-8") as f:
            cfg_data = json.load(f)

        tokenizer = AutoTokenizer.from_pretrained(str(load_path))

        model = MultiTaskClinicalTransformer(
            backbone_model_name=cfg_data["backbone_model"],
            num_primary_classes=len(cfg_data["primary_classes"]),
            num_auxiliary_classes=len(cfg_data["auxiliary_classes"]),
            tabular_feature_dim=12,
            freeze_backbone=cfg_data.get("freeze_backbone", True),
        )

        state_dict = torch.load(weights_file, map_location="cpu")
        model.load_state_dict(state_dict)

        config = {
            "agent6_decision_multitask": {"backbone_model": cfg_data["backbone_model"]},
            "risk_scoring": {
                "alpha": cfg_data["alpha"],
                "beta": cfg_data["beta"],
                "critical_threshold": cfg_data["critical_threshold"],
                "warning_threshold": cfg_data["warning_threshold"],
            },
        }

        agent = cls(
            config=config,
            model=model,
            tokenizer=tokenizer,
            freeze_backbone=cfg_data.get("freeze_backbone", True),
        )
        agent.logger.info(f"Agente 5 ricaricato con successo da: {load_path}")
        return agent
