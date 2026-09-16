import os
import re
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import spacy

from src.agents.base_agent import BaseAgent
from src.utils.config_loader import load_config
from src.utils.logger import get_logger


class OntologicalNERAgent(BaseAgent):
    """
    Agente 2: Riconoscimento Entità Ontologico (NER) & Entity Masking.
    La conoscenza è caricata dinamicamente al 100% dalle ontologie su disco:
    - Drug Abuse Ontology (DAO): data/ontology/dao_substances_map.json
    - Criteri Clinici DSM-5 / MFOMD: data/ontology/dsm5_symptoms_map.json
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        custom_mapping: Optional[Dict[str, str]] = None,
        nlp: Optional[spacy.language.Language] = None
    ):
        if config is None:
            config = load_config()

        super().__init__(
            agent_id="agent_2",
            name="OntologicalNERAgent",
            config=config
        )

        agent_cfg = self.config.get("agent2_ner_ontology", {})
        self.dao_path = agent_cfg.get("dao_ontology_path", "data/ontology/dao_substances_map.json")
        self.dsm5_path = agent_cfg.get("dsm5_ontology_path", "data/ontology/dsm5_symptoms_map.json")
        self.enable_masking = agent_cfg.get("enable_entity_masking", True)

        # Carica o usa la mappatura ontologica personalizzata
        if custom_mapping is not None:
            self.mapping = dict(custom_mapping)
        else:
            self.mapping = self._load_ontologies_dynamically()

        # Estrae DINAMICAMENTE tutte le categorie uniche presenti nella base di conoscenza
        self.ontology_categories = sorted(list(set(self.mapping.values())))
        self.logger.info(
            f"Base di conoscenza caricata con successo: {len(self.mapping)} termini mappati "
            f"su {len(self.ontology_categories)} categorie ontologiche uniche."
        )

        # Inizializza spaCy per supporto linguistico
        spacy_model = self.config.get("agent1_preprocessing", {}).get("spacy_model", "en_core_web_sm")
        if nlp is not None:
            self.nlp = nlp
        else:
            try:
                self.nlp = spacy.load(spacy_model)
            except Exception:
                self.nlp = spacy.blank("en")

        # Compila il motore di ricerca regex ottimizzato
        self._compile_matching_engine()

    def _load_ontologies_dynamically(self) -> Dict[str, str]:
        """
        Carica le definizioni direttamente dai file ontologici su disco,
        senza alcuna lista cablata nel codice o nei file di configurazione.
        """
        full_map: Dict[str, str] = {}
        root_dir = Path(__file__).resolve().parent.parent.parent

        # 1. Caricamento Drug Abuse Ontology (DAO)
        dao_file = Path(self.dao_path)
        if not dao_file.is_absolute():
            dao_file = root_dir / self.dao_path

        if dao_file.exists():
            with open(dao_file, "r", encoding="utf-8") as f:
                dao_data = json.load(f)
                mappings = dao_data.get("mappings", {})
                for k, v in mappings.items():
                    full_map[k.lower().strip()] = v
            self.logger.info(f"Caricati {len(mappings)} termini da DAO: {dao_file.name}")
        else:
            self.logger.warning(f"File DAO non trovato in {dao_file}")

        # 2. Caricamento Sintomi Clinici DSM-5 / MFOMD
        dsm5_file = Path(self.dsm5_path)
        if not dsm5_file.is_absolute():
            dsm5_file = root_dir / self.dsm5_path

        if dsm5_file.exists():
            with open(dsm5_file, "r", encoding="utf-8") as f:
                dsm5_data = json.load(f)
                symptoms = dsm5_data.get("symptoms", {})
                for k, v in symptoms.items():
                    full_map[k.lower().strip()] = v
            self.logger.info(f"Caricati {len(symptoms)} sintomi DSM-5 da: {dsm5_file.name}")
        else:
            self.logger.warning(f"File DSM-5 non trovato in {dsm5_file}")

        return full_map

    def _compile_matching_engine(self) -> None:
        """
        Compila l'espressione regolare ordinata per lunghezza decrescente,
        garantendo che espressioni composte (es. 'dirty sprite', 'panic attack')
        vengano riconosciute prima delle singole parole isolate.
        """
        if not self.mapping:
            self.pattern = None
            return

        sorted_keys = sorted(self.mapping.keys(), key=len, reverse=True)
        escaped_keys = [re.escape(k) for k in sorted_keys]
        regex_str = r"\b(" + "|".join(escaped_keys) + r")\b"
        self.pattern = re.compile(regex_str, flags=re.IGNORECASE)

    def mask_and_extract_entities(self, text: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Scansiona il testo, estrae tutte le entità rilevate e genera la versione mascherata.
        """
        if not text or not self.pattern:
            return text, []

        extracted_entities = []

        def replace_fn(match: re.Match) -> str:
            matched_text = match.group(0)
            matched_key = matched_text.lower()
            category = self.mapping.get(matched_key, "[UNKNOWN_ENTITY]")

            extracted_entities.append({
                "entity": matched_text,
                "normalized_key": matched_key,
                "category": category,
                "start": match.start(),
                "end": match.end()
            })

            return category if self.enable_masking else matched_text

        masked_text = self.pattern.sub(replace_fn, text)
        return masked_text, extracted_entities

    def compute_feature_vector(self, entities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Genera il vettore numerico denso di feature ontologiche per l'Agente 6.
        """
        counts = {cat: 0 for cat in self.ontology_categories}
        for ent in entities:
            cat = ent.get("category")
            if cat in counts:
                counts[cat] += 1

        feature_vector = [float(counts[cat]) for cat in self.ontology_categories]

        return {
            "ontology_feature_vector": feature_vector,
            "ontology_counts": counts,
            "total_entities_found": len(entities)
        }

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Elabora il record, mascherando le entità ed estraendo le feature ontologiche.
        """
        input_text = data.get("clean_text") or data.get("text", "")
        masked_text, entities = self.mask_and_extract_entities(input_text)
        features = self.compute_feature_vector(entities)

        result = dict(data)
        result["masked_text"] = masked_text
        result["extracted_entities"] = entities
        result["ontology_feature_vector"] = features["ontology_feature_vector"]
        result["ontology_counts"] = features["ontology_counts"]
        result["total_entities_found"] = features["total_entities_found"]

        return result

    def save_pretrained(self, save_directory: Union[str, Path]) -> None:
        """
        Salva la configurazione e la base di conoscenza compilata dell'Agente 2.
        """
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)

        config_data = {
            "agent_id": self.agent_id,
            "name": self.name,
            "enable_entity_masking": self.enable_masking,
            "ontology_categories": self.ontology_categories,
            "total_rules": len(self.mapping)
        }

        with open(save_path / "ontology_ner_config.json", "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)

        with open(save_path / "compiled_ontology_map.json", "w", encoding="utf-8") as f:
            json.dump(self.mapping, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Agente 2 (NER Ontologico) salvato con successo in: {save_path}")

    @classmethod
    def from_pretrained(cls, save_directory: Union[str, Path]) -> "OntologicalNERAgent":
        """
        Ricarica l'Agente 2 da disco locale.
        """
        load_path = Path(save_directory)
        cfg_file = load_path / "ontology_ner_config.json"
        map_file = load_path / "compiled_ontology_map.json"

        if not cfg_file.exists() or not map_file.exists():
            raise FileNotFoundError(f"File di configurazione mancanti in: {load_path}")

        with open(cfg_file, "r", encoding="utf-8") as f:
            cfg_data = json.load(f)

        with open(map_file, "r", encoding="utf-8") as f:
            compiled_map = json.load(f)

        config = {
            "agent2_ner_ontology": {
                "enable_entity_masking": cfg_data.get("enable_entity_masking", True)
            }
        }

        agent = cls(config=config, custom_mapping=compiled_map)
        agent.logger.info(f"Agente 2 ricaricato con successo da: {load_path}")
        return agent
