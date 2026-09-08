import sys
import pytest

def test_python_version():
    assert sys.version_info >= (3, 10), "Richiesta versione Python >= 3.10"

def test_imports():
    """Verifica che tutte le librerie core siano installate e importabili."""
    import torch
    import transformers
    import datasets
    import spacy
    import bertopic
    import sentence_transformers
    import sklearn
    import yaml
    import pydantic

    assert torch.__version__ is not None
    assert transformers.__version__ is not None
    assert spacy.__version__ is not None
    assert bertopic.__version__ is not None

def test_torch_device():
    """Verifica la disponibilità di MPS su Apple Silicon o fallback su CPU."""
    import torch
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        x = torch.ones((2, 2), device=device)
        assert x.device.type == "mps"
    else:
        device = torch.device("cpu")
        x = torch.ones((2, 2), device=device)
        assert x.device.type == "cpu"

def test_config_loader():
    """Verifica il caricamento corretto del file config.yaml in inglese."""
    from src.utils.config_loader import load_config
    cfg = load_config()
    assert cfg["general"]["language"] == "en"
    assert "agent1_preprocessing" in cfg
    assert "agent2_ner_ontology" in cfg
    assert "agent3_sentiment_emotion" in cfg
    assert "agent4_topic_modeling" in cfg
    assert "agent6_decision_multitask" in cfg
    assert "risk_scoring" in cfg

def test_spacy_english_model():
    """Verifica che il modello spaCy inglese en_core_web_sm sia correttamente caricabile."""
    import spacy
    nlp = spacy.load("en_core_web_sm")
    doc = nlp("Patient expresses severe anxiety and insomnia symptoms.")
    assert len(doc) > 0
