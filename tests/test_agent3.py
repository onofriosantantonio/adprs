"""
Test unitari per l'Agente 3 (SentimentEmotionAgent).

I test sono organizzati in due livelli:
  1. Test OFFLINE (senza modelli HuggingFace) — mock della pipeline
  2. Test di integrazione leggeri (richiedono i modelli scaricati)

Esegui con:
  pytest tests/test_agent3.py -v
  pytest tests/test_agent3.py -v -k "not integration"   # solo mock
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.agents.agent3_sentiment_emotion import (
    SENTIMENT_CLASSES,
    SentimentEmotionAgent,
    _resolve_device,
)
from src.utils.config_loader import load_config


# ===========================================================================
# Fixture
# ===========================================================================


@pytest.fixture(scope="module")
def minimal_config():
    """Configurazione minimale che non richiede GPU né HuggingFace."""
    return {
        "agent3_sentiment_emotion": {
            "sentiment_model": "cardiffnlp/twitter-roberta-base-sentiment-latest",
            "emotion_model": "SamLowe/roberta-base-go_emotions",
            "emotions": ["joy", "sadness", "anger", "fear", "love", "surprise", "gratitude"],
        },
        "general": {"device": "cpu"},
    }


@pytest.fixture(scope="module")
def agent_no_models(minimal_config):
    """Agente istanziato senza caricare i modelli (lazy loading)."""
    return SentimentEmotionAgent(config=minimal_config)


def _make_mock_sentiment_output():
    """Output sintetico della pipeline Cardiff (return_all_scores=True)."""
    return [[
        {"label": "negative", "score": 0.75},
        {"label": "neutral",  "score": 0.15},
        {"label": "positive", "score": 0.10},
    ]]


def _make_mock_emotion_output(dominant="sadness"):
    """Output sintetico della pipeline GoEmotions (return_all_scores=True)."""
    base_emotions = [
        "admiration", "amusement", "anger", "annoyance", "approval",
        "caring", "confusion", "curiosity", "desire", "disappointment",
        "disapproval", "disgust", "embarrassment", "excitement", "fear",
        "gratitude", "grief", "joy", "love", "nervousness",
        "optimism", "pride", "realization", "relief", "remorse",
        "sadness", "surprise", "neutral",
    ]
    scores = []
    for emo in base_emotions:
        score = 0.80 if emo == dominant else 0.01
        scores.append({"label": emo, "score": score})
    return [scores]


# ===========================================================================
# 1. Test configurazione e inizializzazione
# ===========================================================================


def test_agent_init_no_model_load(agent_no_models):
    """I modelli NON devono essere caricati prima della prima chiamata."""
    assert agent_no_models._sentiment_pipeline is None
    assert agent_no_models._emotion_pipeline is None


def test_target_emotions_count(agent_no_models):
    """Devono esserci esattamente 7 emozioni target."""
    assert len(agent_no_models.target_emotions) == 7


def test_feature_vector_dim(agent_no_models):
    """Il vettore denso deve avere dimensione 10 (3 sentiment + 7 emozioni)."""
    s_vec = [0.5, 0.3, 0.2]
    e_vec = [0.1, 0.8, 0.05, 0.3, 0.1, 0.02, 0.15]
    fv = agent_no_models.build_feature_vector(s_vec, e_vec)
    assert len(fv) == 10
    assert fv == s_vec + e_vec


def test_resolve_device_cpu():
    assert _resolve_device("cpu") == -1


def test_resolve_device_mps():
    assert _resolve_device("mps") == -1


# ===========================================================================
# 2. Test analyze_sentiment (con mock)
# ===========================================================================


def test_analyze_sentiment_label(agent_no_models):
    """Verifica che il label dominante sia 'negative' con i mock."""
    with patch.object(agent_no_models, "_load_models"):
        agent_no_models._sentiment_pipeline = MagicMock(
            return_value=_make_mock_sentiment_output()
        )
        label, score, vector = agent_no_models.analyze_sentiment(
            "I feel terrible and hopeless."
        )

    assert label == "negative"
    assert 0.0 <= score <= 1.0
    assert len(vector) == 3
    assert abs(sum(vector) - 1.0) < 0.01  # i 3 score devono sommare ~1


def test_analyze_sentiment_empty_text(agent_no_models):
    """Testo vuoto → fallback su 'neutral'."""
    agent_no_models._sentiment_pipeline = None  # reset lazy load
    agent_no_models._emotion_pipeline = None

    with patch.object(agent_no_models, "_load_models"):
        label, score, vector = agent_no_models.analyze_sentiment("")

    assert label == "neutral"
    assert score == 0.0
    assert vector == [0.0, 1.0, 0.0]


# ===========================================================================
# 3. Test analyze_emotions (con mock)
# ===========================================================================


def test_analyze_emotions_target_keys(agent_no_models):
    """Il dict restituito deve avere esattamente le 7 emozioni target."""
    with patch.object(agent_no_models, "_load_models"):
        agent_no_models._emotion_pipeline = MagicMock(
            return_value=_make_mock_emotion_output("sadness")
        )
        agent_no_models._sentiment_pipeline = MagicMock(
            return_value=_make_mock_sentiment_output()
        )
        emotion_scores, emotion_vector = agent_no_models.analyze_emotions(
            "Everything feels dark and hopeless."
        )

    assert set(emotion_scores.keys()) == set(agent_no_models.target_emotions)
    assert len(emotion_vector) == 7


def test_analyze_emotions_dominant(agent_no_models):
    """'sadness' deve essere l'emozione con score più alto."""
    with patch.object(agent_no_models, "_load_models"):
        agent_no_models._emotion_pipeline = MagicMock(
            return_value=_make_mock_emotion_output("sadness")
        )
        emotion_scores, _ = agent_no_models.analyze_emotions(
            "I feel deeply sad and alone."
        )

    assert emotion_scores["sadness"] == max(emotion_scores.values())


def test_analyze_emotions_empty_text(agent_no_models):
    """Testo vuoto → tutti gli score a 0.0."""
    with patch.object(agent_no_models, "_load_models"):
        e_scores, e_vec = agent_no_models.analyze_emotions("")

    assert all(v == 0.0 for v in e_scores.values())
    assert all(v == 0.0 for v in e_vec)


# ===========================================================================
# 4. Test process() end-to-end (con mock)
# ===========================================================================


def test_process_adds_all_keys(agent_no_models):
    """Il metodo process() deve aggiungere tutte le chiavi attese al payload."""
    with patch.object(agent_no_models, "_load_models"):
        agent_no_models._sentiment_pipeline = MagicMock(
            return_value=_make_mock_sentiment_output()
        )
        agent_no_models._emotion_pipeline = MagicMock(
            return_value=_make_mock_emotion_output("fear")
        )
        result = agent_no_models.process({
            "clean_text": "I am scared and anxious all the time.",
            "status": "Anxiety",
        })

    assert "sentiment_label" in result
    assert "sentiment_score" in result
    assert "sentiment_vector" in result
    assert "emotion_scores" in result
    assert "emotion_vector_dense" in result

    # Il vettore denso deve avere dim=10
    assert len(result["emotion_vector_dense"]) == 10

    # I dati originali non devono essere persi
    assert result["status"] == "Anxiety"


def test_process_preserves_original_data(agent_no_models):
    """process() non deve rimuovere le chiavi già presenti nel payload."""
    payload = {
        "text": "Some test text",
        "clean_text": "some test text",
        "masked_text": "some test text",
        "entities_detected_count": 0,
        "Unique_ID": 42,
        "status": "Normal",
    }
    with patch.object(agent_no_models, "_load_models"):
        agent_no_models._sentiment_pipeline = MagicMock(
            return_value=_make_mock_sentiment_output()
        )
        agent_no_models._emotion_pipeline = MagicMock(
            return_value=_make_mock_emotion_output()
        )
        result = agent_no_models.process(payload)

    for key in payload:
        assert key in result, f"Chiave originale persa: {key}"


# ===========================================================================
# 5. Test save / load
# ===========================================================================


def test_save_pretrained_creates_config(agent_no_models, tmp_path):
    """save_pretrained() deve creare il file di configurazione JSON."""
    save_dir = tmp_path / "agent3_saved"
    agent_no_models.save_pretrained(save_dir)

    config_file = save_dir / "sentiment_emotion_config.json"
    assert config_file.exists()

    with open(config_file) as f:
        saved = json.load(f)

    assert saved["sentiment_model"] == agent_no_models.sentiment_model_name
    assert saved["emotion_model"] == agent_no_models.emotion_model_name
    assert saved["target_emotions"] == agent_no_models.target_emotions
    assert saved["feature_vector_dim"] == 10


def test_from_pretrained_restores_config(agent_no_models, tmp_path):
    """from_pretrained() deve ripristinare correttamente la configurazione."""
    save_dir = tmp_path / "agent3_reload"
    agent_no_models.save_pretrained(save_dir)

    reloaded = SentimentEmotionAgent.from_pretrained(save_dir)

    assert reloaded.sentiment_model_name == agent_no_models.sentiment_model_name
    assert reloaded.emotion_model_name == agent_no_models.emotion_model_name
    assert reloaded.target_emotions == agent_no_models.target_emotions


def test_from_pretrained_missing_file(tmp_path):
    """from_pretrained() deve sollevare FileNotFoundError se il file manca."""
    with pytest.raises(FileNotFoundError):
        SentimentEmotionAgent.from_pretrained(tmp_path / "non_existing_dir")
