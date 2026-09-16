"""
Test unitari per l'Agente 5 (MultiTaskDecisionAgent & MultiTaskClinicalTransformer).
"""

from pathlib import Path
import pytest
import torch

from src.agents.agent5_decision_multitask import MultiTaskDecisionAgent, PRIMARY_CLASSES, AUXILIARY_CLASSES
from src.models.multitask_transformer import MultiTaskClinicalTransformer


@pytest.fixture(scope="module")
def mock_config():
    return {
        "agent6_decision_multitask": {
            "backbone_model": "FacebookAI/roberta-base",
        },
        "risk_scoring": {
            "alpha": 0.5,
            "beta": 0.5,
            "critical_threshold": 0.75,
            "warning_threshold": 0.50,
        },
        "general": {"device": "cpu"},
    }


def test_transformer_forward_pass():
    """Verifica che la forward pass generi logits con le giuste dimensioni."""
    model = MultiTaskClinicalTransformer(
        backbone_model_name="FacebookAI/roberta-base",
        num_primary_classes=4,
        num_auxiliary_classes=2,
        tabular_feature_dim=12,
    )
    model.eval()

    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones((batch_size, seq_len))
    tabular_features = torch.randn((batch_size, 12))

    with torch.no_grad():
        outputs = model(input_ids, attention_mask, tabular_features)

    assert "primary_logits" in outputs
    assert "auxiliary_logits" in outputs
    assert outputs["primary_logits"].shape == (batch_size, 4)
    assert outputs["auxiliary_logits"].shape == (batch_size, 2)


def test_sleep_dep_score_levels(mock_config):
    """Verifica le soglie di rischio clinico dello SleepDepScore."""
    agent = MultiTaskDecisionAgent(config=mock_config)

    # Score Normale
    score, level = agent.calculate_sleep_dep_score(p_depression=0.2, p_insomnia=0.2)
    assert score == 0.20
    assert level == "NORMAL"

    # Score Warning (>= 0.50)
    score, level = agent.calculate_sleep_dep_score(p_depression=0.6, p_insomnia=0.5)
    assert score == 0.55
    assert level == "WARNING"

    # Score Critical (>= 0.75)
    score, level = agent.calculate_sleep_dep_score(p_depression=0.9, p_insomnia=0.8)
    assert score == 0.85
    assert level == "CRITICAL"


def test_extract_tabular_vector(mock_config):
    """Verifica che il vettore tabulare contenga esattamente 12 dimensioni."""
    agent = MultiTaskDecisionAgent(config=mock_config)

    sample_record = {
        "sent_neg": 0.8,
        "sent_neu": 0.1,
        "sent_pos": 0.1,
        "emo_joy": 0.0,
        "emo_sadness": 0.7,
        "emo_anger": 0.2,
        "emo_fear": 0.5,
        "emo_love": 0.0,
        "emo_surprise": 0.1,
        "emo_gratitude": 0.0,
        "entities_detected_count": 2,
        "topic_probability": 0.95,
    }

    vec = agent.extract_tabular_vector(sample_record)
    assert vec.shape == (1, 12)
    assert vec[0, 0].item() == pytest.approx(0.8)
    assert vec[0, 10].item() == pytest.approx(0.2)  # 2 / 10


def test_predict_and_process(mock_config):
    """Verifica l'interfaccia process e il calcolo delle predizioni."""
    agent = MultiTaskDecisionAgent(config=mock_config)

    record = {
        "clean_text": "I feel hopeless and I can never sleep at night.",
        "sent_neg": 0.9,
        "sent_neu": 0.05,
        "sent_pos": 0.05,
        "emo_sadness": 0.8,
        "entities_detected_count": 1,
        "topic_probability": 0.85,
    }

    result = agent.process(record)

    assert "predicted_condition" in result
    assert result["predicted_condition"] in PRIMARY_CLASSES
    assert "primary_confidence" in result
    assert "sleep_dep_score" in result
    assert "clinical_risk_level" in result
    assert result["clinical_risk_level"] in ["NORMAL", "WARNING", "CRITICAL"]


def test_save_and_reload_agent5(mock_config, tmp_path):
    """Verifica il ciclo di salvataggio e ricaricamento dei pesi."""
    agent = MultiTaskDecisionAgent(config=mock_config)
    save_dir = tmp_path / "saved_agent5"
    agent.save_pretrained(save_dir)

    assert (save_dir / "pytorch_model.bin").exists()
    assert (save_dir / "decision_agent_config.json").exists()

    reloaded = MultiTaskDecisionAgent.from_pretrained(save_dir)
    assert reloaded.is_trained
    assert reloaded.backbone_name == agent.backbone_name
