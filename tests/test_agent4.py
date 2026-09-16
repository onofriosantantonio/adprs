"""
Test unitari per l'Agente 4 (ClinicalTopicModelingAgent).

Coprono:
- Inizializzazione della pipeline da configurazione
- Fit su piccolo corpus di test
- Transform ed estrazione parole chiave
- Metodo process() conforme a BaseAgent
- Salvataggio e ricaricamento (save_pretrained e from_pretrained)
"""

from pathlib import Path
import pytest
from src.agents.agent4_topic_modeling import ClinicalTopicModelingAgent


SAMPLE_TEXTS = [
    # Cluster Insonnia / Sonno
    "I cannot sleep at night, having severe insomnia and restless thoughts.",
    "Trouble sleeping again, waking up exhausted after a few hours of sleep.",
    "No sleep for days, completely deprived of rest and energy.",
    "Lying awake in bed with racing thoughts and unable to fall asleep.",
    "Sleeping disorder is getting worse, insomnia every single night.",
    # Cluster Ansia / Panico
    "Feeling a terrible panic attack coming on, heart is racing fast.",
    "Anxiety and chest pain making me feel terrified and overwhelmed.",
    "Constant nervous feeling and fear of what might happen tomorrow.",
    "Scared, anxious, trembling with panic throughout the whole day.",
    "Panic attacks and uncontrollable worrying keep ruining my routine.",
    # Cluster Depressione / Tristezza
    "Depressed, feeling completely hopeless and empty inside.",
    "No motivation to get out of bed, overwhelmed by deep sadness.",
    "Feeling sorrow and darkness, everything seems meaningless.",
    "Crying all day with persistent depressive thoughts.",
    "Lost all hope and energy, sinking deeper into sadness.",
]


@pytest.fixture(scope="module")
def small_config():
    return {
        "agent4_topic_modeling": {
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "n_neighbors": 4,
            "n_components": 2,
            "min_cluster_size": 3,
        },
        "general": {"seed": 42},
    }


@pytest.fixture(scope="module")
def trained_agent(small_config):
    agent = ClinicalTopicModelingAgent(config=small_config)
    agent.fit(SAMPLE_TEXTS)
    return agent


def test_init_not_fitted(small_config):
    agent = ClinicalTopicModelingAgent(config=small_config)
    assert not agent.is_fitted
    assert agent.agent_id == "agent_4"


def test_fit_and_transform(trained_agent):
    assert trained_agent.is_fitted
    topics, probs = trained_agent.transform(SAMPLE_TEXTS[:3])
    assert len(topics) == 3
    assert len(probs) == 3


def test_get_topic_keywords(trained_agent):
    # Verifica che get_topic_keywords ritorni parole chiave valide
    keywords = trained_agent.get_topic_keywords(0, top_n=3)
    assert isinstance(keywords, list)


def test_process_payload(trained_agent):
    data = {"clean_text": "Severe insomnia and cannot fall asleep at night.", "status": "Anxiety"}
    result = trained_agent.process(data)

    assert "topic_id" in result
    assert "topic_probability" in result
    assert "topic_keywords" in result
    assert "topic_feature_vector" in result
    assert result["status"] == "Anxiety"


def test_save_and_reload(trained_agent, tmp_path):
    save_dir = tmp_path / "saved_agent4"
    trained_agent.save_pretrained(save_dir)

    assert (save_dir / "topic_modeling_config.json").exists()
    assert (save_dir / "topics_info.csv").exists()
    assert (save_dir / "bertopic_model").exists()

    reloaded = ClinicalTopicModelingAgent.from_pretrained(save_dir)
    assert reloaded.is_fitted

    sample = ["Heart racing with severe panic and nervous trembling."]
    orig_topics, _ = trained_agent.transform(sample)
    reloaded_topics, _ = reloaded.transform(sample)

    assert orig_topics == reloaded_topics
