import pytest
from pathlib import Path

from src.agents.agent2_ner_ontology import OntologicalNERAgent
from src.utils.config_loader import load_config


@pytest.fixture(scope="module")
def ner_agent():
    config = load_config()
    return OntologicalNERAgent(config=config)


def test_slang_and_substance_masking(ner_agent):
    text = "I bought some fent and took two bars because I was restless."
    masked, entities = ner_agent.mask_and_extract_entities(text)

    # Verifica il mascheramento dei termini di strada verso le superclassi
    assert "[OPIOID]" in masked
    assert "[BENZODIAZEPINE]" in masked
    assert "fent" not in masked.lower()
    assert "bars" not in masked.lower()

    # Verifica la presenza delle entità estratte
    categories = [e["category"] for e in entities]
    assert "[OPIOID]" in categories
    assert "[BENZODIAZEPINE]" in categories


def test_dsm5_symptom_masking(ner_agent):
    text = "Having a terrible panic attack and severe insomnia every night."
    masked, entities = ner_agent.mask_and_extract_entities(text)

    assert "[DSM5_SYMPTOM]" in masked
    assert len(entities) >= 2
    for ent in entities:
        assert ent["category"] == "[DSM5_SYMPTOM]"


def test_multi_word_precedence(ner_agent):
    text = "Tried dirty sprite and later took some magic mushrooms with friends."
    masked, entities = ner_agent.mask_and_extract_entities(text)

    # Entrambe le espressioni composte devono essere riconosciute correttamente
    assert "[OPIOID]" in masked
    assert "[HALLUCINOGEN]" in masked


def test_ontology_feature_vector(ner_agent):
    data = {
        "clean_text": "Used cocaine and heroin while suffering from insomnia."
    }
    processed = ner_agent.process(data)

    assert "masked_text" in processed
    assert "ontology_feature_vector" in processed
    assert "ontology_counts" in processed

    vector = processed["ontology_feature_vector"]
    # Lunghezza coerente con le 11 categorie ontologiche
    assert len(vector) == len(ner_agent.ontology_categories)

    counts = processed["ontology_counts"]
    assert counts["[STIMULANT]"] >= 1
    assert counts["[OPIOID]"] >= 1
    assert counts["[DSM5_SYMPTOM]"] >= 1


def test_save_and_reload_agent2(ner_agent, tmp_path):
    save_dir = tmp_path / "saved_agent2"
    ner_agent.save_pretrained(save_dir)

    assert (save_dir / "ontology_ner_config.json").exists()
    assert (save_dir / "compiled_ontology_map.json").exists()

    reloaded = OntologicalNERAgent.from_pretrained(save_dir)
    assert len(reloaded.mapping) == len(ner_agent.mapping)
    assert reloaded.ontology_categories == ner_agent.ontology_categories

    sample = "Took some adderall for studying."
    orig_masked, _ = ner_agent.mask_and_extract_entities(sample)
    reloaded_masked, _ = reloaded.mask_and_extract_entities(sample)
    assert orig_masked == reloaded_masked
