import pytest
import shutil
from pathlib import Path

from src.agents.agent1_preprocessor import PreprocessingAgent
from src.utils.config_loader import load_config

@pytest.fixture(scope="module")
def preprocessor():
    config = load_config()
    return PreprocessingAgent(config=config)

def test_clean_noise(preprocessor):
    raw_text = "Check this out https://reddit.com/r/anxiety!! I feel sooo hopeless 😭 [deleted]"
    clean = preprocessor.clean_text(raw_text)
    assert "https://" not in clean
    assert "r/anxiety" not in clean
    assert "[deleted]" not in clean
    assert "😭" not in clean
    assert "hopeless" in clean

def test_clinical_keywords_preservation(preprocessor):
    text = "I am experiencing severe anxiety, insomnia, and feel completely depressed."
    res = preprocessor.lemmatize_and_filter(text)
    lemmas = res["lemmatized_text"]

    # Verifica che le parole sentinella cliniche siano preservate
    assert "anxiety" in lemmas
    assert "insomnia" in lemmas
    assert "depressed" in lemmas or "depress" in lemmas

def test_tokenize_transformer(preprocessor):
    text = "Severe depression and sleepless nights."
    encoded = preprocessor.tokenize_transformer(text)
    assert "input_ids" in encoded
    assert "attention_mask" in encoded
    assert encoded["input_ids"].shape[0] == 1
    assert encoded["input_ids"].shape[1] <= preprocessor.max_seq_length

def test_agent_process_pipeline(preprocessor):
    sample = {"text": "I can't take this anymore, panic attacks every night... https://help.org"}
    output = preprocessor.process(sample)

    assert "clean_text" in output
    assert "lemmatized_text" in output
    assert "tokens_info" in output
    assert "input_ids" in output
    assert "attention_mask" in output
    assert "can't" in output["clean_text"]
    assert "https://" not in output["clean_text"]

def test_save_and_reload_pretrained(preprocessor, tmp_path):
    save_dir = tmp_path / "test_saved_preprocessor"
    preprocessor.save_pretrained(save_dir)

    assert (save_dir / "preprocessor_config.json").exists()
    assert (save_dir / "tokenizer_config.json").exists()

    # Ricarica dall'artefatto salvato
    reloaded = PreprocessingAgent.from_pretrained(save_dir)
    assert reloaded.max_seq_length == preprocessor.max_seq_length
    assert reloaded.clinical_preserved_keywords == preprocessor.clinical_preserved_keywords

    test_sentence = "I'm having insomnia."
    out_orig = preprocessor.clean_text(test_sentence)
    out_reloaded = reloaded.clean_text(test_sentence)
    assert out_orig == out_reloaded
