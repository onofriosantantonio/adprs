#!/usr/bin/env python3
"""
Script di utilità per caricare i pesi e gli artefatti dell'Agente 5 Decisionale
su Hugging Face Hub (Model Repository).
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from huggingface_hub import HfApi, login
from src.utils.logger import get_logger

logger = get_logger("UploadToHF")


def upload_model(
    model_dir: str = "models/decision_multitask",
    repo_id: str = "",
    token: str = None,
    private: bool = False,
):
    path = Path(model_dir)
    if not path.exists():
        logger.error(f"Cartella modello non trovata: {path}")
        sys.exit(1)

    if not repo_id:
        logger.error("Specificare il repo_id (es. 'username/adprs-decision-multitask') tramite --repo-id")
        sys.exit(1)

    if token:
        login(token=token)

    api = HfApi()
    logger.info(f"Creazione/Verifica repository su Hugging Face: {repo_id}...")
    api.create_repo(repo_id=repo_id, exist_ok=True, private=private, repo_type="model")

    logger.info(f"Caricamento contenuti da {path} a {repo_id}...")
    api.upload_folder(
        folder_path=str(path),
        repo_id=repo_id,
        repo_type="model",
        commit_message="Upload MultiTaskDecisionAgent trained weights and configs",
    )
    logger.info(f"Modello caricato con successo! Disponibile su: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Carica modello su Hugging Face Hub.")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models/decision_multitask",
        help="Percorso cartella locale contenente pesi e config",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Identificativo repository HF (es. 'username/adprs-decision-multitask')",
    )
    parser.add_argument("--token", type=str, default=None, help="Token HF di scrittura (opzionale se già loggato)")
    parser.add_argument("--private", action="store_true", help="Crea repository come privato")
    args = parser.parse_args()

    upload_model(
        model_dir=args.model_dir,
        repo_id=args.repo_id,
        token=args.token,
        private=args.private,
    )
