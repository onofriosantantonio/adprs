#!/usr/bin/env bash
# ==============================================================================
# Script di configurazione automatica dell'ambiente virtuale
# Compatibile con macOS (Apple Silicon/Intel) e Linux
# ==============================================================================

set -e

echo "=== [1/4] Verifica interprete Python ==="
PYTHON_CMD=""

if command -v python3.11 &>/dev/null; then
    PYTHON_CMD="python3.11"
elif [ -f "/Library/Frameworks/Python.framework/Versions/3.11/bin/python3" ]; then
    PYTHON_CMD="/Library/Frameworks/Python.framework/Versions/3.11/bin/python3"
elif command -v python3 &>/dev/null; then
    PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    echo "Rilevata versione Python: $PY_VER"
    PYTHON_CMD="python3"
else
    echo "Errore: Python 3 non trovato. Installa Python 3.10 o 3.11."
    exit 1
fi

echo "Utilizzo interprete: $PYTHON_CMD ($($PYTHON_CMD --version))"

echo "=== [2/4] Creazione Virtual Environment (se non presente) ==="
if [ ! -d "venv" ]; then
    $PYTHON_CMD -m venv venv
    echo "Virtual environment creato in ./venv"
else
    echo "Virtual environment già esistente in ./venv"
fi

echo "=== [3/4] Aggiornamento pip e installazione dipendenze ==="
./venv/bin/pip install --upgrade pip setuptools wheel
./venv/bin/pip install -r requirements.txt

echo "=== [4/4] Download modello spaCy inglese (en_core_web_sm) ==="
./venv/bin/python -m spacy download en_core_web_sm || echo "Attenzione: download en_core_web_sm fallito o da rieseguire."

echo "=============================================================================="
echo " Setup completato con successo!"
echo " Per attivare l'ambiente virtuale esegui:"
echo "   source venv/bin/activate"
echo "=============================================================================="
