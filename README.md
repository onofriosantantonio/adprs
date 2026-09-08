# Sistema Multiagente per l'Identificazione di Patologie e Comportamenti a Rischio

Pipeline neurale ed elaborativa per l'analisi clinico-emotiva e multi-task di dati testuali non strutturati (post social media, forum di supporto, cartelle/note cliniche).

Basato sulle specifiche tecniche definite in [schema-architettura-multiagente.pdf](schema-architettura-multiagente.pdf).

---

## 🏛️ Architettura del Sistema

Il sistema orchestra una serie di agenti sincroni e specializzati la cui inferenza converge verso un decisore Multi-Task Learning (MTL):

```
[Testo Grezzo]
      │
      ▼
[Agente 1: Preprocessing & Normalizzazione] (spaCy, tokenizzazione transformer, preservazione lemmi clinici)
      │
      ▼
[Agente 2: NER Ontologico & Entity Masking] (Estrazione sintomi/sostanze, DAO mapping es. 'fent' -> '[OPIOID]')
      ├───► [Agente 3: Sentiment & 7 Emozioni] (Wang et al. 7 classi emotive + polarità)
      ├───► [Agente 4: Topic Modeling] (BERTopic, Sentence-BERT, UMAP, HDBSCAN)
      └───► [Agente 6: Multi-Task Classification] (RoBERTa/ClinicalBERT + Teste lineari Sigmoid)
                  │
                  ▼
      [Calcolo Indice di Rischio Integrato] (SleepDepScore & Matrice Triaging)
```

### Dettaglio Agenti
1. **Agente 1 (Preprocessing)**: Pulizia deterministica (rimozione rumore, URL, emoji) preservando le parole sentinella psicologiche/cliniche.
2. **Agente 2 (NER Ontologico & Masking)**: Riconosce sostanze/farmaci e sostituisce i termini colloquiali/gergali con superclassi standardizzate della **Drug Abuse Ontology (DAO)**, riducendo il bias semantico.
3. **Agente 3 (Sentiment & Emozioni)**: Calcola la polarità e genera un vettore denso a 7 dimensioni (*Joy, Sadness, Anger, Fear, Love, Surprise, Gratitude*).
4. **Agente 4 (Topic Modeling)**: Estrae i topic clinici latenti con approccio c-TF-IDF basato su BERTopic.
5. **Agente 6 (Decision Multi-Task)**: Unisce l'embedding di contesto `[CLS]` con i vettori estratti dagli altri agenti per stimare indipendentemente le probabilità di *Insonnia, Depressione, Ansia e Abuso di Sostanze*, calcolando lo **SleepDepScore**.

---

## 🚀 Replicabilità e Setup Rapido

Il progetto è progettato per essere replicabile in modo identico su macOS (con accelerazione Apple Silicon `mps`), Linux (con CUDA) e Google Colab / server remoti.

### 1. Prerequisiti
- Python 3.10 o 3.11 (consigliato 3.11 per compatibilità completa con PyTorch, spaCy e BERTopic).

### 2. Installazione Automatica
Esegui lo script di configurazione:
```bash
./setup_env.sh
```

Oppure manualmente:
```bash
# Crea il virtual environment
python3.11 -m venv venv

# Attiva l'ambiente
source venv/bin/activate

# Aggiorna pip e installa le dipendenze
pip install --upgrade pip
pip install -r requirements.txt

# Scarica i modelli linguistici spaCy
python -m spacy download it_core_news_sm
python -m spacy download en_core_web_sm
```

### 3. Attivazione dell'Ambiente
```bash
source venv/bin/activate
```

---

## 📁 Struttura della Repository

```
.
├── schema-architettura-multiagente.pdf  # Specifica tecnica originale
├── requirements.txt                    # Dipendenze con versioni compatibili
├── setup_env.sh                        # Script di bootstrap ambiente
├── configs/
│   └── config.yaml                     # Parametri ipermetrici e modelli
├── data/
│   ├── raw/                            # Dataset grezzi (es. SMHD, SWMH, Mental-Health 4-Class)
│   ├── processed/                      # Dati pre-elaborati e cache
│   └── ontology/
│       └── dao_substances_map.json     # Mappature Drug Abuse Ontology (DAO)
├── src/
│   ├── agents/                         # Implementazione Agenti 1, 2, 3, 4, 6
│   ├── models/                         # Architettura neurale Multi-Task & Risk Scorer
│   ├── pipeline/                       # Orchestratore sincrono del MAS
│   └── utils/                          # Logger, caricamento config e helper
└── tests/                              # Unit test e verifiche dei singoli agenti
```

---

## 📊 Dataset Open-Source di Riferimento

- **Mental-Health 4-Class** (`ourafla/mental-health-4-class` su Hugging Face Hub): ideale per baseline MVP rapida.
- **SMHD** (Self-reported Mental Health Diagnoses): post da Reddit per depressione e controllo.
- **PsySym**: 38 classi di sintomi DSM-5 per training fine dell'Agente 2 e classificazione.
- **SWMH**: SuicideWatch & Mental Health collection per robustezza multi-task.
