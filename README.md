# ADPRS: Multi-Agent System for Psychiatric Disorder & Risk Behavior Detection

[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![HuggingFace Transformers](https://img.shields.io/badge/%F0%9F%A4%97-Transformers-orange)](https://huggingface.co/)

**ADPRS** (*Advanced Detection of Psychiatric Risks System*) è un framework multiagente (MAS) e multi-task basato sull'Elaborazione del Linguaggio Naturale (NLP) e sull'Apprendimento Profondo (Deep Learning). Il sistema è progettato per l'analisi clinico-emotiva di testi non strutturati (es. post di social media o forum di supporto) al fine di identificare la presenza di patologie psicologiche e comportamenti a rischio, quali **insonnia**, **depressione**, **ansia** e **disturbi da uso di sostanze**.

---

## 📌 1. Obiettivo del Progetto

Il progetto si propone di affrontare la complessità della diagnostica preliminare in ambito di salute mentale tramite analisi testuale. Le manifestazioni sintomatiche nei testi non strutturati presentano spesso sovrapposizioni semantiche e cliniche (es. co-morbilità tra insonnia, ansia e depressione) e l'uso di linguaggio gergale o informale (es. nomi commerciali di farmaci, slang di strada per sostanze).

**Obiettivi chiave del sistema:**
1. **Analisi Olistica Multi-Faccettata**: Integrare feature testuali, ontologiche, affettive/emotive e tematiche per supportare il processo decisionale.
2. **Classificazione Multi-Task Efficiente**: Modellare simultaneamente più condizioni cliniche per catturarne le dipendenze e ridurre il costo computazionale rispetto a modelli single-task separati.
3. **Valutazione Integrata del Rischio (Risk Scoring)**: Calcolare un indice sintetico e interpretabile di rischio clinico per facilitare operazioni di screening.

---

## 🏗️ 2. Architettura del Sistema (Multi-Agent System)

L'architettura del sistema si basa su una pipeline elaborativa sincrona e modulare composta da **4 Agenti Specializzati** e un **Agente Decisionale Multi-Task Finale**.

### Dettaglio degli Agenti:

* **Agente 1: Preprocessing & Normalizzazione**
  * Pulizia deterministica del testo (rimozione URL, emoji, ripetizioni di punteggiatura) e normalizzazione.
  * Lemmatizzazione tramite spaCy (`en_core_web_sm`), preservando le stopwords ad alta valenza psicologico-clinica (es. *triste*, *ansia*, *insonne*).
  * Tokenizzazione fissa tramite Hugging Face `AutoTokenizer` (128–512 token) per prevenire troncamenti spuri.

* **Agente 2: Entity Masking**
  * Mappatura semantica tramite **Drug Abuse Ontology (DAO)** per standardizzare slang di strada, acronimi e sinonimi (es. *fent* $\rightarrow$ *fentanyl*).
  * **Entity Masking**: Sostituzione dei termini estratti con concetti ontologici generali (es. `[OPIOID]`), riducendo la memorizzazione specifica del modello e abbassando il bias.

* **Agente 3: Profilazione Sentiment ed Emotiva**
  * **Sentiment**: Stima della polarità (Positiva, Negativa, Neutrale) ed intensità affettiva tramite [`cardiffnlp/twitter-roberta-base-sentiment-latest`](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest).
  * **Emotion Mapping**: Generazione di un vettore denso che mappi le 28 emozioni fondamentali identificate dal modello [`SamLowe/roberta-base-go_emotions`](https://huggingface.co/SamLowe/roberta-base-go_emotions).

* **Agente 4: Modellazione Tematica (Topic Agent)**
  * Estrazione degli argomenti latenti tramite **BERTopic** (Sentence-BERT + UMAP per la riduzione dimensionale + HDBSCAN per il clustering semantico).
  * Mappatura dei temi clinici tramite c-TF-IDF per distinguere discussioni generiche da manifestazioni sintomatiche (es. attacchi di panico, astinenza).

* **Agente 5: Multi-Task Classification (Decision Agent)**
  * Adotta un'architettura **Multi-Task Learning (MTL)** con un encoder Transformer condiviso [`FacebookAI/roberta-base`](https://huggingface.co/FacebookAI/roberta-base).
  * Concatena l'embedding del token `[CLS]` estratto dall'encoder con i vettori ausiliari provenienti dagli Agenti 2, 3 e 4.
  * Alimenta $k$ teste di classificazione lineari indipendenti in parallelo con attivazione *Sigmoid* per la stima delle probabilità di ciascuna condizione.

---

## 🧮 3. Calcolo dell'Indice di Rischio Integrato (SleepDepScore)

Il Decision Agent integra un modulo per il calcolo di un punteggio sintetico che quantifica l'effetto sinergico tra insonnia cronica e depressione:

$$\text{SleepDepScore}_i = \alpha \cdot P(Y_i^{(\text{insonnia})} = 1) + \beta \cdot P(Y_i^{(\text{depressione})} = 1)$$

* $P(Y_i = 1)$ rappresenta la probabilità stimata dalla rispettiva testa di classificazione dell'Agente 6.
* $\alpha$ e $\beta$ sono parametri di ponderazione (tarati di default a $0.5$).
* Una soglia critica ($\text{SleepDepScore} > 0.75$) consente l'identificazione immediata dei soggetti a prioritario rischio clinico.

---

## 📊 4. Dataset Utilizzati e Integrazione

Il sistema è stato addestrato e validato sul dataset [`ourafla/Mental-Health_Text-Classification_Dataset`](https://huggingface.co/datasets/ourafla/Mental-Health_Text-Classification_Dataset), descritto come segue sul portale Hugging Face: 

>This dataset contains short, user‑generated texts labeled for 4‑class mental health classification: Suicidal, Depression, Anxiety, and Normal. It is a derived dataset created by combining and cleaning three public mental‑health corpora, then re‑labeling them into a unified 4‑class scheme and exporting CSV files suitable for both classical ML and modern NLP models.
>
>The repository includes:
>
> 1. An unbalanced main training corpus (realistic class skew).
> 2. A strictly balanced test split for fair evaluation.
> 3. A feature‑engineered file with basic text statistics (length, URLs, emojis, punctuation, etc.).


---

## ⚙️ 5. Accortezze Progettuali e Stratagemmi di Addestramento

Durante la progettazione dell'architettura e l'addestramento dei modelli sono state adottate le seguenti soluzioni metodologiche:

1. **Entity Masking contro Overfitting Semantico**: L'uso dell'Entity Masking (Agente 2) per sostituire i nomi specifici delle sostanze o dei farmaci con tag ontologici evita che il modello memorizzi singole parole d'ordine, migliorando le prestazioni di generalizzazione.
2. **Mitigazione dello Sbilanciamento delle Classi (Weighted Loss)**: L'Agente 5 adotta una funzione di costo combinata e ponderata:
   $$\mathcal{L}_{\text{total}} = \sum_{j=1}^{k} w_j \cdot \mathcal{L}_{\text{BCE}}(y_j, \hat{y}_j)$$
   in cui ogni Binary Cross-Entropy (BCE) è pesata per bilanciare la presenza disomogenea dei campioni nei dataset clinici.
3. **Mitigazione Overfitting**: L'Agente 5 adotta i meccanismi standard di mitigazione dell'overfitting come Early Stopping e Dropout

