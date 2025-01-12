# EmotiClass
Questo repository contiene uno strumento per l'analisi del sentiment e la classificazione delle emozioni basata su tecniche di pre-elaborazione del testo e modelli di Machine Learning.

## Struttura del Progetto

Il progetto è suddiviso in più moduli per gestire la pipeline, che include pre-elaborazione del testo, addestramento e valutazione del modello, e predizioni. Ecco i file principali inclusi:

- `metrics.py`: Funzioni per il calcolo e la visualizzazione delle metriche di performance del modello.
- `data_preprocessing.py`: Moduli per la pulizia del testo, gestione dei valori nulli e bilanciamento del dataset (sottocampionamento).
- `train_and_evaluate.py`: Contiene funzioni per l'addestramento del modello e la classificazione.
- `model_utils.py`: Funzioni per salvare e caricare il modello, il vectorizer e il label encoder.
- `predict.py`: Script principale per classificare frasi singole o dataset e per valutare il modello con etichette conosciute.

## Requisiti

- **Python 3.7+**
- Librerie principali:
  - `sklearn`
  - `pandas`
  - `matplotlib`
  - `nltk`
  - `joblib`

Installa tutte le dipendenze richieste usando:

```bash
pip install -r requirements.txt
```

## Esecuzione

### Pre-elaborazione e Addestramento

Per addestrare il modello e pre-elaborare i dati:

```bash
python train_and_evaluate.py --data <percorso_dataset> --save_path <cartella_output> [--verbose] [--stopwords_flag] [--n_samples <int>] [--vectorizer tfidf_vec|count_vec]
```

Opzioni principali:
- `--data`: Percorso al file CSV contenente il dataset.
- `--save_path`: Cartella per salvare i modelli e i risultati.
- `--stopwords_flag`: Opzionale, rimuove le stopwords durante la pulizia del testo.
- `--n_samples`: Opzionale, specifica il numero massimo di campioni per classe (sottocampionamento).
- `--vectorizer`: Specifica il tipo di vectorizer (`count_vec` o `tfidf_vec`).

### Predizione e Valutazione

Per eseguire predizioni su una frase o dataset esistente, usa `predict.py`:

```bash
python predict.py --model_path <cartella_modello> --sentence "<frase>" [--demo] [--verbose]
```

Opzioni principali:
- `--model_path`: Cartella dove sono salvati il modello, il vectorizer e l'encoder.
- `--sentence`: Una singola frase da classificare.
- `--data_path`: Predici le etichette di un intero dataset.
- `--evaluate_mode`: Valuta le predizioni rispetto alle etichette originali.

### Output

- **Metriche di performance**: Precision, Recall, Accuracy e F1 Score salvati in un file `metrics.txt`.
- **Visualizzazioni**:
  - Grafico della distribuzione delle previsioni (corrette/errate).
  - Matrice di confusione.
  - Distribuzione delle emozioni nel dataset pre-elaborato.

## Dataset di Esempio

Puoi utilizzare il tuo dataset in formato CSV con almeno le seguenti colonne:
- `statement`: Il testo da analizzare.
- `status`: L'etichetta corrispondente all'emozione (per addestramento e valutazione).
