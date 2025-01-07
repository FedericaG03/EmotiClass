import argparse

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import os

from torch.backends.mkl import verbose

from utils.metrics import compute_metrics
from utils.model_utils import save_model
from utils.data_preprocessing import preprocessing

def train(x, y, opt):
    # Inizializza il CountVectorizer per trasformore il testo in una rappresentazione numerica
    vectorizer = CountVectorizer()

    # Il CountVectorizer richiede una lista di stringhe come input.
    # x_encoded conterrà la matrice numerica (sparse) delle parole.
    if opt.verbose:
        print(f"Shape of X before encoding: {x.shape}")

    # Applica il CountVectorizer alla colonna 'cleanText' di x, ottenendo una matrice sparse di valori numerici
    x_encoded = vectorizer.fit_transform(x['cleanText'].values.flatten())

    if opt.verbose:
        print(f"Shape of encoded matrix (x_encoded): {x_encoded.shape}")


    # Encoding dell'emozione
    emotion_encoder = LabelEncoder()
    y_encoded = emotion_encoder.fit_transform(y)

    if opt.verbose:
        # Creazione di una mappatura delle etichette
        emotion_mapping = dict(zip(emotion_encoder.classes_, range(len(emotion_encoder.classes_))))
        print("Label mapping):\n")
        for emotion, label in emotion_mapping.items():
            print(f"{emotion}: {label}")

    ## Divisione dataset in traing e test
    x_train, x_test, y_train, y_test = train_test_split(x_encoded, y_encoded, test_size=0.2, random_state=42)
    if opt.verbose:
        # Stampa della forma dei dataset di addestramento e test
        print(f"Shape of x_train: {x_train.shape}\nShape of x_test: {x_test.shape}")
        print("Dataset has been split into training and test sets.")

    ## Train
    model = MultinomialNB()
    model.fit(x_train, y_train)
    print("Model trained with Multinomial Naive Bayes")

    ## Salvataggio modello
    save_model(model, vectorizer, emotion_encoder,opt.save_path)
    print(f"Model, vectorizer, and encoder saved in the directory '{opt.save_path}'")

    return model, x_test, y_test

def classify(model, x_test, x = None, enc = None, save_path = None):
    print('Dimension len x_test:', x_test.toarray().shape)

    # Esegui la previsione sul dataset di test
    y_pred = model.predict(x_test)

    if save_path:
        # Decodifica le etichette previste
        status = enc.inverse_transform([y_pred])
        predictions_df = pd.DataFrame({
            'statement': x,
            'status': status
        })
        file_path = save_path + "predictions.csv"
        predictions_df.to_csv(file_path, index=False)

    return y_pred

def parse_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", type=str, default=r"..\Project\data\emotionSentimentAnalysis.csv", help="path to the dataset")
    parser.add_argument("--save_path", type=str, default=r"experiment_1/", help="path where to save model and results")
    parser.add_argument("--verbose", action='store_true', help="verbose mode")
    parser.add_argument("--stopwords_flag", action='store_true', help="If use stopwords removal")
    parser.add_argument("--n_samples", type=int, default=3000, help="Number of samples for downsampling")

    return parser.parse_args()

def main(opt):

    # Creare la directory la directory del percorso dove salvare il modello
    parent_dir = os.path.dirname(opt.save_path)
    # Crea la directory genitore se non esiste
    os.makedirs(parent_dir, exist_ok=True)

    print(f'The model will be trained on {opt.data}')
    ## Caricamento del dataset
    df = pd.read_csv(opt.data)

    # Pre-elaborazione del dataset
    df_preprocessed = preprocessing(df, opt.verbose, opt.n_samples, opt.stopwords_flag, opt.save_path)

    x, y = df_preprocessed[['cleanText']], df_preprocessed['status']

    # Addestramento del modello
    model, x_test, y_test = train(x, y, opt)

    # Previsione sul set di test
    y_pred = classify(model, x_test)

    #Valutazione del modello
    compute_metrics(y_pred, y_test, opt.save_path)

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)