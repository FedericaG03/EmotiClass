import argparse

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import os

from utils.metrics import compute_metrics
from utils.model_utils import save_model
from utils.data_preprocessing import preprocessing

def train(x, y, opt):
    ## Load/initialize model
    # Trasforma il testo in numeri con CountVectorizer
    vectorizer = CountVectorizer()
    ''' Vectorizer, vuole una lista di stringhe devi gestirlo
    x_encoded = vectorizer.fit_transform(x)  # Ottieni la matrice numerica (sparse)'''
    x_encoded = vectorizer.fit_transform(x['cleanText'].values.flatten())  # Usa l'array di stringhe
    print(f"Cosa fa vectorizer {x_encoded}")

    # Encoding dell'emozione (se status non è numerico)
    emotion_encoder = LabelEncoder()
    y_encoded = emotion_encoder.fit_transform(y)
    print("Mapping labels:", dict(zip(emotion_encoder.classes_, range(len(emotion_encoder.classes_)))))

    ## Set training parameters
    x_train, x_test, y_train, y_test = train_test_split(x_encoded, y_encoded, test_size=0.2, random_state=42)
    print("Dataset suddiviso in train e test.")

    ## Train
    model = MultinomialNB()
    model.fit(x_train, y_train)
    print("Modello addestrato con Multinomial Naive Bayes.")

    ## Save model
    save_model(model, vectorizer, emotion_encoder,opt.save_path)
    print(f"Modello, vectorizer ed encoder salvati nella directory '{opt.save_path}'")

    return model, x_test, y_test

def classify(model, x_test, x = None, enc = None, save_path = None):

    y_pred = model.predict(x_test)

    if save_path:
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
    parent_dir = os.path.dirname(opt.save_path)
    # Creare la directory genitore
    os.makedirs(parent_dir, exist_ok=True)

    print(f'The model will be trained on {opt.data}')
    ## Load Dataset
    df = pd.read_csv(opt.data)

    ## Preprocess Dataset
    df_preprocessed = preprocessing(df, opt.verbose, opt.n_samples, opt.stopwords_flag, opt.save_path)

    x, y = df_preprocessed[['cleanText']], df_preprocessed['status']

    #Training
    model, x_test, y_test = train(x, y, opt)
    #Prediction on test set
    y_pred = classify(model, x_test)
    #Model evaluation
    compute_metrics(y_pred, y_test, opt.save_path)

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)