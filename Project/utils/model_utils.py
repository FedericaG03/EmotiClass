import joblib
import os

def save_model(model, vectorizer, emotion_encoder, path):
    """Save the model, vectorizer, and encoder to a specified path."""
    path = f'{path}/model/'

    parent_dir = os.path.dirname(path)
    # Creare la directory genitore
    os.makedirs(parent_dir, exist_ok=True)

    # Salvataggio dei file
    joblib.dump(model, f"{path}naive_bayes_model.pkl")
    joblib.dump(vectorizer, f"{path}vectorizer.pkl")
    joblib.dump(emotion_encoder, f"{path}label_encoder.pkl")

def load_model(input_path):

    model = joblib.load(f'{input_path}naive_bayes_model.pkl')
    # Carica il vectorizer e il label encoder
    vec = joblib.load(f'{input_path}vectorizer.pkl')
    enc = joblib.load(f'{input_path}label_encoder.pkl')

    return model, vec, enc