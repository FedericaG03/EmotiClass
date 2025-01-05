import joblib

def save_model(model, vectorizer, emotion_encoder, path):
    """Save the model, vectorizer, and encoder to a specified path."""
    # Salvataggio dei file
    joblib.dump(model, f"{path}naive_bayes_model.pkl")
    joblib.dump(vectorizer, f"{path}vectorizer.pkl")
    joblib.dump(emotion_encoder, f"{path}label_encoder.pkl")

