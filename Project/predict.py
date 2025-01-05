import joblib

from Project.utils.data_preprocessing import clean_text
from Project.utils.model_utils import load_model

input_path = 'models/model_new_data/'
# Carica il modello di machine learning
model, vec, enc = load_model(input_path)

#Step , prendere la frase da input
sentence = "i love the life, and i hate the people"#"Today i don't want to meet anyone" #'i am normal people'#'normal'#'Life feels balanced right now  neither good nor bad.' #'i am normal people'#'i love you, and i am happy ' #Today I am happy and it is a really nice day, I am normal'#("I feel so sad today, I don't want to meet anyone") #("I hate you")

new_sentence = clean_text(sentence)

#Trasformare la frase con vectorizer
transformed_sentence = vec.transform([new_sentence])

#Fare la previsione
y_pred = model.predict(transformed_sentence)

#decoficare l'etichetta
y_status = enc.inverse_transform(y_pred)

print("Frase:", new_sentence)
print("Etichetta predetta:", y_status[0])

