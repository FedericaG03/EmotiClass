import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def evaluate_model(model, x_test, y_test):
    """Evaluate the model and print the metrics."""

    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)


    predictions_df = pd.DataFrame({
        'True Label' : y_test,# Etichette reali (opzionale)
        'Predicted_Label': y_pred
    })
    predictions_df.to_csv('predizione.csv', index=False)

    #Grafico
    predictions_df = pd.DataFrame({
        'True Label': y_test,
        'Predicted_Label': y_pred
    })

    # Conta previsioni corrette ed errate
    predictions_df['Correct'] = predictions_df['True Label'] == predictions_df['Predicted_Label']
    counts = predictions_df['Correct'].value_counts()

    # Grafico a barre
    plt.bar(['Previsioni Corrette', 'Previsioni Errate'], counts, color=['green', 'red'])
    plt.title("Distribuzione Previsioni")
    plt.ylabel("Numero di Previsioni")
    plt.show()

    return accuracy, precision, recall, f1

