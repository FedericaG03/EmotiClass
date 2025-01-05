from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

import matplotlib.pyplot as plt


def evaluate_model(model, x_test, y_test, path):
    """Evaluate the model and print the metrics."""
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    '''average='micro':Viene calcolata la metrica complessiva a livello globale considerando ogni previsione e ogni vero esempio, quindi sommando i veri positivi, i falsi positivi e i falsi negativi per tutti gli esempi. È una media "globale", che tratta tutte le classi come se fossero equivalenti.
    Questo è utile quando hai un numero di esempi significativamente diverso per ciascuna classe (es. un dataset sbilanciato) e vuoi una misura di performance che tenga conto di tutte le previsi'''
    precision = precision_score(y_test, y_pred, average='micro')
    recall = recall_score(y_test, y_pred, average='micro')
    f1 = f1_score(y_test, y_pred, average='micro')

    file = path + "metriche.txt"
    #salvare le metriche su file
    with open(file, 'w') as f:
        f.write("Accuracy del modello: {:.4f}\n".format(accuracy))
        f.write("Precision: {:.4f}\n".format(precision))
        f.write("Recall: {:.4f}\n".format(recall))
        f.write("F1 Score: {:.4f}\n".format(f1))

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


