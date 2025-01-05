#from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import sklearn.metrics as metrics
import pandas as pd

import matplotlib.pyplot as plt


def evaluate_model(model, x_test, y_test, path):
    """Evaluate the model and print the metrics."""
    y_pred = model.predict(x_test)

    print("Stampa predizione , stampa label", y_pred, y_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    '''average='macro':
     Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.'''
    precision = metrics.precision_score(y_test, y_pred, average='macro')
    recall = metrics.recall_score(y_test, y_pred, average='macro')
    f1 = metrics.f1_score(y_test, y_pred, average='macro')

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

    #confusion matrix
    confusion_matrix = metrics.confusion_matrix(y_test, y_pred)

    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix )

    cm_display.plot()
    plt.show()

    return accuracy, precision, recall, f1


