import sklearn.metrics as metrics
import pandas as pd
import matplotlib.pyplot as plt

def compute_metrics(y_pred, y_test, path):
    """Evaluate the model and print the metrics."""

    predictions_df = pd.DataFrame({
        'True_Label': y_test,
        'Predicted_Label': y_pred
    })

    accuracy = metrics.accuracy_score(y_test, y_pred)
    '''average='macro':
     Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.'''
    precision = metrics.precision_score(y_test, y_pred, average='macro')
    recall = metrics.recall_score(y_test, y_pred, average='macro')
    f1 = metrics.f1_score(y_test, y_pred, average='macro')

    file = path + "metrics.txt"
    #salvare le metriche su file
    with open(file, 'w') as f:
        f.write("Accuracy: {:.4f}\n".format(accuracy))
        f.write("Precision: {:.4f}\n".format(precision))
        f.write("Recall: {:.4f}\n".format(recall))
        f.write("F1 Score: {:.4f}\n".format(f1))

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")

    # Conta previsioni corrette ed errate
    predictions_df['Correct'] = predictions_df['True_Label'] == predictions_df['Predicted_Label']
    counts = predictions_df['Correct'].value_counts()

    # Grafico a barre
    plt.bar(['Previsioni Corrette', 'Previsioni Errate'], counts, color=['green', 'red'])
    plt.title("Distribuzione Previsioni")
    plt.ylabel("Numero di Previsioni")
    plt.savefig(path + "Correct preds vs Wrong preds.png")
    plt.show()

    #confusion matrix
    confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix)
    #cm_display.plot(cmap = plt.cm.magma)
    cm_display.plot(cmap=plt.cm.Blues)
    plt.savefig(path + "ConfusionMatrix.png")
    plt.show()


    return accuracy, precision, recall, f1


