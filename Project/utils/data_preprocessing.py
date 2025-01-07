from tabnanny import verbose

import nltk
import re
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
nltk.download('stopwords', quiet=True)

def under_sampling(df, n_samples):
    print(df.head())
    print(f'Distribution of the classes before normalization: {df['status'].value_counts()}')

    # Analizzare la frequenza degli elementi per bilanciare il set di dati
    status_counts = df['status'].value_counts()

    if verbose:
        most_frequent = status_counts.idxmax()
        least_frequent = status_counts.idxmin()

        print(f"Most frequent element: {most_frequent}")
        print(f"Least frequent element: {least_frequent}")

    #Sottocampionamento controllato
    balanced_dataset = df.groupby('status').apply(
        lambda x: x.sample(n_samples) if len(x) >= n_samples else x
    ).reset_index(drop=True)

    print("Balanced distribution:")
    print(balanced_dataset['status'].value_counts())

    return balanced_dataset

def clean_text(statement, stopwords_flag = False):
    statement = statement.lower()
    if stopwords_flag:
        stop_words = set(stopwords.words('english'))
        statement = ' '.join([word for word in statement.split() if word not in stop_words])
    statement = re.sub(r'\s+', ' ', statement).strip()  # Gestisce spazi multipli
    statement = ' '.join(dict.fromkeys(statement.split()))  # Rimuove duplicati
    statement = re.sub(r'[^a-zA-Z0-9\s]', '', statement) #Rimuovere caratteri speciali

    return statement

def remove_value_null(df, verbose, path):
    if verbose:
        print('Missing value in "statement":', df['statement'].isnull().sum())
        missing_by_status = df.groupby('status')['statement'].apply(lambda x: x.isnull().sum())
        print("Missing value for 'status':", missing_by_status)

        # Creazione del Grafico
        plt.figure(figsize=(8, 5))
        missing_by_status.plot(kind='bar', color='blue')
        plt.title("Distribuzione dei Valori Mancanti nella Colonna 'statement' in base allo Status")
        plt.xlabel("Emozione")
        plt.ylabel("Conteggio dei Valori Mancanti")
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(path + "valori_mancanti_statement_per_status.png")
        plt.show()

    # Rimozione valori nulli
    df_cleaned = df.dropna(subset=['statement'])
    return df_cleaned

def preprocessing(df, verbose, n_samples, stopwords_flag, path):
    if verbose:
        print("Info Dataset:")
        information(df, path)

        print("Cleaning...")
    df = remove_value_null(df,verbose, path)
    if stopwords_flag:
        # Definizione di una funzione wrapper per integrare il flag nell'elaborazione del testo
        def clean_text_with_flag(text):
            return clean_text(text, stopwords_flag)  # Passa stopwords_flag come True

        # Utilizzo della funzione wrapper per applicare la pulizia del testo alla colonna 'statement'
        df['cleanText'] = df['statement'].apply(clean_text_with_flag)
    else:
        # Se il flag non è impostato, applica direttamente la funzione clean_text senza flag
        # È utile per i casi in cui clean_text può gestire il default (senza stopwords)
        df['cleanText'] = df['statement'].apply(clean_text)

    if n_samples:   ##Bilancia il dataset se viene fornito il numero di campioni
        print("Under_sampling...")
        df = under_sampling(df, n_samples)

    return  df

def information(df, path):
    print("Dataset dimensions:", df.shape)
    frequencies = df['status'].value_counts()
    print("Class frequencies:", frequencies)

    # Plot
    plt.figure(figsize=(8, 8))
    frequencies.plot(kind='bar', color='blue')
    plt.title("Distribuzione delle emozioni in base allo Status")
    plt.xlabel("Status")
    plt.ylabel("Frequenza delle emozioni")
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(path + "frequenza_emozioni_per_status.png")
    plt.show()