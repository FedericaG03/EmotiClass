import pandas as pd
import nltk
import re
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
nltk.download('stopwords', quiet=True)

def normalizing(df, n_samples):
    print(df.head())
    print(f'Distribution of the classes before normalization: {df['status'].value_counts()}')

    # Analysing elements' frequency in order to balance the dataset
    status_counts = df['status'].value_counts()
    most_frequent = status_counts.idxmax()
    least_frequent = status_counts.idxmin()

    print(f"Most frequent element: {most_frequent}")
    print(f"Least frequent element: {least_frequent}")

    # Sottocampionamento
    balanced_dataset = df.groupby('status').apply(lambda x: x.sample(n_samples)).reset_index(drop=True)

    print("Balanced distribution:")
    print(balanced_dataset['status'].value_counts())

    return balanced_dataset

def clean_text(statement):
    statement = statement.lower()
    stop_words = set(stopwords.words('english'))
    statement = ' '.join([word for word in statement.split() if word not in stop_words])
    statement = re.sub(r'\s+', ' ', statement).strip()  # Gestisce spazi multipli
    statement = ' '.join(dict.fromkeys(statement.split()))  # Rimuove duplicati
    statement = re.sub(r'[^a-zA-Z0-9\s]', '', statement) #Rimuovere caratteri speciali

    return statement

def preprocess_data(df):
    # Create the 'cleanText' colun
    df['cleanText'] = df['statement'].apply(clean_text)
    return df

def remove_value_null(df, verbose):
    if verbose:
        print('Valori mancanti in "statement":', df['statement'].isnull().sum())
        missing_by_status = df.groupby('status')['statement'].apply(lambda x: x.isnull().sum())
        print("Valori mancanti per classe 'status':", missing_by_status)

        # Grafico
        plt.figure(figsize=(8, 5))
        missing_by_status.plot(kind='bar', color='orange')
        plt.title("Valori Mancanti nella colonna 'statement' per Status")
        plt.xlabel("Status")
        plt.ylabel("Numero di Valori Mancanti")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()

    # Rimozione valori nulli
    df_cleaned = df.dropna(subset=['statement'])  # Rimuove righe con 'statement' NaN

    return df_cleaned

def preprocessing(df, verbose, n_samples):
    if verbose:
        print("Info Dataset:")
        print(df.info())

    print("Remove null values..")
    df = remove_value_null(df,verbose)
    print("Cleaning...")
    df = preprocess_data(df)
    print("Normalizing...")
    df = normalizing(df, n_samples)

    return  df

def information(df):
    print("Dataset dimensions:", df.shape)
    frequencies = df['status'].value_counts()
    print("Class frequencies:", frequencies)

    # Plot
    plt.figure(figsize=(8, 5))
    frequencies.plot(kind='bar', color='orange')
    plt.title("Class Frequencies")
    plt.xlabel("Class")
    plt.ylabel("Number of Occurrences")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()