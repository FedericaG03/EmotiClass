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
    #balanced_dataset = df.groupby('status').apply(lambda x: x.sample(n_samples)).reset_index(drop=True)
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

'''
def preprocess_data(df):
    # Create the 'cleanText' colun
    df['cleanText'] = df['statement'].apply(clean_text)
    return df
'''

def remove_value_null(df, verbose, path):
    if verbose:
        print('Valori mancanti in "statement":', df['statement'].isnull().sum())
        missing_by_status = df.groupby('status')['statement'].apply(lambda x: x.isnull().sum())
        print("Valori mancanti per classe 'status':", missing_by_status)

        # Grafico
        plt.figure(figsize=(8, 5))
        missing_by_status.plot(kind='bar', color='blue')
        plt.title("Valori Mancanti nella colonna 'statement' per Status")
        plt.xlabel("Status")
        plt.ylabel("Numero di Valori Mancanti")
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig(path + "Valori nulli.png")
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
        # Funzione "wrapper" che prende il testo e il flag come argomenti
        def clean_text_with_flag(text):
            return clean_text(text, stopwords_flag)  # Passa stopwords_flag come True o False

        # Ora puoi usare clean_text_with_flag con .apply()
        df['cleanText'] = df['statement'].apply(clean_text_with_flag)
    else:
        df['cleanText'] = df['statement'].apply(clean_text)

    print("Normalizing...")
    df = normalizing(df, n_samples)

    return  df

def information(df, path):
    print("Dataset dimensions:", df.shape)
    frequencies = df['status'].value_counts()
    print("Class frequencies:", frequencies)

    # Plot
    plt.figure(figsize=(8, 8))
    frequencies.plot(kind='bar', color='blue')
    plt.title("Frequenza delle emozioni divisa per status")
    plt.xlabel("Status")
    plt.ylabel("Numero di Frequenza divisa")
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(path + "Frequenza delle emozioni divisa per classi.png")
    plt.show()