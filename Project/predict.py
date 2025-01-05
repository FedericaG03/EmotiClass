import argparse

import pandas as pd
from nltk.parse.corenlp import transform

from train_and_evaluate import evaluate
from utils.data_preprocessing import clean_text, preprocessing
from utils.model_utils import load_model

def predict_sentence(model, vec, enc, sentence):

    #Step , prendere la frase da input
    #sentence = "i love the life, and i hate the people"#"Today i don't want to meet anyone" #'i am normal people'#'normal'#'Life feels balanced right now  neither good nor bad.' #'i am normal people'#'i love you, and i am happy ' #Today I am happy and it is a really nice day, I am normal'#("I feel so sad today, I don't want to meet anyone") #("I hate you")

    new_sentence = clean_text(sentence)
    #Trasformare la frase con vectorizer
    transformed_sentence = vec.transform([new_sentence])

    #Fare la previsione
    y_pred = model.predict(transformed_sentence)

    #decoficare l'etichetta
    y_status = enc.inverse_transform(y_pred)

    print("Frase:", new_sentence)
    print("Etichetta predetta:", y_status[0])

def parse_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, default=r"experiments\exp_emotion_data_model_nayveMNB_test20%_n_samples3000\model/", help="path to load the model")
    parser.add_argument("-sentence", type=str,  help="The sentence to classify")
    parser.add_argument('-data_path', type=str, help='path to data')

    return parser.parse_args()

def main(opt):
    print(f'Loading the model: {opt.model_path}')

    model, vec, enc = load_model(opt.model_path)

    if opt.sentence:
        predict_sentence(model, vec, enc, opt.sentence)
    elif opt.data_path:
        ## process input
        df = pd.read_csv(opt.data_path)
        df_preprocessed = preprocessing(df, opt.verbose, opt.n_samples, opt.stopwords_flag, opt.save_path)

        x, y = df_preprocessed[['cleanText']], df_preprocessed['status']
        x_test = vec.fit_transform(x['cleanText'].values.flatten())
        y_test = enc.fit_transform([y])

        evaluate(model, x_test, y_test, opt.save_path)
    else:
        print('No data is provided')

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)

