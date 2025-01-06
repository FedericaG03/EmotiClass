import argparse
from idlelib.browser import file_open

import pandas as pd

from train_and_evaluate import classify
from utils.data_preprocessing import clean_text, preprocessing
from utils.model_utils import load_model

def predict_sentence(model, vec, enc, sentence = '', demo_mode = False):

    #Verificare se la frase è stata inserita o no
    if demo_mode:
        sentence = input("Tell me something and I will classify your emotion.\nYour sentence:")

    new_sentence = clean_text(sentence)
    #Trasformare la frase con vectorizer
    transformed_sentence = vec.transform([new_sentence])

    #Fare la previsione
    y_pred = model.predict(transformed_sentence)

    #decoficare l'etichetta
    y_status = enc.inverse_transform(y_pred)

    print("Sentence:", new_sentence)
    print("Status predict:", y_status[0])

def parse_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, default=r"experiments\exp_emotion_data_model_nayveMNB_test20%_n_samples3000\model/", help="path to load the model")
    parser.add_argument("-sentence", type=str,  help="The sentence to classify")
    parser.add_argument('-data_path', type=str, help='path to data')
    parser.add_argument("-demo", action='store_true', help="Demo mode")
    parser.add_argument("--evaluate_mode", action='store_true', help="If labels are provided, evaluate model performance")

    return parser.parse_args()

def main(opt):
    print(f'Loading the model: {opt.model_path}')

    model, vec, enc = load_model(opt.model_path)

    if opt.sentence or opt.demo:
        predict_sentence(model, vec, enc, opt.sentence, opt.demo)
    elif opt.data_path:
        ## process input
        df = pd.read_csv(opt.data_path)
        df_preprocessed = preprocessing(df, opt.verbose, opt.n_samples, opt.stopwords_flag, opt.save_path)

        x = df_preprocessed[['cleanText']]
        x_enc = vec.fit_transform(x['cleanText'].values.flatten())

        y_pred = classify(model, x_enc, x, enc, opt.save_path)

        if opt.evaluate_mode:
            ## open file and compute metrics
            file_path = opt.save_path + 'predictions.csv'

            model_predictions = pd.read_csv(file_path)

            statement = model_predictions['statement']
            status = model_predictions['status']

            labels = [0, 2, 3 , 1 ,4 ,5]


    else:
        print('No data is provided')

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)

