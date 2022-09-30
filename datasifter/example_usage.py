import os.path
import pickle

import pandas as pd

import datasifter

if __name__ == '__main__':
    # preprocessing
    raw_data_path = "stay_length_4000_with_notes.csv"
    raw = pd.read_csv(raw_data_path)['TEXT']
    notes_list, tokenized_notes = datasifter.clean_data(raw)
    # identify sensitive outcomes
    outcomes = ['length_of_stay_avg', 'Gender', 'Religion']
    print(outcomes)
    if os.path.exists("keywords.pkl"):
        print("open pkl")
        with open('keywords.pkl', 'rb') as f:
            keywords = pickle.load(f)
    else:
        keywords, outcomes = datasifter.get_outcomes(notes_list, raw_data_path, outcomes, dump_to='keywords.pkl')

    # length_of_stay_avg
    LOSA_keywords = keywords[outcomes[0]]
    # Gender
    Gender_keywords = keywords[outcomes[1]]
    # Religion
    Religion_keywords = keywords[outcomes[2]]

    LOSA_list = list(LOSA_keywords.Feature)
    Gender_list = list(Gender_keywords.Feature)
    Religion_list = list(Religion_keywords.Feature)

    keywords_dict = {}
    keywords_dict.setdefault(outcomes[0], LOSA_list)
    keywords_dict.setdefault(outcomes[1], Gender_list)
    keywords_dict.setdefault(outcomes[2], Religion_list)

    w2v_model = datasifter.W2VGN()
    words_replacement = w2v_model.generate_words_for_replacement(keywords_dict)

    final_result_small = datasifter.obfuscate(raw, words_replacement, 'small', outcomes[1])
