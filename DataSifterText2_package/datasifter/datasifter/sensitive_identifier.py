import pandas as pd
import numpy as np
import lightgbm as lgb
from matplotlib import pyplot as plt
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from .functions import get_features, generate_model, text_clean, split_class
import pickle


def get_outcomes(notes_list, raw_data_path='stay_length_4000_with_notes.csv',
                 outcomes=None, dump_to=None):
    if outcomes is None:
        outcomes = ['length_of_stay_avg', 'Gender', 'Religion']
    consider_text = True  # consider text OR features
    num_keywords = 10  # num keywords generated
    drop_list = [0, 2, 4, 5, 6]  # useless df features
    num_class = 5  # for continuous outcomes
    control = {'remove_number': True, 'lemmatize': True, 'normalize': True, 'stop_words': True, 'gram': 1,
               'remove_name': True}
    # read data frame
    df = pd.read_csv(raw_data_path)
    df = df.drop(df.columns[drop_list], axis=1)
    # get all categorical columns
    cat_columns = df.select_dtypes(['object']).columns
    # convert all categorical columns to numeric
    df[cat_columns] = df[cat_columns].apply(lambda x: pd.factorize(x)[0])
    df.info()
    df['TEXT'] = notes_list
    # detect outcome column
    outcomes = get_features(outcomes, df)

    print("good here")
    keywords = {}
    for outcome in outcomes:
        y = np.array(df[outcome])
        num_class = len(np.unique(y))
        X = np.array(df['TEXT'])
        vectorizer = CountVectorizer(min_df=0.2, ngram_range=(1, 1))  # unigrams
        X = vectorizer.fit_transform(notes_list).toarray()
        name = vectorizer.get_feature_names()

        x_train_all, x_predict, y_train_all, y_predict = train_test_split(X, y, test_size=0.10, random_state=100)
        x_train, x_test, y_train, y_test = train_test_split(x_train_all, y_train_all, test_size=0.2, random_state=100)
        train_data = lgb.Dataset(data=x_train, label=y_train)
        test_data = lgb.Dataset(data=x_test, label=y_test)
        param = {'num_leaves': 31, 'objective': 'multiclass', 'num_class': num_class}
        param['metric'] = 'multi_logloss'
        num_round = 10
        evals = {}
        model = lgb.train(param, train_data, num_round, valid_sets=[test_data],
                          callbacks=[lgb.record_evaluation(evals)])

        # model = generate_model(X, y, num_class)
        feature_imp = pd.DataFrame({"value": model.feature_importance(), 'Feature': vectorizer.get_feature_names()})
        keywords[outcome] = pd.DataFrame(
            feature_imp.sort_values(by=feature_imp.columns[0], ascending=False)[0:num_keywords])
        # lgb.plot_metric(evals)
    if dump_to:
        with open(dump_to, 'wb') as f:
            pickle.dump(keywords, f)
    return keywords, outcomes
