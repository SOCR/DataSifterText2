import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import spacy
import string
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tag.stanford import StanfordNERTagger

def text_clean(text, control):
    text = text.copy()

    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    for i in range(len(text)):
        text[i] = extract_word(text[i])
        if control['lemmatize']:
            for j in range(len(text[i])):
                text[i][j] = lemmatizer.lemmatize(text[i][j])

    if control['stop_words']:
        for i in range(len(text)):
            for j in range(len(text[i])):
                if text[i][j] in stop_words:
                    text[i][j] = ""

    if control['remove_number']:
        for i in range(len(text)):
            for j in range(len(text[i])):
                if text[i][j].isnumeric():
                    text[i][j] = ""

    for i in range(len(text)):
        text[i] = ' '.join(text[i])

    vectorizer = CountVectorizer(max_df=0.8, min_df=3, ngram_range=(1,control['gram']))
    X = vectorizer.fit_transform(text)
    return vectorizer.get_feature_names(), X.toarray()

def generate_model(X, y, num_class):
    x_train_all, x_predict, y_train_all, y_predict = train_test_split(X, y, test_size=0.10, random_state=100)
    x_train, x_test, y_train, y_test = train_test_split(x_train_all, y_train_all, test_size=0.2, random_state=100)
    train_data = lgb.Dataset(data=x_train, label=y_train)
    test_data = lgb.Dataset(data=x_test, label=y_test)
    param = {'num_leaves': 31, 'objective': 'multiclass', 'num_class':num_class}
    param['metric'] = 'multi_logloss'
    num_round = 10
    bst = lgb.train(param, train_data, num_round, valid_sets=[test_data])
    # dd = bst.trees_to_dataframe()
    # print(dd)
    # graph = lgb.create_tree_digraph(bst)
    # graph.render()
    # return ["awd", "adw"]
    return bst

def get_features(aims, df):
    titles = df.columns.values.tolist()
    nlp = spacy.load("en_core_web_lg")
    new_aims = aims.copy()
    curr_feature = titles[0]
    for i in range(len(aims)):
        aim = word_pp(aims[i])
        doc1 = nlp(aim)
        best_sim = 0
        for title in titles:
            temp_t = word_pp(title)
            doc2 = nlp(temp_t)
            curr_sim = doc1.similarity(doc2)
            if curr_sim > best_sim:
                best_sim = curr_sim
                curr_feature = title
        new_aims[i] = curr_feature
    return new_aims


def word_pp(word):
    word = list(word)
    for i in range(len(word)):
        if word[i] in string.punctuation:
            word[i] = ' '
    return ''.join(word)

def extract_word(input_string):
    pu = string.punctuation
    for p in pu:
        input_string = input_string.replace(p, ' ')
    return input_string.lower().split()


# TODO: percentage way
def split_class(y, n):
    m = np.mean(y)
    y[y > m * 2] = m * 2
    sd = max(y) / n
    for i in range(len(y)):
        for j in range(n):
            if y[i] >= j * sd and y[i] <= (j + 1)*sd:
                y[i] = j