import re
import contractions
import nltk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('stopwords')
stopwords = stopwords.words('english')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
lemmatizer = WordNetLemmatizer()
proterstemmer = PorterStemmer()


def clean_data(raw):
    raw_df = raw.copy()

    # word stemmer
    raw_df.info()
    raw_df.head()
    raw_df = raw_df.apply(lambda x:
                          " ".join(re.sub(r'[^a-zA-Z]', " ", w).lower() for w in x.split()
                                   if re.sub(r'^a-zA-Z', ' ', w).lower() not in stopwords))
    notes_list, tokenized_notes = preprocessing(raw_df)
    return notes_list, tokenized_notes


def decontracted(phrase):
    """decontracted takes text and convert contractions into natural form.
     ref: https://stackoverflow.com/questions/19790188/expanding-english-language-contractions-in-python/47091490#47091490"""

    # specific
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase


def expand_contractions(phrase):
    """use contractions package to do decontraction"""
    phrase = contractions.fix(phrase)
    return phrase


def preprocessing(notes):
    """
    preprocess notes including lemmatization, decontraction, stemming, lowercase,
    return notes(preprocessed notes), tokenized_notes(list)
    """
    tokenized_notes = []
    for i, text in enumerate(notes):
        notes[i] = decontracted(text)
        text_word_list = word_tokenize(text)

        # expand contraction
        text_word_list = [expand_contractions(word) for word in text_word_list]
        # lemmatization
        text_word_list = [lemmatizer.lemmatize(word) for word in text_word_list]
        # stemming
        text_word_list = [proterstemmer.stem(word) for word in text_word_list]
        # to lower
        text_word_list = [word.lower() for word in text_word_list]

        tokenized_notes.append(text_word_list)
        notes[i] = " ".join(text_word_list)
    return notes, tokenized_notes
