import gensim
import pandas as pd
from gensim.models import Word2Vec
import random

from nltk import WordNetLemmatizer, PorterStemmer

lemmatizer = WordNetLemmatizer()
proterstemmer = PorterStemmer()


class MySentences(object):
    def __init__(self, text):
        self.text = text

    def __iter__(self):
        for line in self.text:
            yield line.split()

    def __len__(self):
        return len(self.text)

    def __getitem__(self, i):
        return self.text[i]

    def __setitem__(self, index, item1):
        self.text[index] = item1


class W2V(object):
    def __init__(self, sentences, cores, vec_size) -> None:
        self.sentences = sentences
        self.model = Word2Vec(min_count=10,
                              window=10,
                              vector_size=vec_size,
                              sample=6e-5,
                              alpha=0.03,
                              min_alpha=0.0007,
                              negative=20,
                              workers=cores - 1)

    def train(self):
        self.model.build_vocab(self.sentences)
        self.model.train(self.sentences, total_examples=self.model.corpus_count, epochs=30, report_delay=1)

    def get_similar_word(self, keywords=None, start_radius=0, end_radius=100):
        if keywords is None:
            keywords = ["wife"]
        word_list = [self.model.wv.most_similar(keywords[i], topn=end_radius)[start_radius:] for i in
                     range(len(keywords))]
        return [random.choice(words)[0] for words in word_list]

    def get_similar_word_2(self, keywords=None, start_radius=0, end_radius=100):
        # print(len(keywords))
        # print(self.model.wv.most_similar(keywords[i], topn=end_radius))
        if keywords is None:
            keywords = ["wife"]
        word_list = [self.model.wv.most_similar(keywords[i], topn=end_radius)[start_radius:] for i in
                     range(len(keywords))]
        return word_list

    def get_similar_word_list(self, keywords, start_radius=0, end_radius=100):
        word_list = None
        if isinstance(keywords, list):
            TypeError("keywords should be a list")
        else:
            keywords = [keywords]
            for keyword in keywords:
                word_list = self.model.wv.most_similar(keyword, topn=end_radius)
        return word_list

    def save(self, route):
        self.model.save(route)

    def load(self, route):
        self.model = Word2Vec.load(route)

    def find_word_idx(self, keywords=None):
        if keywords is None:
            keywords = ["wife"]
        return [[i for i in range(len(self.sentences)) if keywords[j] in self.sentences[i]] for j in
                range(len(keywords))]

    def obfuscate(self, sentence, keywords=None):
        if keywords is None:
            keywords = ["wife"]
        word_list = self.get_similar_word(keywords=keywords)
        for i in range(len(keywords)):
            sentence = sentence.replace(keywords[i], word_list[i])
        return sentence


def find_3levels_word(w2v_model, keyword_list, small=33, mid=66):
    """"keywords is a list of predictive words of a certain outcome,
    small is the percentile criteria for small level of obfuscation,
    mid is the percentile criteria for mid level of obfuscation,

    return are 3 word list: each derived by random sampling from the corresponding obfuscation level"""

    small_level_list = []
    mid_level_list = []
    large_level_list = []
    for word in keyword_list:
        try:
            similar_words = sorted(w2v_model.get_similar_word_list(keywords=word, end_radius=100), key=lambda x: x[1],
                                   reverse=True)
        except:
            continue
        else:
            word_list, cos_list = list(zip(*similar_words))
            small_range = word_list[:small]
            mid_range = word_list[small:mid]
            large_range = word_list[mid:]
            small_level_list.append(random.sample(small_range, 1)[0])
            mid_level_list.append(random.sample(mid_range, 1)[0])
            large_level_list.append(random.sample(large_range, 1)[0])

    return small_level_list, mid_level_list, large_level_list


def get_replace_word(w2v_model, keywords, level, small=33, mid=66):
    if isinstance(level, str):
        small, mid, large = find_3levels_word(w2v_model, keywords, small, mid)
        level = level.lower()
        if level == "small":
            return small
        elif level == "mid":
            return mid
        elif level == "large":
            return large
    else:
        raise NotImplemented


def obfuscate(notes, words_replacement, level, sensitive_factor):
    """wordForReplacement: dictionary containing all senstitive factors and their corresponding words for replacement
    level: obfuscation level (small, mid, large)
    sensitive_factor: obfuscation factor (outcomes)"""
    word_for_replacement = words_replacement.copy()
    #     sensitive_factor = "Religion"
    all_note = notes.copy()
    for i, note in enumerate(all_note):
        replace_dict = {}
        for word in note.split():
            curr_word = lemmatizer.lemmatize(word).lower()
            if curr_word in list(word_for_replacement[sensitive_factor]['keywords']):
                if level == "small":
                    potential_words = word_for_replacement[sensitive_factor]
                    word_for_replace = potential_words[potential_words['keywords'] == curr_word]['small_level'].values[
                        0]
                    replace_dict[curr_word] = word_for_replace
        # print(replace_dict)
        for target, replace in replace_dict.items():
            note = note.replace(target, replace)
        all_note[i] = note

    return all_note


class W2VGN:
    def __init__(self):
        self.model = gensim.models.keyedvectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

    def find_3levels_word_GN(self, keyword_list, small=33, mid=66):
        """"keywords is a list of predictive words of a certain outcome,
        small is the percentile criteria for small level of obfuscation,
        mid is the percentile criteria for mid level of obfuscation,

        return are 3 word list: each derived by random sampling from the corresponding obfuscation level"""

        small_level_list = []
        mid_level_list = []
        large_level_list = []
        for word in keyword_list:
            try:
                # TODO: Who on earth write this shit? UNBELIEVABLE
                similar_words = sorted(
                    self.model.similar_by_word(word=keyword_list[0], topn=100),
                    key=lambda x: x[1], reverse=True)
            except:
                continue
            else:
                word_list, cos_list = list(zip(*similar_words))
                small_range = word_list[:small]
                mid_range = word_list[small:mid]
                large_range = word_list[mid:]
                small_level_list.append(random.sample(small_range, 1)[0])
                mid_level_list.append(random.sample(mid_range, 1)[0])
                large_level_list.append(random.sample(large_range, 1)[0])

        return small_level_list, mid_level_list, large_level_list

    def get_replace_word_GN(self, keywords, level, small=33, mid=66):
        if isinstance(level, str):
            small, mid, large = self.find_3levels_word_GN(keywords, small, mid)
            level = level.lower()
            if level == "small":
                return small
            elif level == "mid":
                return mid
            elif level == "large":
                return large
        else:
            raise NotImplemented

    def generate_words_for_replacement(self, keywords_dictionary):
        """input:
        keywords_dict: dictionary, with key = sensitive factor, value = predictive words of that factor
        return the data frame containing the words for replacement of each sensitive factor"""
        words_for_replacement = {}
        for key, item in keywords_dictionary.items():
            words_for_replacement[key] = pd.DataFrame({'keywords': keywords_dictionary[key],
                                                       'small_level': self.get_replace_word_GN(keywords_dictionary[key],
                                                                                               "small"),
                                                       'mid_level': self.get_replace_word_GN(keywords_dictionary[key],
                                                                                             "mid"),
                                                       'large_level': self.get_replace_word_GN(keywords_dictionary[key],
                                                                                               "large")})

        return words_for_replacement
