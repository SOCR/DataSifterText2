from gensim.models import Word2Vec
import random

from pyparsing import WordStart

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
    def __init__(self, sentences, cores) -> None:
        self.sentences = sentences
        self.model = Word2Vec(min_count=10,
                     window=10,
                     vector_size=300,
                     sample=6e-5, 
                     alpha=0.03, 
                     min_alpha=0.0007, 
                     negative=20,
                     workers=cores-1)
    
    def train(self):
        self.model.build_vocab(self.sentences)
        self.model.train(self.sentences, total_examples=self.model.corpus_count, epochs=30, report_delay=1)

    def get_similar_word(self, keywords=["wife"], start_radius=0, end_radius=100):
        word_list = [self.model.wv.most_similar(keywords[i], topn=end_radius)[start_radius:] for i in range(len(keywords))]
        return [random.choice(words)[0] for words in word_list]

    def save(self, route):
        self.model.save(route)

    def load(self, route):
        self.model = Word2Vec.load(route)

    def find_word_idx(self, keywords=["wife"]):
        return [[i for i in range(len(self.sentences)) if keywords[j] in self.sentences[i]] for j in range(len(keywords))]

    def obfuscate(self, sentence, keywords=["wife"]):
        word_list = self.get_similar_word(keywords=keywords)
        for i in range(len(keywords)):
            sentence = sentence.replace(keywords[i], word_list[i])
        return sentence

    