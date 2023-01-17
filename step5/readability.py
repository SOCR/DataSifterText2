import spacy
from textstat.textstat import textstatistics
 
# Splits the text into sentences
def break_sentences(text):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    return list(doc.sents)
 
# Returns Number of Words in the text
def word_count(text):
    sentences = break_sentences(text)
    words = 0
    for sentence in sentences:
        words += len([token for token in sentence])
    return words
 
# Returns the number of sentences in the text
def sentence_count(text):
    sentences = break_sentences(text)
    return len(sentences)
 
# Returns average sentence length
def avg_sentence_length(text):
    words = word_count(text)
    sentences = sentence_count(text)
    average_sentence_length = float(words / sentences)
    return average_sentence_length
 
# Textstat is a python package, to calculate statistics from
# text to determine readability
def syllables_count(word):
    return textstatistics().syllable_count(word)
 
# Returns the average number of syllables per
# word in the text
# def avg_syllables_per_word(text):
#     syllable = syllables_count(text)
#     words = word_count(text)
#     ASPW = float(syllable) / float(words)
#     return legacy_round(ASPW, 1)
 
# Return total Difficult Words in a text
def difficult_words(text):
     
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    # Find all words in the text
    words = []
    sentences = break_sentences(text)
    for sentence in sentences:
        words += [str(token) for token in sentence]
 
    # difficult words are those with syllables >= 2
    # easy_word_set is provide by Textstat as
    # a list of common words
    diff_words_set = set()
     
    for word in words:
        syllable_count = syllables_count(word)
        if word not in nlp.Defaults.stop_words and syllable_count >= 2:
            diff_words_set.add(word)
 
    return len(diff_words_set)
 
# A word is polysyllablic if it has more than 3 syllables
# this functions returns the number of all such words
# present in the text
def poly_syllable_count(text):
    count = 0
    words = []
    sentences = break_sentences(text)
    for sentence in sentences:
        words += [token for token in sentence]
     
 
    for word in words:
        syllable_count = syllables_count(word)
        if syllable_count >= 3:
            count += 1
    return count
 
 
# def flesch_reading_ease(text):
#     """
#         Implements Flesch Formula:
#         Reading Ease score = 206.835 - (1.015 × ASL) - (84.6 × ASW)
#         Here,
#           ASL = average sentence length (number of words
#                 divided by number of sentences)
#           ASW = average word length in syllables (number of syllables
#                 divided by number of words)
#     """
#     FRE = 206.835 - float(1.015 * avg_sentence_length(text)) -\
#           float(84.6 * avg_syllables_per_word(text))
#     return legacy_round(FRE, 2)
 
 
def gunning_fog(text):
    per_diff_words = (difficult_words(text) / word_count(text) * 100) + 5
    grade = 0.4 * (avg_sentence_length(text) + per_diff_words)
    return grade
 
 
def smog_index(text):
    """
        Implements SMOG Formula / Grading
        SMOG grading = 3 + ?polysyllable count.
        Here,
           polysyllable count = number of words of more
          than two syllables in a sample of 30 sentences.
    """
 
    if sentence_count(text) >= 3:
        poly_syllab = poly_syllable_count(text)
        SMOG = (1.043 * (30*(poly_syllab / sentence_count(text)))**0.5) \
                + 3.1291
        return legacy_round(SMOG, 1)
    else:
        return 0
 
 
# def dale_chall_readability_score(text):
#     """
#         Implements Dale Challe Formula:
#         Raw score = 0.1579*(PDW) + 0.0496*(ASL) + 3.6365
#         Here,
#             PDW = Percentage of difficult words.
#             ASL = Average sentence length
#     """
#     words = word_count(text)
#     # Number of words not termed as difficult words
#     count = word_count - difficult_words(text)
#     if words > 0:
 
#         # Percentage of words not on difficult word list
 
#         per = float(count) / float(words) * 100
     
#     # diff_words stores percentage of difficult words
#     diff_words = 100 - per
 
#     raw_score = (0.1579 * diff_words) + \
#                 (0.0496 * avg_sentence_length(text))
     
#     # If Percentage of Difficult Words is greater than 5 %, then;
#     # Adjusted Score = Raw Score + 3.6365,
#     # otherwise Adjusted Score = Raw Score
 
#     if diff_words > 5:      
 
#         raw_score += 3.6365
         
#     return legacy_round(score, 2)

def evaluate_readability(text):
    return gunning_fog(text)
