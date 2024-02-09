import json
import nltk
import numpy as np

def extract_keyword(total_text, max_word=3):
    lines = total_text.split('\n')
    
    text = lines[0]
    keyword = json.loads(lines[-1])
    
    if  max_word < len(keyword):
        keyword_nouns = [list(item[1].keys())[0] for item in list(keyword.items())[:max_word]]
    else:
        keyword_nouns = [list(item[1].keys())[0] for item in list(keyword.items())]
        pass
    
    return text, keyword_nouns

def get_tag(tokenized, tags):
    ret = []
    for (word, pos) in nltk.pos_tag(tokenized):
        for tag in tags:
            if pos == tag:
                ret.append(word)
                
    return ret

def extract_nouns(text, max_word=3):
    tokenized = nltk.word_tokenize(text)
    nouns = []
    
    if len(tokenized) > 0:
        nouns = get_tag(tokenized, ['NN', 'NNS', 'NNP', 'VBG', 'VB', 'VBD', 'VBN', 'VBP', 'VBZ'])
    
    if len(nouns) > 0:
        select_nouns = np.random.choice(nouns, min(max_word, len(nouns)), replace=False)
    return select_nouns


def extract_words(total_text):
    text, keywords = extract_keyword(total_text)
    nouns = extract_nouns(text)
    
    return text, nouns, keywords
