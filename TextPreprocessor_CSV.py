
import re # The following statement imports the regex package.
import gensim # The following statement imports the gensim package.
import nltk # The following statement imports the NLTK package.
import pkg_resources
import pandas as pd
import os
import numpy as np

from num2words import num2words
from decimal import Decimal
from symspellpy import SymSpell, Verbosity
from nltk import pos_tag, word_tokenize
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

class TextPreprocessor_CSV(object):
    """description of class"""
    __corpus = None
    
    posts = None

    @property
    def DataFrame(self):
        return self.posts

    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)

    def __init__(self, fileName:str):
         # Read data 
        isFileExist = os.path.isfile(fileName)

        if(isFileExist == True):
            self.posts = pd.read_csv(fileName)
        else:
            print("File not found: " + os.getcwd() + fileName)

    def preprocessText(self):
         # Print head
        #print(self.posts.head())
        
        #drop data with empty data points
        self.posts['text'].replace('', np.nan, inplace=True)
        self.posts['class'].replace('', np.nan, inplace=True)
        self.posts.dropna(subset=['text'], inplace=True)
        self.posts.dropna(subset=['class'], inplace=True)

        #to lower case
        self.posts['text'] = self.posts['text'].str.lower()
        
        #remove punctuation
        self.posts['text'] = self.posts['text'].apply(lambda t: re.sub(r'[^\w\s]', '', t))

        #self.posts['text'] = self.posts['text'].apply(lambda t: gensim.parsing.preprocessing.preprocess_string(t))

        #convert num to words
        self.posts['text'] = self.posts['text'].apply(lambda t: self.Num2Words(t))

        #Fixing typos
        #https://medium.com/@yashj302/spell-check-and-correction-nlp-python-f6a000e3709d
        #self.posts['text'] = self.posts['text'].apply(lambda t: str(TextBlob(t).correct()))
        symsp = SymSpell(max_dictionary_edit_distance=3, prefix_length=7)
        dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
        symsp.load_dictionary(dictionary_path, 0, 1)
        self.posts['text'] = self.posts['text'].apply(lambda t: symsp.lookup_compound(t, max_edit_distance=3)[0] )

        #remove stop words

        self.posts['text'] = self.posts['text'].apply(lambda t: gensim.parsing.preprocessing.remove_stopwords(str(t)))

        #lemmatization, note: Lemmatization includes stemming as discussed in class.
        self.posts['text_lemmatized_with_postag'] = self.posts['text'].apply(lambda t: self.pos_tag_and_lemmatize(t))

        self.posts['text_lemmatized'] = self.posts['text_lemmatized_with_postag'].apply(lambda t: self.getLemmatizedText(t))
        #print(self.posts.head())

      
    def Num2Words(self, sentence):
        
        numbers_to_words = ""
        sentence = sentence.replace("\n", "")
        tokens = word_tokenize(sentence.strip())

        for i in range(len(tokens)):
            value = tokens[i].strip()
            if len(value) > 0:
                if value.isdigit():
                    value = num2words(Decimal(str(value)))
                
                numbers_to_words = numbers_to_words + ' ' + value

        return numbers_to_words
    
    def get_wordnet_pos(self,treebank_tag):
        """
        return WORDNET POS compliance to WORDENT lemmatization (a,n,r,v) 
        """
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            # As default pos in lemmatization is Noun
            return wordnet.NOUN

    def pos_tag_and_lemmatize(self, sentence):
        # find the pos tagging for each tokens [('What', 'WP'), ('can', 'MD'), ('I', 'PRP') ....

        pos_tokens = [nltk.pos_tag(word_tokenize(sentence))]
        lemmatizer = WordNetLemmatizer()
        # lemmatization using pos tag  
        pos_tokens = [ [(token, lemmatizer.lemmatize(token,self.get_wordnet_pos(pos_tag)), [pos_tag]) for (token,pos_tag) in pos] for pos in pos_tokens]
        return pos_tokens

    def getLemmatizedText(self, pos_tag):
        lst = list()
        for postag_tuple in pos_tag:
            for p in postag_tuple:
                #print(p[1])
                value = re.sub(r'[^\w\s]','',p[1])
                if(len(value.strip()) > 0):
                    if value.isdigit():
                        lst.append(num2words(value))
                    else:
                        lst.append(value)

        return lst
