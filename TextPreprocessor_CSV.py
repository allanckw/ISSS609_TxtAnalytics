
import re # The following statement imports the regex package.
import gensim # The following statement imports the gensim package.
import nltk # The following statement imports the NLTK package.
import pkg_resources
import pandas as pd
import os
import numpy as np
from symspellpy import SymSpell, Verbosity
from nltk import pos_tag, word_tokenize
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

class TextPreprocessor_CSV(object):
    """description of class"""
    corpus_dictionary = None
    
    posts = None

    suicidal_data_TF = []
    suicidal_data_TFIDF = []
         
    non_suicidal_data_TF = []
    non_suicidal_data_TFIDF = []

    all_labeled_data = []
    all_tfidf_labeled_data = []

    @property
    def DataFrame(self):
        return self.posts

    @property
    def CorpusDictionary(self):
        return self.corpus_dictionary

    @property
    def tf_labeled_data(self):
        return self.all_labeled_data

    @property
    def tfidf_labeled_data(self):
       return self.all_tfidf_labeled_data

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
        
        #remove punctuation and numbers
        self.posts['text'] = self.posts['text'].apply(lambda t: re.sub(r'[^a-zA-Z ]', '', t))
    
        #remove stop words first round, this is to reduce the number of words that symspell need to process...
        #however typo cannot be removed
        self.posts['text'] = self.posts['text'].apply(lambda t: gensim.parsing.preprocessing.remove_stopwords(str(t)))

        #Fixing typos
        #https://medium.com/@yashj302/spell-check-and-correction-nlp-python-f6a000e3709d
        #self.posts['text'] = self.posts['text'].apply(lambda t: str(TextBlob(t).correct()))
        #set max_dictionary_edit_distance = 1 to use Levenshtein instead of Damerau-Levenshtein algorithm (max_dictionary_edit_distance=2) to improve performance
        #In general, if you need to correct complex spelling errors that involve transpositions or other types of character 
        #changes, Damerau-Levenshtein may be the best choice. If you're working with a large dataset or need to perform spelling 
        #correction in real time, Levenshtein may be a good compromise between speed and accuracy.
        symsp = SymSpell(max_dictionary_edit_distance=1, prefix_length=7) #Damerau-Levenshtein algorithm = 2, Levenshtein = 1
        dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
        symsp.load_dictionary(dictionary_path, term_index=0, count_index=1)
        self.posts['text'] = self.posts['text'].apply(lambda t: symsp.lookup_compound(t, max_edit_distance=1)[0].term )

        #remove stop words again after typo corrected
        self.posts['text'] = self.posts['text'].apply(lambda t: gensim.parsing.preprocessing.remove_stopwords(str(t)))

        #lemmatization, note: Lemmatization includes stemming as discussed in class.
        self.posts['text_lemmatized_with_postag'] = self.posts['text'].apply(lambda t: self.pos_tag_and_lemmatize(t))

        self.posts['text_lemmatized'] = self.posts['text_lemmatized_with_postag'].apply(lambda t: self.getLemmatizedText(t))
        #print(self.posts.head())

        self.generateVectors()

    def generateVectors(self):
         #create dictionary
        all_docs5 = []

        #create corpus dictionary
        self.corpus_dictionary = gensim.corpora.Dictionary()

        for i in range(len(self.posts)):
            all_docs5.append(self.posts.loc[i, "text_lemmatized"])
            
        self.corpus_dictionary.add_documents(all_docs5)

        #print(self.corpus_dictionary)

        # Convert all documents to term frequency (TF) vectors
        all_tf_vectors = [self.corpus_dictionary.doc2bow(doc) for doc in all_docs5]
        #ntc = n = raw, t = zero-corrected idf, c = cosine - https://radimrehurek.com/gensim/models/tfidfmodel.html
        tfidf = gensim.models.TfidfModel(all_tf_vectors, smartirs='ntc')
        corpus_tfidf = tfidf[all_tf_vectors]

        all_data_as_dict = [{id:tf_value for (id, tf_value) in vec} for vec in all_tf_vectors]
        tfidf_data_as_dict = [{id:tf_value for (id, tf_value) in vec} for vec in corpus_tfidf]


        for i in range(len(all_data_as_dict)):
            doc_tf = all_data_as_dict[i]
            doc_tfidf = tfidf_data_as_dict[i]
            doc_label = self.posts.loc[i, "class"]

            if(doc_label == "suicide"):
                self.suicidal_data_TF.append((doc_tf, doc_label))
                self.suicidal_data_TFIDF.append((doc_tfidf, doc_label))

            else:
                self.non_suicidal_data_TF.append((doc_tf, doc_label))
                self.non_suicidal_data_TFIDF.append((doc_tfidf, doc_label))
      
        self.all_labeled_data = self.suicidal_data_TF + self.non_suicidal_data_TF
        self.all_tfidf_labeled_data  = self.suicidal_data_TFIDF + self.non_suicidal_data_TFIDF

    def SaveDictionary(self):
        self.CorpusDictionary.save("suicidalDataset.dict")


    def get_wordnet_pos(self,treebank_tag):
        """
        return WORDNET POS compliance to WORDNET lemmatization (a,n,r,v) 
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
                value = re.sub(r'[^a-zA-Z ]', '', p[1])
                lst.append(value)

        return lst

    #deprecated
    def Num2Words(self, sentence):
        
        numbers_to_words = ""
        sentence = sentence.replace("\n", "")
        tokens = word_tokenize(sentence.strip())

        for i in range(len(tokens)):
            value = tokens[i].strip()
            if len(value) > 0:
                if value.isdigit():
                    value = num2words(value)
                
                numbers_to_words = numbers_to_words + ' ' + value

        print(numbers_to_words)

        return numbers_to_words