from TextPreprocessor_CSV import TextPreprocessor_CSV
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB

txtProcessorCSV_Train = TextPreprocessor_CSV('./data/train.csv')
txtProcessorCSV_Train.preprocessText()

txtProcessorCSV_Validate = TextPreprocessor_CSV('./data/validate.csv')
txtProcessorCSV_Validate.preprocessText()
#print(txtProcessorCSV.DataFrame.head())

#print(txtProcessorCSV.DataFrame['text_lemmatized'].to_string(index=False))

multinomialNaivesBayes_classifier = SklearnClassifier(MultinomialNB())

multinomialNaivesBayes_classifier.train(txtProcessorCSV.all_tfidf_labeled_data)
       
print(f'Accuracy of multinomial Naives Bayes classifier: {nltk.classify.accuracy(multinomialNaivesBayes_classifier, txtProcessorCSV_Validate.tfidf_labeled_data)}')
