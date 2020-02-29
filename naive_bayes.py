import pandas as pd
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.naive_bayes import BernoulliNB

# load data
Corpus = pd.read_csv("all.csv")

# pre-processing
Corpus['question_text'].dropna(inplace=True)
# Change all the text to lower case
Corpus['question_text'] = [entry.lower() for entry in Corpus['question_text']]
# corpus will be broken into set of words
Corpus['question_text']= [word_tokenize(entry) for entry in Corpus['question_text']]

# WordNetLemmatizer requires Pos tags to understand if the word is noun or verb or adjective etc. default is Noun
tag_map = defaultdict(lambda: wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV

for index, entry in enumerate(Corpus['question_text']):

    Final_words = []
    lemmatizer = WordNetLemmatizer()

    for word, tag in pos_tag(entry):
        if word not in stopwords.words('english') and word.isalpha():
            word_Final = lemmatizer.lemmatize(word, tag_map[tag[0]])
            Final_words.append(word_Final)

    Corpus.loc[index, 'processed'] = str(Final_words)


Corpus.to_csv('processed_text.csv')
# The Training Data will have 70% of the corpus and Test data will have the remaining 30%
# Train_X = training data predictor
# Test_X = training data target
# Train_Y = test data predictor
# Test_Y = test data target
Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(Corpus['processed'], Corpus['target'], test_size=0.3)


Encoder = LabelEncoder()
Train_Y = Encoder.fit_transform(Train_Y)
Test_Y = Encoder.fit_transform(Test_Y)


Tfidf_vect = TfidfVectorizer()
Tfidf_vect.fit(Corpus['processed'])

Train_X_Tfidf = Tfidf_vect.transform(Train_X)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)

# another naive bayes classifier
clf = BernoulliNB()
model = clf.fit(Train_X_Tfidf, Train_Y)

# predict the labels on validation data set
predictions_Bernoulli = clf.predict(Test_X_Tfidf)
predictions_Bernoulli2 = clf.predict(Train_X_Tfidf)


# another classifier (has not been used in this code)
Naive = naive_bayes.MultinomialNB()
Naive.fit(Train_X_Tfidf, Train_Y)
predictions_NB = Naive.predict(Test_X_Tfidf)
predictions_NB2 = Naive.predict(Train_X_Tfidf)


# accuracy
print("Naive Bayes accuracy for test data: ", accuracy_score(predictions_Bernoulli, Test_Y) * 100)
print("Naive Bayes accuracy for train data: ", accuracy_score(predictions_Bernoulli2, Train_Y) * 100)

# f1 measure for test data
print("f1 measure, micro:", f1_score(Test_Y, predictions_Bernoulli, average='micro'))
print("f1 measure, macro:", f1_score(Test_Y, predictions_Bernoulli, average='macro'))
print("f1 measure, none:", f1_score(Test_Y, predictions_Bernoulli, average=None))


# f1 measure for train data
print("f1 measure, micro:", f1_score(Train_Y, predictions_Bernoulli2, average='micro'))
print("f1 measure, macro:", f1_score(Train_Y, predictions_Bernoulli2, average='macro'))
print("f1 measure, none:", f1_score(Train_Y, predictions_Bernoulli2, average=None))

# change type from series to data frame
Train_X = Train_X.to_frame()
Test_X = Test_X.to_frame()

# add a column for predicted values by Naive Bayes
Train_X['predicted value'] = predictions_Bernoulli2.tolist()
Test_X['predicted value'] = predictions_Bernoulli.tolist()

# write to csv file
Train_X.to_csv(r'train_data_set_predicted_values.csv')
Test_X.to_csv(r'test_data_set_predicted_values.csv')
