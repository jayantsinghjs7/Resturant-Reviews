# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)
 # we do this because it is expecting a csv file 
 # code to ignore the quotes is 3

# Cleaning the texts
import re        # contains great tools to clean the texts
import nltk      # it will help us to look at words that are generically irrelevant like "THIS"
nltk.download('stopwords')  
# nltk.download('popular') to download popular terms


from nltk.corpus import stopwords 
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i]) 
    # stemming is about taking the root of the words 
    #sub removes the punctions question marks etc
    # we dont want to remove letter so we write after ^. we write here what not to remove rather than what to remove
    review = review.lower()   
    review = review.split()
    ps = PorterStemmer() 
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]  # as it conatins many languages
    review = ' '.join(review)
    corpus.append(review) 

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)   # keeping only the 1500 most frequent word used
X = cv.fit_transform(corpus).toarray()   # SPARSE MATRIX converted to array first fitted to corpus for reading the reviews and then transformed to form the sparse matrix
y = dataset.iloc[:, 1].values 

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train) 

# Predicting the Test set results
y_pred = classifier.predict(X_test) 

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)