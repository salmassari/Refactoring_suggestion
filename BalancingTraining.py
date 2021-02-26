from collections import Counter

import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import NearMiss
from nltk import collections
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


#dataset =pd.read_csv("./dataset/Nyamawe-work.csv")
dataset =pd.read_csv("./dataset/FR-Dataset.csv")
df_tags = dataset.dropna()
y=df_tags["Class"].str.lstrip()
x=df_tags['Text'].astype('U')

# Display the quantity of data points for each class
c = collections.Counter(y)
for key, value in c.items():
    print(key, value)


# To spilt the dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=9000)


##VECTORIZING##
vect = CountVectorizer()
tft = TfidfTransformer()

#    To vectorize the texts in training set
text = vect.fit_transform(x_train)
X = tft.fit_transform(text)
#    To vectorize the texts in training set
B1= vect.transform(x_test)
B1 = tft.transform(B1)

##ENCODING##
lb = LabelEncoder()
#   To encode the labels in training set
Y = lb.fit_transform(y_train)
#   To encode the labels in testing set
B2 = lb.transform(y_test)


## Balancing the training set ##
ros = RandomOverSampler(random_state=9000)
X_tfidf_resampled, Y_tfidf_resampled = ros.fit_resample(X, Y)



# CLASSIFCATION MODELS
sgd = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=6, tol=None)
lr = LogisticRegression()
mn = MultinomialNB()
svm = LinearSVC()
rf =RandomForestClassifier()
mlp =MLPClassifier(solver='adam', alpha=0.0001,hidden_layer_sizes=(35), activation="tanh",shuffle=True)
for classifier in [svm]:
    clf = OneVsRestClassifier(classifier)
    clf.fit(X, Y)
    y_pred = clf.predict(B1)
    print(classification_report(B2, y_pred))

    ## Represenation of classess with its encode.
    le_name_mapping = dict(zip(lb.classes_, lb.transform(lb.classes_)))
    print(le_name_mapping)

