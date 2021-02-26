from collections import Counter

import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from nltk import collections
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

dataset =pd.read_csv("./dataset/Nyamawe-work.csv")

# print('here the result\n')
# print(len(dataset))
# print()
num_classes = 14
# grouped_tags = dataset.groupby("Class").size().reset_index(name='count')
# most_common_tags = grouped_tags.nlargest(num_classes, columns="count")
df_tags = dataset.dropna()
c = collections.Counter(df_tags["Class"])
# for key, value in c.items():
#     print(key, value)
y=df_tags["Class"].str.lstrip()


lb = LabelEncoder()
Y = lb.fit_transform(y)
#print(lb.classes_)


# count_vect = CountVectorizer()
count_vect = CountVectorizer(stop_words='english',max_features=1000,analyzer = 'word',
                                            ngram_range=(1,2))
X_counts = count_vect.fit_transform(df_tags['Commit Message'].astype('U'))



tfidf_transformer = TfidfTransformer()
X_tfidf = tfidf_transformer.fit_transform(X_counts)
ros = RandomOverSampler(random_state=9000)

X_tfidf_resampled, Y_tfidf_resampled = ros.fit_resample(X_tfidf, Y)

c = collections.Counter(Y_tfidf_resampled)
for key, value in c.items():
    print(key, value)
print('here the result\n')
print(len(Y_tfidf_resampled))
x_train_tfidf, x_test_tfidf, y_train_tfidf, y_test_tfidf = train_test_split(X_tfidf_resampled, Y_tfidf_resampled, test_size=0.2, random_state=1234)


# def hamming_score(y_true, y_pred, normalize=True, sample_weight=None):
#     print(y_true.shape[0])
#     print(y_pred)
#
#     acc_list = []
#     for i in range(y_true.shape[0]):
#         set_true = set(np.where(y_true[i])[0])
#         set_pred = set(np.where(y_pred[i])[0])
#         tmp_a = None
#         if len(set_true) == 0 and len(set_pred) == 0:
#             tmp_a = 1
#         else:
#             # tmp_a = len(set_true.union(set_pred))
#             tmp_a = len(set_true.intersection(set_pred))/float(len(set_true.union(set_pred)) )
#         acc_list.append(tmp_a)
#     # print(acc_list)
#     return np.mean(acc_list)
#
# def print_score(y_pred, clf):
#     print("Clf: ", clf.__class__.__name__)
#     # print("Hamming loss: {}".format(hamming_loss(y_test_tfidf, y_pred)))
#     print("Hamming score: {}".format(hamming_score(y_test_tfidf, y_pred)))
#     # print('Subset accuracy: {0}'.format(accuracy_score(y_test_tfidf, y_pred, normalize=True, sample_weight=None)))
#     # print('Subset precision: {0}'.format(precision_score(y_test_tfidf, y_pred, average='samples')))
#     print("---")


sgd = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=5, tol=None)
#lr = LogisticRegression()
mn = MultinomialNB()
svm = LinearSVC()
rf =RandomForestClassifier()
mlp =MLPClassifier(solver='adam', alpha=0.0001,hidden_layer_sizes=(35), activation="tanh",shuffle=True)


for classifier in [svm]:
    svm = OneVsRestClassifier(classifier)
    svm.fit(x_train_tfidf, y_train_tfidf)
    pred = svm.predict(x_test_tfidf)

    # print_score(y_pred, classifier)
print(classification_report(y_test_tfidf, pred))
le_name_mapping = dict(zip(lb.classes_, lb.transform(lb.classes_)))
print(le_name_mapping)