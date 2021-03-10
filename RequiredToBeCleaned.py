import matplotlib.pyplot as plt
import pandas as pd
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import NearMiss, RandomUnderSampler
from nltk import collections, ConfusionMatrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.metrics import classification_report, plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

#dataset =pd.read_csv("./dataset/Nyamawe-work.csv")
from textblob import TextBlob

dataset =pd.read_csv("./dataset/FR-Dataset.csv")
df_tags = dataset.dropna()
y=df_tags["Class"].str.lstrip()


# Display the quantity of data points for each class
c = collections.Counter(y)
for key, value in c.items():
    print(key, value)

def textblob_tokenizer(str_input):
    blob = TextBlob(str_input.lower())
    tokens = blob.words
    words = [token.stem() for token in tokens]
    return words


def word_cleaning(df_text):
    X=df_text.iloc[:, 1]
    data = X
    data=data.split('[A-Z]', expand=True)
    # data=data.str.lstrip()
    data = data.replace({r'[^a-zA-Z\s]': ' '}, regex=True, inplace=False)
    # data=data.replace('\-\d*', '', regex=True, inplace=True)
    # # data=data.replace('\d+', '', regex=True, inplace=True)
    # data=data.str.lstrip()
    data = data.replace("  ", '').replace('\n', '').replace('\t', '').replace(
        '', 'NONE').replace(' ', 'NONE')
    return data
print(df_tags)

x= word_cleaning(dataset)
print(x)

# To spilt the dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=9000)


##VECTORIZING##
vect = CountVectorizer(analyzer="word",max_df=0.50, lowercase=True,stop_words="english", strip_accents="ascii",ngram_range=(1,2), tokenizer=textblob_tokenizer)
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
run = RandomUnderSampler(random_state=9000)
smote = SMOTE()
nm = NearMiss()
#X_tfidf_resampled, Y_tfidf_resampled = ros.fit_resample(X, Y)
#X_tfidf_resampled, Y_tfidf_resampled = run.fit_resample(X, Y)
#X_tfidf_resampled, Y_tfidf_resampled = nm.fit_sample(X, Y)
X_tfidf_resampled, Y_tfidf_resampled = smote.fit_resample(X, Y)

# CLASSIFCATION MODELS
sgd = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=6, tol=None)
lr = LogisticRegression()
mn = MultinomialNB()
svm = LinearSVC()
rf =RandomForestClassifier()
mlp =MLPClassifier(solver='adam', alpha=0.0001,hidden_layer_sizes=(35), activation="tanh",shuffle=True)
for classifier in [svm]:
    clf = OneVsRestClassifier(classifier)
    clf.fit(X_tfidf_resampled, Y_tfidf_resampled)
    y_pred = clf.predict(B1)
    print(classification_report(B2, y_pred))

    # plot_confusion_matrix(clf, B2, B1)
    # plt.show()
    voc=vect.vocabulary_
    #print(dict(sorted(voc.items(), key=lambda item: item[1])))
    cm = ConfusionMatrix(B2, y_pred)
    print(cm)
    plt.show()
    ## Represenation of classess with its encode.
    le_name_mapping = dict(zip(lb.classes_, lb.transform(lb.classes_)))
    print(le_name_mapping)

