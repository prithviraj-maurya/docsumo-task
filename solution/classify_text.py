## Imports
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

## Data
train = pd.read_csv('../data/train_set.csv')
test = pd.read_csv('../data/test_set.csv')
print(f"Train {train.shape}")
print(f"Test {test.shape}")
print(train.head())

## preprocessing
# encode label
le = LabelEncoder()
train.label = le.fit_transform(train.label)
# remove stop words
# train.text = train.text.apply(lambda row: [word for word in row if word not in stopwords.words('english')])
# stemming

print("Preprocessed data")
print(train.head())

# split data
x_train, x_val, y_train, y_val = train_test_split(train.text.values, train.label.values, test_size=0.2, shuffle=True)
print("Splitted data")
print(x_train.shape, x_val.shape)

## Build Classifier
text_clf = Pipeline([
     ('vect', CountVectorizer(stop_words='english')),
     ('tfidf', TfidfTransformer()),
     # ('nb', MultinomialNB(fit_prior=False, random_state=42)),
    # ('clf-svm', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42)),
    ('decision_tree', DecisionTreeClassifier(random_state=42))
])

clf = text_clf.fit(x_train, y_train)

## Performance
predicted = clf.predict(x_val)
print(np.mean(predicted == y_val))

## Hyperparameter tuning
# param_grid = {'nb__alpha': np.arange(0, 1, 0.05)}
# clf = GridSearchCV(text_clf, param_grid).fit(x_train, y_train)


tree_param = {'decision_tree__criterion':['gini','entropy'],'decision_tree__max_depth':[4,5,6,7,8,9,10,11,12,15,20,30,40,50,70,90,120,150]}
clf = RandomizedSearchCV(text_clf, tree_param).fit(x_train, y_train)

print(clf.best_estimator_)
print('best score:')
print(clf.best_score_)
