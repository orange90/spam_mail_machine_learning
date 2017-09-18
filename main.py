from data import get_train_test_split
from model import *
from sklearn.metrics import accuracy_score,confusion_matrix,recall_score, f1_score, roc_curve
from sklearn.model_selection import cross_val_score

pipeline = get_naive_bayes_model()
X_train, X_test, y_train, y_test = get_train_test_split()

pipeline.fit(X_train, y_train)
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5)
print cv_scores
predicted = pipeline.predict(X_test) # ['spam', 'ham']
print confusion_matrix(y_test, predicted)
print accuracy_score(y_test, predicted)
print f1_score(y_test, predicted)
print recall_score(y_test, predicted)
# roc_curve(y_test, predicted)
