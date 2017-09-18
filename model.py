from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer


def get_naive_bayes_model():
    pipeline = Pipeline([
        ('vectorizer',  CountVectorizer()),
        ('classifier',  MultinomialNB())])
    return pipeline
