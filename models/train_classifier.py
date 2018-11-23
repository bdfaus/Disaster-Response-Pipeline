import sys
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import pickle

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import classification_report, hamming_loss
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

#nltk for tokenizing/lemmatizing
import nltk
nltk.download(['punkt','wordnet'])



def load_data(database_filepath):
    engine = create_engine('sqlite:///' + str(database_filepath))
    df = pd.read_sql_table('MessageTable',con=engine)

    X = df['message']
    Y = df.iloc[:,3:]
    cats = Y.columns

    return X, Y, cats


def tokenize(text):
    #split into tokens
    tokens = nltk.word_tokenize(text)
    
    #get to the ROOT of the words (lemmatize)
    lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
    
    clean_tokens = []
    for token in tokens:
        clean_token = lemmatizer.lemmatize(token, pos='v').lower()
        clean_tokens.append(clean_token)
    
    return clean_tokens


def build_model():
    pipeline = Pipeline(
    [('vectorizer',CountVectorizer(tokenizer=tokenize)),
     ('tfidf',TfidfTransformer()),
     ('clf',KNeighborsClassifier(leaf_size=5, n_neighbors = 5))]
    )

    return pipeline

def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)

    print(classification_report(Y_test, Y_pred, target_names=category_names))


def save_model(model, model_filepath):
    
    pickle.dump(model, open(model_filepath, 'wb'))

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()