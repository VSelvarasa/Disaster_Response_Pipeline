import pickle
import re
import sys
import warnings
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sqlalchemy import create_engine


def load_data(database_filepath):
    """Load cleaned data from database into dataframe.
    Args:
        database_filepath: path (String)
        table_name: (String)
    Returns:
       X: numpy.ndarray. Disaster messages.
       y: numpy.ndarray. Disaster categories for each messages.
       category_name: (list)
    """
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('disaster_messages', con=engine)

    category_names = df.columns[4:]

    X = df[['message']].values[:, 0]
    y = df[category_names].values

    return X, y, category_names


def tokenize(message):
    """Tokenize message
    Args:
        message: A disaster message. (String)
    Returns:
        a list containing tokens.
    """
    #https://stackoverflow.com/questions/28840908/perfect-regex-for-extracting-url-with-re-findall
    url = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

    # Detect URLs and replace them
    url_detected = re.findall(url, message)
    for i in url_detected:
        message = message.replace(i, ' ')

    # Normalize and tokenize
    tokens = nltk.word_tokenize(re.sub(r"[^a-zA-Z0-9]", " ", message.lower()))

    # Remove stopwords
    tokens = [t for t in tokens if t not in stopwords.words('english')]

    # Lemmatize
    lemmatizer = [WordNetLemmatizer().lemmatize(t) for t in tokens]

    return lemmatizer




def build_model():
    """Build model.
    Returns:
        pipeline
    """
    # Set pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(
            AdaBoostClassifier(
                base_estimator=DecisionTreeClassifier(max_depth=1, class_weight='balanced'),learning_rate=0.3,n_estimators=200)
            # Max_depth = 1 is used for preventing overfitting, to avoid tree growing very deep.
        ))
    ])

    # Set parameters for grid search
    parameters = {
        'clf__estimator__learning_rate': [0.1, 0.3],
        'clf__estimator__n_estimators': [100, 200]
    }

    # Set grid search
    cv = GridSearchCV(estimator=pipeline, param_grid=parameters, cv=3, scoring='f1_weighted', verbose=3)

    return cv


def evaluate_model(model, X_test, y_test, categories):
    """Evaluate model
    Args:
        model: sklearn.model_selection.GridSearchCV.  
        X_test: numpy.ndarray. Disaster messages.
        y_test: numpy.ndarray. Disaster categories for each messages
        category_names: Disaster category names.
    """
    # Predict categories of messages.
    y_pred = model.predict(X_test)

    # Print accuracy, precision, recall and f1_score for each categories
    for i in range(0, len(categories)):
        print(categories[i])
        print(classification_report(y_test[:, i], y_pred[:, i]))
        print("--------------------------------------------------")


def save_model(model, model_filepath):
    """Save model
    Args:
        model: sklearn.model_selection.GridSearchCV. 
        model_filepath: path where the pickle (train model) is saved. (String)
    """
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        warnings.filterwarnings('ignore')
        
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

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