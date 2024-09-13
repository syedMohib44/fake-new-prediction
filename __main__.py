import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import os.path

def get_data(file: str):
    if os.path.exists(file) == False:
       return None
    data = pd.read_csv(file)
    return data

def predict_fake_news(file: str):
    data = get_data(file)

    if data is None:
        print("File ""\"%s""\" not found" % file)
        return

    data['fake'] = data['label'].apply(lambda x: 0 if x == "REAL" else 1)
    data = data.drop("label", axis=1)
    X, y = data["text"], data["fake"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    X_train_vectorized = vectorizer.fit_transform(X_train) 
    X_test_vectorized = vectorizer.transform(X_test)
    
    # Using it for text classification 
    clf = LinearSVC()
    clf.fit(X_train_vectorized, y_train)

    print("Total score: %f" % clf.score(X_test_vectorized, y_test))

    article_text = X_test.iloc[10]
    vectorized_text = vectorizer.transform([article_text])
    print(clf.predict(vectorized_text))

    print(y_test.iloc[10])


predict_fake_news("fake_or_real_news.csv")