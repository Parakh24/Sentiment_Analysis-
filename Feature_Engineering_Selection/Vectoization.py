from sklearn.model_selection import train_test_split 
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from Data_PreProcessing.data_cleaning import df 
from nltk.stem import WordNetLemmatizer 
from nltk import word_tokenize  
from sklearn.linear_model import LogisticRegression 
from prettytable import PrettyTable 
import pandas as pd


# ===========================
# Data Preprocessing
# ===========================
def preprocess_data(df):
    # Map ratings to binary labels
    df['Label'] = df['Ratings'].apply(lambda x: '1' if x >= 7 else ('0' if x <= 4 else '2'))
    df = df[df.Label < '2']  # Remove neutral
    data = df[['Reviews_clean', 'Label']]
    print("Label distribution:\n", data['Label'].value_counts())
    return data


# ===========================
# Lemmatization Tokenizer
# ===========================
class LemmaTokenizer(object):
    def __init__(self):
        self.wordnetlemma = WordNetLemmatizer()

    def __call__(self, reviews):
        return [self.wordnetlemma.lemmatize(word) for word in word_tokenize(reviews)]


# ===========================
# Train-Test Split & Vectorization
# ===========================
def vectorize_unigram(data, min_df=10, max_features=500):
    train, test = train_test_split(data, test_size=0.3, random_state=42, shuffle=True)
    
    # Count Vectorizer
    countvect = CountVectorizer(analyzer="word", tokenizer=LemmaTokenizer(),
                                ngram_range=(1, 1), min_df=min_df, max_features=max_features)
    x_train_count = countvect.fit_transform(train['Reviews_clean']).toarray()
    x_test_count = countvect.transform(test['Reviews_clean']).toarray()
    
    # TFIDF Vectorizer
    tfidfvect = TfidfVectorizer(analyzer="word", tokenizer=LemmaTokenizer(),
                                ngram_range=(1, 1), min_df=min_df, max_features=max_features)
    x_train_tfidf = tfidfvect.fit_transform(train['Reviews_clean']).toarray()
    x_test_tfidf = tfidfvect.transform(test['Reviews_clean']).toarray()
    
    y_train = train['Label']
    y_test = test['Label']
    
    return train, test, countvect, tfidfvect, x_train_count, x_test_count, x_train_tfidf, x_test_tfidf, y_train, y_test


# ===========================
# Feature Importance using Logistic Regression
# ===========================
def feature_importance_lr(x_train, y_train, x_test, y_test, vectorizer, top_n=200):
    lgr = LogisticRegression()
    lgr.fit(x_train, y_train)
    score = lgr.score(x_test, y_test)
    
    importantfeature = PrettyTable(["Feature", "Score"])
    for i, (feature, importance) in enumerate(zip(vectorizer.get_feature_names(), lgr.coef_[0])):
        if i <= top_n:
            importantfeature.add_row([feature, importance])
    
    print(f"Accuracy: {score}")
    print(importantfeature)


# ===========================
# MAIN EXECUTION
# ===========================
if __name__ == "__main__":
    # Preprocess data
    data = preprocess_data(df)
    
    # Vectorize with unigram CountVectorizer and TfidfVectorizer
    train, test, countvect, tfidfvect, x_train_count, x_test_count, x_train_tfidf, x_test_tfidf, y_train, y_test = vectorize_unigram(data)
    
    # Feature importance for CountVectorizer
    feature_importance_lr(x_train_count, y_train, x_test_count, y_test, countvect, top_n=200)
    
    # Feature importance for TFIDF Vectorizer
    feature_importance_lr(x_train_tfidf, y_train, x_test_tfidf, y_test, tfidfvect, top_n=100)
