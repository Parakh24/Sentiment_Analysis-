import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn.model_selection import train_test_split 
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from Data_PreProcessing.data_cleaning import df 
from nltk.stem import WordNetLemmatizer 
from nltk import word_tokenize  
from sklearn.linear_model import LogisticRegression 
from prettytable import PrettyTable 
import nltk 
import re 

# ===========================
# Download NLTK resources
# ===========================
nltk.download('punkt')
nltk.download('wordnet')


# ===========================
# Data Preprocessing
# ===========================
def preprocess_data(df):
    # If Label already exists, DO NOT overwrite it
    if 'Label' not in df.columns:
        df['Label'] = df['Ratings'].apply(lambda x: 1 if x >= 4 else 0)

    data = df[['Reviews', 'Label']].dropna()
    print("Label distribution:\n", data['Label'].value_counts())
    return df, data


# ===========================
# Lemmatization Tokenizer
# ===========================
class LemmaTokenizer(object):
    def __init__(self):
        self.wordnetlemma = WordNetLemmatizer()

    def __call__(self, reviews):
        # Lowercase + remove non-letters
        reviews = reviews.lower()
        reviews = re.sub(r"[^a-zA-Z\s]", "", reviews)

        tokens = word_tokenize(reviews)
        return [self.wordnetlemma.lemmatize(word) for word in tokens]


# ===========================
# Train-Test Split & Vectorization
# ===========================
def vectorize_unigram(data, min_df=1, max_features=5000):
    train, test = train_test_split(data, test_size=0.3, random_state=42, shuffle=True, stratify=data['Label'])
    
    # Count Vectorizer
    countvect = CountVectorizer(
        analyzer="word",
        tokenizer=LemmaTokenizer(),
        ngram_range=(1, 2),
        min_df=min_df,
        max_features=max_features
    )
    x_train_count = countvect.fit_transform(train['Reviews'])
    x_test_count = countvect.transform(test['Reviews'])
    
    # TFIDF Vectorizer
    tfidfvect = TfidfVectorizer(
        analyzer="word",
        tokenizer=LemmaTokenizer(),
        ngram_range=(1, 2),
        min_df=1,
        max_features=max_features
    )
    x_train_tfidf = tfidfvect.fit_transform(train['Reviews'])
    x_test_tfidf = tfidfvect.transform(test['Reviews'])
    
    y_train = train['Label']
    y_test = test['Label']
    
    return train, test, countvect, tfidfvect, x_train_count, x_test_count, x_train_tfidf, x_test_tfidf, y_train, y_test


# ===========================
# Feature Importance using Logistic Regression
# ===========================
def feature_importance_lr(x_train, y_train, x_test, y_test, vectorizer, top_n=200):
    lgr = LogisticRegression(max_iter=1000, class_weight="balanced")
    lgr.fit(x_train, y_train)
    score = lgr.score(x_test, y_test)
    
    importantfeature = PrettyTable(["Feature", "Weight"])
    
    feature_names = vectorizer.get_feature_names_out()
    
    coefs = lgr.coef_[0]
    top_idx = abs(coefs).argsort()[::-1][:top_n]

    for i in top_idx:
        importantfeature.add_row([feature_names[i], coefs[i]])
    
    print(f"Accuracy: {score}")
    print(importantfeature)


# ===========================
# MAIN EXECUTION
# ===========================
if __name__ == "__main__":
    # Preprocess data
    df, data = preprocess_data(df)
    
    # Vectorize with unigram CountVectorizer and TfidfVectorizer
    train, test, countvect, tfidfvect, x_train_count, x_test_count, x_train_tfidf, x_test_tfidf, y_train, y_test = vectorize_unigram(data)
    
    # Feature importance for CountVectorizer
    feature_importance_lr(x_train_count, y_train, x_test_count, y_test, countvect, top_n=200)
    
    # Feature importance for TFIDF Vectorizer
    feature_importance_lr(x_train_tfidf, y_train, x_test_tfidf, y_test, tfidfvect, top_n=100)

    # Quick sanity test
    print("Test sentence prediction (TFIDF):")
    test_sentences = ["I loved this movie", "This was terrible and boring"]
    print(tfidfvect.transform(test_sentences))
