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
from sklearn.feature_selection import chi2
import numpy as np
import nltk

# ===========================
# Download NLTK resources (needed once)
# ===========================
nltk.download('punkt')
nltk.download('wordnet')


# ===========================
# Data Preprocessing
# ===========================
def preprocess_data(df):
    # Labeling (use integers, not strings)
    df['Label'] = df['Ratings'].apply(lambda x: 1 if x >= 5 else (0 if x <= 4 else 2))
    df = df[df['Label'] < 2]
    data = df[['Reviews', 'Label']]
    print("Label distribution:\n", data['Label'].value_counts())
    return data


# ===========================
# Lemmatizer Tokenizer
# ===========================
class LemmaTokenizer(object):
    def __init__(self):
        self.wordnetlemma = WordNetLemmatizer()

    def __call__(self, reviews):
        return [self.wordnetlemma.lemmatize(word) for word in word_tokenize(reviews)]


# ===========================
# Train-Test Split & Vectorization
# ===========================
def vectorize_trigram(data, ngram_range=(3, 3), max_features=500, min_df=10):
    train, test = train_test_split(data, test_size=0.3, random_state=42, shuffle=True)
    
    # Count Vectorizer
    countvect = CountVectorizer(
        analyzer="word",
        tokenizer=LemmaTokenizer(),
        ngram_range=ngram_range,
        min_df=min_df,
        max_features=max_features
    )
    x_train_count = countvect.fit_transform(train['Reviews']).toarray()
    x_test_count = countvect.transform(test['Reviews']).toarray()
    
    # TFIDF Vectorizer
    tfidfvect = TfidfVectorizer(
        analyzer="word",
        tokenizer=LemmaTokenizer(),
        ngram_range=ngram_range,
        min_df=min_df,
        max_features=max_features
    )
    x_train_tfidf = tfidfvect.fit_transform(train['Reviews']).toarray()
    x_test_tfidf = tfidfvect.transform(test['Reviews']).toarray()
    
    y_train = train['Label']
    y_test = test['Label']
    
    return train, test, countvect, tfidfvect, x_train_count, x_test_count, x_train_tfidf, x_test_tfidf, y_train, y_test


# ===========================
# Feature Importance using Logistic Regression
# ===========================
def feature_importance_lr(x_train, y_train, x_test, y_test, vectorizer, top_n=200):
    lgr = LogisticRegression(max_iter=500)
    lgr.fit(x_train, y_train)
    score = lgr.score(x_test, y_test)
    
    importantfeature = PrettyTable(["Feature", "Score"])
    
    feature_names = vectorizer.get_feature_names_out()
    
    for i, (feature, importance) in enumerate(zip(feature_names, lgr.coef_[0])):
        if i < top_n:
            importantfeature.add_row([feature, importance])
    
    print(f"Accuracy: {score}")
    print(importantfeature)


# ===========================
# Feature Selection with Chi-Squared
# ===========================
def chi2_feature_selection(x_train_tfidf, train, tfidfvect, N=5000):
    Number = 1
    
    for category in train['Label'].unique():
        features_chi2 = chi2(x_train_tfidf, train['Label'] == category)
        indices = np.argsort(features_chi2[0])
        feature_names = np.array(tfidfvect.get_feature_names_out())[indices]
        
        unigrams = [x for x in feature_names if len(x.split(' ')) == 1]
        bigrams = [x for x in feature_names if len(x.split(' ')) == 2]
        trigrams = [x for x in feature_names if len(x.split(' ')) == 3]
        
        print("%s. %s :" % (Number, category))
        print("\t# Unigrams :\n\t. %s" % ('\n\t. '.join(unigrams[-N:])))
        print("\t# Bigrams :\n\t. %s" % ('\n\t. '.join(bigrams[-N:])))
        print("\t# Trigrams :\n\t. %s" % ('\n\t. '.join(trigrams[-N:])))
        
        Number += 1


# ===========================
# MAIN EXECUTION
# ===========================
if __name__ == "__main__":
    data = preprocess_data(df)

    # Trigram vectorization and Logistic Regression
    train, test, countvect, tfidfvect, x_train_count, x_test_count, x_train_tfidf, x_test_tfidf, y_train, y_test = vectorize_trigram(data, ngram_range=(3,3))
    feature_importance_lr(x_train_count, y_train, x_test_count, y_test, countvect)
    feature_importance_lr(x_train_tfidf, y_train, x_test_tfidf, y_test, tfidfvect)

    # 4-gram vectorization and Logistic Regression
    train, test, countvect, tfidfvect, x_train_count, x_test_count, x_train_tfidf, x_test_tfidf, y_train, y_test = vectorize_trigram(data, ngram_range=(4,4))
    feature_importance_lr(x_train_count, y_train, x_test_count, y_test, countvect)
    feature_importance_lr(x_train_tfidf, y_train, x_test_tfidf, y_test, tfidfvect)

    # Chi-squared feature selection
    chi2_feature_selection(x_train_tfidf, train, tfidfvect)
