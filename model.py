import joblib
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from Data_PreProcessing.data_cleaning import df
from Feature_Engineering_Selection.Vectoization import preprocess_data

# Lemmatizer Tokenizer
class LemmaTokenizer(object):
    def __init__(self):
        self.wordnetlemma = WordNetLemmatizer()

    def __call__(self, reviews):
        # Lowercase + remove non-letters
        reviews = reviews.lower()
        reviews = re.sub(r"[^a-zA-Z\s]", "", reviews)
        tokens = word_tokenize(reviews)
        return [self.wordnetlemma.lemmatize(word) for word in tokens]

# Preprocess
df_clean, data = preprocess_data(df)

X = data["Reviews"]
y = data["Label"]

# Build pipeline with unigrams, bigrams, and trigrams
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        tokenizer=LemmaTokenizer(),
        ngram_range=(1, 3),   # unigrams, bigrams, trigrams
        min_df=10,
        max_features=5000
    )),
    ("clf", LogisticRegression(max_iter=1000))
])

# Train
model.fit(X, y)

# Save
joblib.dump(model, "sentiment_model.pkl")

print("Model saved as sentiment_model.pkl")
