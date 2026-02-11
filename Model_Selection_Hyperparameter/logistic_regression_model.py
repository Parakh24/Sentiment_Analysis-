import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn.linear_model import LogisticRegression
from Feature_Engineering_Selection.Vectoization import vectorize_unigram, preprocess_data
from Data_PreProcessing.data_cleaning import df
from sklearn.metrics import f1_score, precision_score, roc_auc_score


df_clean, data = preprocess_data(df)

train, test, countvect, tfidfvect, x_train_count, x_test_count, x_train_tfidf, x_test_tfidf, y_train, y_test = vectorize_unigram(data)


model_1 = LogisticRegression(max_iter=1000 , class_weight="balanced")
model_1.fit(x_train_tfidf, y_train)

print("Precision Score on training dateset for Logistic Regression: %s" % precision_score(y_train, model_1.predict(x_train_tfidf), average='micro'))
print("AUC Score on training dateset for Logistic Regression: %s" % roc_auc_score(y_train, model_1.predict_proba(x_train_tfidf)[:, 1]))
f1_score_train_1 = f1_score(y_train, model_1.predict(x_train_tfidf), average="weighted")
print("F1 Score training dateset for Logistic Regression: %s" % f1_score_train_1)

print("Precision Score on test for Logistic Regression: %s" % precision_score(y_test, model_1.predict(x_test_tfidf), average='micro'))
print("AUC Score on test for Logistic Regression: %s" % roc_auc_score(y_test, model_1.predict_proba(x_test_tfidf)[:, 1]))
f1_score_1 = f1_score(y_test, model_1.predict(x_test_tfidf), average="weighted")
print("F1 Score for Logistic Regression: %s" % f1_score_1)
