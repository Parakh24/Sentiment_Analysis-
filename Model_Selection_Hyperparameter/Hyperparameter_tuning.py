import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from Feature_Engineering_Selection.Vectoization import preprocess_data, vectorize_unigram
from Data_PreProcessing.data_cleaning import df
from sklearn.metrics import f1_score, precision_score, roc_auc_score


def hyperparamtune(classifier, param_grid, metric, verbose_value, cv):
    model = model_selection.GridSearchCV(
        estimator=classifier,
        param_grid=param_grid,
        scoring=metric,
        verbose=verbose_value,
        cv=cv
    )
    

    model.fit(x_train_tfidf, y_train)
    
    
    return model.best_estimator_, model.best_params_



df_clean, data = preprocess_data(df)

train, test, countvect, tfidfvect, x_train_count, x_test_count, x_train_tfidf, x_test_tfidf, y_train, y_test = vectorize_unigram(data)


"""
Hyperparameter Tuning of Logistic Regression 
"""
param_gd = {
    "penalty": ["l2", "l1"],
    "C": [0.01, 0.1, 1.0, 10],
    "tol": [0.0001, 0.001, 0.01],
    "max_iter": [100, 200]
}


model_7, best_param = hyperparamtune(
    LogisticRegression(solver="liblinear"),
    param_gd,
    "accuracy",
    10,
    5
)

print("Best Parameters:", best_param)


print("Precision Score on training dataset for Finetuned Logistic Regression Classifier: %s" % precision_score(y_train, model_7.predict(x_train_tfidf), average='micro'))
print("AUC Score on training dataset for Finetuned Logistic Regression Classifier: %s" % roc_auc_score(y_train, model_7.predict_proba(x_train_tfidf)[:, 1]))
f1_score_train_7 = f1_score(y_train, model_7.predict(x_train_tfidf), average="weighted")
print("F1 Score training dataset for Finetuned Logistic Regression Classifier: %s" % f1_score_train_7)

print("Precision Score on test for Finetuned Logistic Regression Classifier: %s" % precision_score(y_test, model_7.predict(x_test_tfidf), average='micro'))
print("AUC Score on test for Finetuned Logistic Regression Classifier: %s" % roc_auc_score(y_test, model_7.predict_proba(x_test_tfidf)[:, 1]))
f1_score_7 = f1_score(y_test, model_7.predict(x_test_tfidf), average="weighted")
print("F1 Score for Finetuned Logistic Regression Classifier: %s" % f1_score_7)
