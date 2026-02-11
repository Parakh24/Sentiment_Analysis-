import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))




import shap 
from Feature_Engineering_Selection.Vectoization import  vectorize_unigram 
from Feature_Engineering_Selection.Vectoization import preprocess_data 
from Data_PreProcessing.data_cleaning import df
from sklearn.linear_model import LogisticRegression 



shap.initjs() 

df_clean , data = preprocess_data(df)
train, test, countvect, tfidfvect, x_train_count, x_test_count, x_train_tfidf, x_test_tfidf, y_train, y_test = vectorize_unigram(data)

model_1 = LogisticRegression(max_iter=1000)
model_1.fit(x_train_tfidf, y_train)

explainer = shap.LinearExplainer(model_1, x_train_tfidf)
shap_values = explainer(x_test_tfidf)

shap.plots.beeswarm(shap_values)
 

