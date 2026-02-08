#Feature Engineering 
 
#Mapping Rating Data to Binary Label 1 (+ve) if rating >= 7 and 0 (-ve) if rating <=4 and 2 (neutral) 

#importing Dependencies for Feature Engineering 

import sys 
import os 
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier 
import pandas as pd 
from prettytable import PrettyTable 

 
from Data_PreProcessing.data_cleaning import df 




    






