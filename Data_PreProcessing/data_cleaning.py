# Importing all dependencies 

import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
import plotly.express as px 
import nltk 
from sklearn.feature_extraction.text import CountVectorizer 
from wordcloud import WordCloud , STOPWORDS 
from nltk.stem import WordNetLemmatizer 
from nltk.tokenize import word_tokenize 
import re , string , unicodedata 
from sklearn.metrics import classification_report , confusion_matrix , accuracy_score , f1_score 
from sklearn.model_selection import train_test_split 
from string import punctuation  
from nltk import pos_tag 
from nltk.corpus import wordnet 
import warnings 
warnings.filterwarnings('ignore') 



#Loading of Data 
df = pd.read_csv('IMDB Dataset.csv') 





# Data Cleaning and Preprocessing Functions 
## Customize stopword as per the data 

#this imports nltk's built-in list of common ENglish stopwords
from nltk.corpus import stopwords           

#stop_words is a list containing standard English stopwords
stop_words = stopwords.words('english')

#these are the external stopwords we want to add to the standard list since it might not be relevant ffor our analysis
new_stopwords = ['movie' , 'one' , 'film' , 'would' , 'shall' , 'could' , 'might'] 


#extend the stop_words list by adding new_stopwords to it
stop_words.extend(new_stopwords) 

#not word is removed from stopwords list to retain the negation contect in sentiment analysis
stop_words.remove('not') 

#list is converted to sets for faster processing and removing duplicates
STOP_WORDS = set(stop_words)   


print(df) 




