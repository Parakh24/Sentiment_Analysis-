# Importing all dependencies 

import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
import plotly.express as px 
import nltk 
from sklearn.feature_extraction.text import CountVectorizer 
import warnings 
warnings.filterwarnings('ignore') 



#Loading of Data 
df = pd.read_csv(r'C:\Users\parak\sentimentAnalysis\Data_PreProcessing\IMDB-Dataset.csv') 


 
from nltk.corpus import stopwords           

 
stop_words = stopwords.words('english')

new_stopwords = ['movie' , 'one' , 'film' , 'would' , 'shall' , 'could' , 'might'] 

  
stop_words.extend(new_stopwords)  

stop_words.remove('not') 

STOP_WORDS = set(stop_words)   







