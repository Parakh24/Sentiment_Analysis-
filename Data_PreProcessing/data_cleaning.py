# Importing all dependencies 

 

import warnings 
warnings.filterwarnings('ignore') 
import os
import pandas as pd

base_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_dir, "IMDB-Dataset.csv")

df = pd.read_csv(file_path, engine="python", on_bad_lines="skip", encoding="utf-8")


print(df.columns)




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







