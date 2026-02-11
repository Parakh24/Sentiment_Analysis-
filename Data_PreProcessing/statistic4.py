#-------------------------------------------------Negative Reviews-------------------------------------------------------------

from data_cleaning import df 
from sklearn.feature_extraction.text import CountVectorizer 
import pandas as pd 
import plotly.express as px

pd.options.display.max_colwidth = 1000 







def get_top_text_ngrams(corpus , n , g):

    vec = CountVectorizer(ngram_range=(g,g)).fit(corpus)


    bag_of_words = vec.transform(corpus) 

    sum_words = bag_of_words.sum(axis=0) 

    words_freq = [(word , sum_words[0 , idx]) for word , idx in vec.vocabulary_.items()] 

    words_freq = sorted(words_freq , key = lambda x: x[1] , reverse = True) 

    return words_freq[:n] 







most_common_uni = get_top_text_ngrams(df.Reviews[df['Ratings']<=4] , 20 , 1) 

most_common_uni = dict(most_common_uni) 

temp = pd.DataFrame(columns = ["Common_words" , "Count"])

temp["Common_words"] = list(most_common_uni.keys()) 

temp["Count"] = list(most_common_uni.values())

fig = px.bar(temp , x="Count" , y="Common_words" , title = "Common Words in Negative Reviews" , orientation = 'h' , width = 700,
            height = 700 , color = 'Common_words')  

fig.show() 















most_common_uni = get_top_text_ngrams(df.Reviews[df['Ratings']<=4] , 20 , 2) 

most_common_uni = dict(most_common_uni) 

temp = pd.DataFrame(columns = ["Common_words" , "Count"])

temp["Common_words"] = list(most_common_uni.keys()) 

temp["Count"] = list(most_common_uni.values())

fig = px.bar(temp , x="Count" , y="Common_words" , title = "Common Bigrams in Negative Reviews" , orientation = 'h' , width = 700,
            height = 700 , color = 'Common_words')  

fig.show()  












most_common_uni = get_top_text_ngrams(df.Reviews[df['Ratings']<=4] , 20 , 3) 


most_common_uni = dict(most_common_uni) 

temp = pd.DataFrame(columns = ["Common_words" , "Count"])

temp["Common_words"] = list(most_common_uni.keys()) 

temp["Count"] = list(most_common_uni.values())

fig = px.bar(temp , x="Count" , y="Common_words" , title = "Common Trigrams in Negative Reviews" , orientation = 'h' , width = 700,
            height = 700 , color = 'Common_words')  

fig.show()  









most_common_uni = get_top_text_ngrams(df.Reviews[df['Ratings']<=4] , 20 , 4) 


most_common_uni = dict(most_common_uni) 

temp = pd.DataFrame(columns = ["Common_words" , "Count"])

temp["Common_words"] = list(most_common_uni.keys()) 

temp["Count"] = list(most_common_uni.values())

fig = px.bar(temp , x="Count" , y="Common_words" , title = "Common 4-grams in Negative Reviews" , orientation = 'h' , width = 700,
            height = 700 , color = 'Common_words')  


fig.show() 










most_common_uni = get_top_text_ngrams(df.Reviews[df['Ratings']<=4] , 20 , 5) 


most_common_uni = dict(most_common_uni) 

temp = pd.DataFrame(columns = ["Common_words" , "Count"])

temp["Common_words"] = list(most_common_uni.keys()) 

temp["Count"] = list(most_common_uni.values())

fig = px.bar(temp , x="Count" , y="Common_words" , title = "Common 5-grams in Negative Reviews" , orientation = 'h' , width = 700,
            height = 700 , color = 'Common_words')  


fig.show()  






pd.options.display.max_colwidth = 1000 

df[['Reviews' , 'Ratings' , 'Movies']][df(['Ratings']<=4)&(df['Reviews'].str.contains("good|great"))].head(100) 















