#-----------------------------------Get important feature by Count vectorizer-----------------------------------------


from data_cleaning import df 
from sklearn.feature_extraction.text import CountVectorizer 
import pandas as pd 
from plotly import px


def get_top_text_ngrams(corpus , n , g):

    vec = CountVectorizer(ngram_range=(g,g)).fit(corpus)

    bag_of_words = vec.transform(corpus) 

    sum_words = bag_of_words.sum(axis=0) 

    words_freq = [(word , sum_words[0 , idx]) for word , idx in vec.vocabulary_.items()] 

    words_freq = sorted(words_freq , key = lambda x: x[1] , reverse = True) 

    return words_freq[:n]  





most_common_uni = get_top_text_ngrams(df.Reviews_clean[df['Ratings']>=7] , 20 , 1) 

most_common_uni = dict(most_common_uni) 

temp = pd.DataFrame(columns = ["Common_words" , "Count"])

temp["Common_words"] = list(most_common_uni.keys()) 

temp["Count"] = list(most_common_uni.values())

fig = px.bar(temp , x="Count" , y="Common_words" , title = "Common Words in Positive Reviews" , orientation = 30 , width = 700,
            height = 700 , color = 'Common words')  

fig.show() 







most_common_uni = get_top_text_ngrams(df.Reviews_clean[df['Ratings']>=7],20,2)

most_common_uni = dict(most_common_uni) 

temp = pd.DataFrame(columns = ["Common_words" , 'Count'])  

temp["Common_words"] = list(most_common_uni.keys()) 

temp["Count"] = list(most_common_uni.values()) 

fig = px.bar(temp , x="Count" , y = "Common_words" , title = "Common bigram in Positive Review" , orientation = 30 , width = 700,
            height = 700 , color = 'Common words') 

fig.show() 






most_common_uni = get_top_text_ngrams(df.Reviews_clean[df['Ratings']>=7],20,3)

most_common_uni = dict(most_common_uni) 

temp = pd.DataFrame(columns = ["Common_words" , 'Count'])  

temp["Common_words"] = list(most_common_uni.keys()) 

temp["Count"] = list(most_common_uni.values()) 

fig = px.bar(temp , x="Count" , y = "Common_words" , title = "Common bigram in Positive Review" , orientation = 30 , width = 700,
            height = 700 , color = 'Common words') 

fig.show()  





most_common_uni = get_top_text_ngrams(df.Reviews_clean[df['Ratings']>=7],20,4)

most_common_uni = dict(most_common_uni) 

temp = pd.DataFrame(columns = ["Common_words" , 'Count'])  

temp["Common_words"] = list(most_common_uni.keys()) 

temp["Count"] = list(most_common_uni.values()) 

fig = px.bar(temp , x="Count" , y = "Common_words" , title = "Common bigram in Positive Review" , orientation = 30 , width = 700,
            height = 700 , color = 'Common words') 

fig.show()   








most_common_uni = get_top_text_ngrams(df.Reviews_clean[df['Ratings']>=7],20,5)

most_common_uni = dict(most_common_uni) 

temp = pd.DataFrame(columns = ["Common_words" , 'Count'])  

temp["Common_words"] = list(most_common_uni.keys()) 

temp["Count"] = list(most_common_uni.values()) 

fig = px.bar(temp , x="Count" , y = "Common_words" , title = "Common bigram in Positive Review" , orientation = 30 , width = 700,
            height = 700 , color = 'Common words') 

fig.show() 















   

