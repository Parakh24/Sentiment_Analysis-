#-------------------------------------------------Negative Reviews-------------------------------------------------------------

from data_cleaning import df 
from sklearn.feature_extraction.text import CountVectorizer 
import pandas as pd 
from plotly import px

pd.options.display.max_colwidth = 1000 

<<<<<<< HEAD
=======
df[['Reviews' , 'Ratings' , 'Movies']][df(['Ratings']>=7)&(df['Reviews'].str.contains("blah blah blah | la la la la | mario mario mario mario"))] 



>>>>>>> 050280b1b8e3035d23fe347ec0a98704b2d7307b


def get_top_text_ngrams(corpus , n , g):

    vec = CountVectorizer(ngram_range=(g,g)).fit(corpus)
<<<<<<< HEAD
    bag_of_words = vec.transform(corpus) 
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word , sum_words[0 , idx]) for word , idx in vec.vocabulary_.items()] 
    words_freq = sorted(words_freq , key = lambda x: x[1] , reverse = True) 
=======

    bag_of_words = vec.transform(corpus) 

    sum_words = bag_of_words.sum(axis=0) 

    words_freq = [(word , sum_words[0 , idx]) for word , idx in vec.vocabulary_.items()] 

    words_freq = sorted(words_freq , key = lambda x: x[1] , reverse = True) 

>>>>>>> 050280b1b8e3035d23fe347ec0a98704b2d7307b
    return words_freq[:n] 





<<<<<<< HEAD
#this code is almost similar to that of the previous code just this is the case for Negative Reviews for one-gram
most_common_uni = get_top_text_ngrams(df.Reviews_clean[df['Ratings']<=4] , 20 , 1) 
most_common_uni = dict(most_common_uni) 
temp = pd.DataFrame(columns = ["Common_words" , "Count"])
temp["Common_words"] = list(most_common_uni.keys()) 
temp["Count"] = list(most_common_uni.values())
fig = px.bar(temp , x="Count" , y="Common_words" , title = "Common Words in Negative Reviews" , orientation = 30 , width = 700,
            height = 700 , color = 'Common words')  
=======

most_common_uni = get_top_text_ngrams(df.Reviews_clean[df['Ratings']<=4] , 20 , 1) 

most_common_uni = dict(most_common_uni) 

temp = pd.DataFrame(columns = ["Common_words" , "Count"])

temp["Common_words"] = list(most_common_uni.keys()) 

temp["Count"] = list(most_common_uni.values())

fig = px.bar(temp , x="Count" , y="Common_words" , title = "Common Words in Negative Reviews" , orientation = 30 , width = 700,
            height = 700 , color = 'Common words')  

>>>>>>> 050280b1b8e3035d23fe347ec0a98704b2d7307b
fig.show() 










<<<<<<< HEAD
most_common_uni = get_top_text_ngrams(df.Reviews_clean[df['Ratings']<=4] , 20 , 2) 
most_common_uni = dict(most_common_uni) 
temp = pd.DataFrame(columns = ["Common_words" , "Count"])
temp["Common_words"] = list(most_common_uni.keys()) 
temp["Count"] = list(most_common_uni.values())
fig = px.bar(temp , x="Count" , y="Common_words" , title = "Common Bigrams in Negative Reviews" , orientation = 30 , width = 700,
            height = 700 , color = 'Common words')  
=======




most_common_uni = get_top_text_ngrams(df.Reviews_clean[df['Ratings']<=4] , 20 , 2) 

most_common_uni = dict(most_common_uni) 

temp = pd.DataFrame(columns = ["Common_words" , "Count"])

temp["Common_words"] = list(most_common_uni.keys()) 

temp["Count"] = list(most_common_uni.values())

fig = px.bar(temp , x="Count" , y="Common_words" , title = "Common Bigrams in Negative Reviews" , orientation = 30 , width = 700,
            height = 700 , color = 'Common words')  

>>>>>>> 050280b1b8e3035d23fe347ec0a98704b2d7307b
fig.show()  












most_common_uni = get_top_text_ngrams(df.Reviews_clean[df['Ratings']<=4] , 20 , 3) 
<<<<<<< HEAD
most_common_uni = dict(most_common_uni) 
temp = pd.DataFrame(columns = ["Common_words" , "Count"])
temp["Common_words"] = list(most_common_uni.keys()) 
temp["Count"] = list(most_common_uni.values())
fig = px.bar(temp , x="Count" , y="Common_words" , title = "Common Trigrams in Negative Reviews" , orientation = 30 , width = 700,
            height = 700 , color = 'Common words')  
=======

most_common_uni = dict(most_common_uni) 

temp = pd.DataFrame(columns = ["Common_words" , "Count"])

temp["Common_words"] = list(most_common_uni.keys()) 

temp["Count"] = list(most_common_uni.values())

fig = px.bar(temp , x="Count" , y="Common_words" , title = "Common Trigrams in Negative Reviews" , orientation = 30 , width = 700,
            height = 700 , color = 'Common words')  

>>>>>>> 050280b1b8e3035d23fe347ec0a98704b2d7307b
fig.show()  









most_common_uni = get_top_text_ngrams(df.Reviews_clean[df['Ratings']<=4] , 20 , 4) 
<<<<<<< HEAD
most_common_uni = dict(most_common_uni) 
temp = pd.DataFrame(columns = ["Common_words" , "Count"])
temp["Common_words"] = list(most_common_uni.keys()) 
temp["Count"] = list(most_common_uni.values())
fig = px.bar(temp , x="Count" , y="Common_words" , title = "Common 4-grams in Negative Reviews" , orientation = 30 , width = 700,
            height = 700 , color = 'Common words')  
=======

most_common_uni = dict(most_common_uni) 

temp = pd.DataFrame(columns = ["Common_words" , "Count"])

temp["Common_words"] = list(most_common_uni.keys()) 

temp["Count"] = list(most_common_uni.values())

fig = px.bar(temp , x="Count" , y="Common_words" , title = "Common 4-grams in Negative Reviews" , orientation = 30 , width = 700,
            height = 700 , color = 'Common words')  

>>>>>>> 050280b1b8e3035d23fe347ec0a98704b2d7307b
fig.show() 










most_common_uni = get_top_text_ngrams(df.Reviews_clean[df['Ratings']<=4] , 20 , 5) 
<<<<<<< HEAD
most_common_uni = dict(most_common_uni) 
temp = pd.DataFrame(columns = ["Common_words" , "Count"])
temp["Common_words"] = list(most_common_uni.keys()) 
temp["Count"] = list(most_common_uni.values())
fig = px.bar(temp , x="Count" , y="Common_words" , title = "Common 5-grams in Negative Reviews" , orientation = 30 , width = 700,
            height = 700 , color = 'Common words')  
=======

most_common_uni = dict(most_common_uni) 

temp = pd.DataFrame(columns = ["Common_words" , "Count"])

temp["Common_words"] = list(most_common_uni.keys()) 

temp["Count"] = list(most_common_uni.values())

fig = px.bar(temp , x="Count" , y="Common_words" , title = "Common 5-grams in Negative Reviews" , orientation = 30 , width = 700,
            height = 700 , color = 'Common words')  

>>>>>>> 050280b1b8e3035d23fe347ec0a98704b2d7307b
fig.show()  





<<<<<<< HEAD
=======
pd.options.display.max_colwidth = 1000 

df[['Reviews' , 'Ratings' , 'Movies']][df(['Ratings']<=4)&(df['Reviews_clean'].str.contains("good|great"))].head(100) 




>>>>>>> 050280b1b8e3035d23fe347ec0a98704b2d7307b










