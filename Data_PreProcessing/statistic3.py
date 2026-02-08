#-----------------------------------Get important feature by Count vectorizer-----------------------------------------


from data_cleaning import df 
from sklearn.feature_extraction.text import CountVectorizer 
import pandas as pd 
from plotly import px


#this function finds the top n most frequent n-grams(word groups) of size g from a list of texts(corpus)
#in simple words -> give it many reviews / sentences and tell it g = 1 , 2 or 3 single words , word pairs or word triplets 
#n = 10 -> give me top 10 most common  


#So in the sentiment analysis project, this is used to see: 
  #which words or phrases appears the most 
  #help with EDA  



#defines a function: 
 #corpus -> list of texts(reviews) 
 #n = how many top results you want 
 #g = size of n-gram(1 word , 2 words , etc) 
def get_top_text_ngrams(corpus , n , g):


    #creates a text counter that breaks text into chunks of g words and learns all unique such chunks in the corpus
    vec = CountVectorizer(ngram_range=(g,g)).fit(corpus)
    
    #converts text into a count matrix: 
    #rows = documents 
    #columns = n-grams 
    #Values = how many times each n-gram appears in each document
    bag_of_words = vec.transform(corpus) 
    
    #adds up counts across all documents 
    sum_words = bag_of_words.sum(axis=0) 
    
    #how many times in all documents combined 
    words_freq = [(word , sum_words[0 , idx]) for word , idx in vec.vocabulary_.items()] 
    
    #Sorts the list by frequency and then returns the top n most frequent n-grams. 
    words_freq = sorted(words_freq , key = lambda x: x[1] , reverse = True) 
    
    #Returns only the top n most frequent n-grams
    return words_freq[:n]  



#this code: finds the 20 most common words in positive reviews and draws a bar chart showing their frequencies 

#So in project terms: -> EDA/Visualization step to understand what words appear most often in positive reviews
most_common_uni = get_top_text_ngrams(df.Reviews_clean[df['Ratings']>=7] , 20 , 1) 


#converts list of pairs into a dictionary 
most_common_uni = dict(most_common_uni) 


#Creates a table like structure which stores two dataframes i.e. Common_words and Count
temp = pd.DataFrame(columns = ["Common_words" , "Count"])

#takes all the keys from the dictionary converts them into a list and stores that into a list
temp["Common_words"] = list(most_common_uni.keys()) 


#takes all the keys from the dictionary converts them into a list and stores that into a "Count" 
temp["Count"] = list(most_common_uni.values())


#uses plotly express to create a bar chart from the DataFrame temp
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















   

