import seaborn as sns 
from data_cleaning import df , STOP_WORDS
from matplotlib import pyplot as plt
from wordcloud import WordCloud

#checking for class Imbalance


#Creates a bar chart showing how many times each Ratings appears
sns.countplot(x=df['Ratings']) 

#Displays the chart
plt.show()  

#It prints the count of each unique rating in the console 
print(df['Ratings'].value_counts()) 


#-----------------------Visualization of Important Words from Positive Reviews-------------------------------------


#this stores the reviews columns i.e. text data into a variable named sentences
sentences = df['Reviews'] 

#sentences[df['rating']>=7] means sentence variable selects only those rows whose ratings >= 7
# map(str , sentences[df[ratings]>=7]) -> converts everything into the string 
# ''.join -> creates it into a single string and stores it in pos i.e. positive reviews
pos = ' '.join(map(str , sentences[df['Ratings']>=7]))


#similar but stores the negative reviews
neg = ' '.join(map(str , sentences[df['Ratings']<=4]))  



#creates a WordCloud Object with width and height representes size of the image background_color as black background 
#stopwords = STOP_WORDS which removes common words like is an the and creates the wordcloud from positive review text
pos_wordcloud = WordCloud(width = 1500 , height = 800 , background_color = 'black',
                         stopwords = STOP_WORDS , min_font_size = 15).generate(pos) 



plt.figure(figsize = (10 , 10)) 

#displays the imshow(pos_wordcloud) 
plt.imshow(pos_wordcloud) 

#title(positive reviews)
plt.title('PositiveReviews') 

#removes axis lines i.e. x and y axis
plt.axis('off')

