import seaborn as sns 
from data_cleaning import df , STOP_WORDS
from matplotlib import pyplot as plt
from wordcloud import WordCloud , STOPWORDS 

#checking for class Imbalance


#Creates a bar chart showing how many times each rating appears
sns.countplot(x=df['Ratings']) 

#Displays the chart
plt.show()  

#It prints the count of each unique rating in the console 
print(df['Ratings'].value_counts()) 


#-----------------------Visualization of Important Words from Positive Reviews-------------------------------------

sentences = df['review'] 
pos = ' '.join(map(str , sentences[df['rating']>=7]))
neg = ' '.join(map(str , sentences[df['rating']<=4]))  


pos_wordcloud = WordCloud(width = 1500 , height = 800 , background_color = 'black',
                         stopwords = STOP_WORDS , min_font_size = 15).generate(pos) 

plt.figure(figsize = (10 , 10)) 
plt.imshow(pos_wordcloud) 
plt.title('Positive Reviews')    
plt.axis('off')


