<<<<<<< HEAD
# =========================
# Imports
# =========================
import seaborn as sns
from data_cleaning import df, STOP_WORDS
from matplotlib import pyplot as plt
from wordcloud import WordCloud

                                               
# =========================                                      
# 1. Checking for Class Imbalance  
# =========================        

# Creates a bar chart showing how many times each Ratings appears
sns.countplot(x=df['Ratings'])

# Displays the chart
plt.show()

# Prints the count of each unique rating in the console
print(df['Ratings'].value_counts())


# =========================
# 2. Visualization of Important Words from Reviews
# =========================

# This stores the reviews column (text data) into a variable
sentences = df['Reviews']

# Combine all positive reviews into one string
pos = ' '.join(map(str, sentences[df['Ratings'] >= 7]))

# Combine all negative reviews into one string
neg = ' '.join(map(str, sentences[df['Ratings'] <= 4]))


# =========================
# 3. WordCloud for Positive Reviews
# =========================

# Creates a WordCloud object:
# - width and height define image size
# - background_color is black
# - stopwords removes common words like "is", "an", "the"
# - generates the wordcloud from positive review text
pos_wordcloud = WordCloud(
    width=1500,
    height=800,
    background_color='black',
    stopwords=STOP_WORDS,
    min_font_size=15
).generate(pos)

# Create figure
plt.figure(figsize=(10, 10))

# Display the wordcloud
plt.imshow(pos_wordcloud)

# Title
plt.title('PositiveReviews')

# Remove axis lines
plt.axis('off')
=======
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

>>>>>>> 050280b1b8e3035d23fe347ec0a98704b2d7307b
