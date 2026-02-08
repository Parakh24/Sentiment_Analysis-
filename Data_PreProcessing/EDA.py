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
