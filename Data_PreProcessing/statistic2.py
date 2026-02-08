#-------------------------------------Visualize average of words in reviews--------------------------------------


from matplotlib import pyplot as plt
from data_cleaning import df 
import seaborn as sns
import numpy as np  


#this code compares positive and negative reviews by visualizing the distribution of the average word length
figure , (pos_ax,neg_ax) = plt.subplots(1 , 2 , figsize = (15 , 8))


#Filters positive reviews (ratings ≥ 7)
#Takes the Reviews column
#Splits each review into words
#Converts each word into its length
#Result: for each review, you get a list like [4, 2, 7, 5, ...]
pos_word = df[df['Ratings']>=7]['Reviews'].str.split().apply(lambda x: [len(i) for i in x]) 


#Computes the average word length per review
#Plots the distribution of those averages on pos_ax
#Colors it green
sns.distplot(pos_word.map(lambda x: np.mean(x)) , ax = pos_ax , color = 'green') 
pos_ax.set_title('Positive Reviews') 


#similar code as before
neg_word = df[df['Ratings']<=4]['Reviews'].str.split().apply(lambda x: [len(i) for i in x]) 


#computes average word length per negative review 
#plots its distribution on neg_ax 
#Colors it red 
sns.distplot(neg_word.map(lambda x: np.mean(x)) , ax = neg_ax , color = 'red') 
neg_ax.set_title('Negative reviews') 

#sets the title for the overall plot 
figure.suptitle('Average word length in reviews')  

#show the plot 
plt.show() 

