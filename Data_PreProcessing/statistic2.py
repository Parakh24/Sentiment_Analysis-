#-------------------------------------Visualize average of words in reviews--------------------------------------


from matplotlib import pyplot as plt
from data_cleaning import df 
import seaborn as sns
import numpy as np  

figure , (pos_ax,neg_ax) = plt.subplots(1 , 2 , figsize = (15 , 8))

pos_word = df[df['Ratings']>=7]['Reviews'].str.split().apply(lambda x: [len(i) for i in x]) 

sns.distplot(pos_word.map(lambda x: np.mean(x)) , ax = pos_ax , color = 'green') 

pos_ax.set_title('Positive Reviews') 

neg_word = df[df['Ratings']<=4]['Reviews'].str.split().apply(lambda x: [len(i) for i in x]) 

sns.distplot(neg_word.map(lambda x: np.mean(x)) , ax = neg_ax , color = 'red') 

neg_ax.set_title('Negative reviews') 

figure.subtitle('Average word length in reviews')  
