# -------------------------------Basic Statistic of Review Data-------------------------------------

# Visualization of number of character in reviews 


from matplotlib import pyplot as plt
from data_cleaning import df



figure , (pos_ax , neg_ax) = plt.subplots(1,2,figsize=(15,8))

len_pos_review = df[df['Ratings']>=7]['Reviews'].str.len()

pos_ax.hist(len_pos_review,color = 'green') 

pos_ax.set_title('Positive reviews') 

len_neg_review = df[df['Ratings']<=4]['Reviews'].str.len() 

neg_ax.hist(len_neg_review , color = 'red') 

neg_ax.set_title('Negative reviews') 

figure.subtitle('Number of characters in reviews') 

plt.show() 