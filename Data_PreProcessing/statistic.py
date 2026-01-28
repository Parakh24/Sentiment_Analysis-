# -------------------------------Basic Statistic of Review Data-------------------------------------

# Visualization of number of character in reviews 


from matplotlib import pyplot as plt
from data_cleaning import df


#figure -> the whole window , figsize(15 , 8) -> controls he size of the entire window 
#pos_ax -> axis for the left plot , neg_ax -> axis for the right plot 
#plt.subplots((1 ,2)) -> 1 row and 2 columns 
figure , (pos_ax , neg_ax) = plt.subplots(1,2,figsize=(15,8))


#df[condtion] -> give me only that rows where the condition is true 
#df[df['Ratings']>=7]['Reviews'] means give only that rows for Reviews where ratings is >= 7 
#len_pos_review -> stores the pandas series of the characters of the reviews
len_pos_review = df[df['Ratings']>=7]['Reviews'].str.len()


#hist() -> draws a histogram 
#Each bar shows how many reviews fall into a certain length range 
#color = 'range' 
pos_ax.hist(len_pos_review,color = 'green') 


#sets the title to the left plot
pos_ax.set_title('Positive reviews') 


#similar code as compared to the above
len_neg_review = df[df['Ratings']<=4]['Reviews'].str.len() 


#similar code compared to the above 
neg_ax.hist(len_neg_review , color = 'red') 


#this adds subtitle to the right 
neg_ax.set_title('Negative reviews') 


#this adds a main title for the entire figure, not just one plot 
figure.suptitle('Number of characters in reviews') 


#displays the plot
plt.show() 