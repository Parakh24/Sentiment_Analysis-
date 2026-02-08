<<<<<<< HEAD
# ------------------------------- Basic Statistic of Review Data -------------------------------------

# Visualization of number of characters in reviews
# This code is trying to answer:
# Do positive and negative reviews differ in length?

# Separates the positive and negative reviews
# Measures how long each review is
# Draws two histograms:
#   - One for positive reviews (green)
#   - One for negative reviews (red)
# Lets you visually compare them

# This helps us understand writing behavior in the dataset
# Why useful -> because this analysis tells the user:
# Are positive reviews usually longer or are negative reviews usually shorter?
# It can influence Feature Engineering
=======
# -------------------------------Basic Statistic of Review Data-------------------------------------

# Visualization of number of character in reviews 
>>>>>>> 050280b1b8e3035d23fe347ec0a98704b2d7307b


from matplotlib import pyplot as plt
from data_cleaning import df


<<<<<<< HEAD
# Create two plots side by side and make one big figure with two graphs next to each other
figure, (pos_ax, neg_ax) = plt.subplots(1, 2, figsize=(15, 8))


# Takes only reviews with rating >= 7 and stores the total characters of each review
len_pos_review = df[df['Ratings'] >= 7]['Reviews'].str.len()


# Draw a histogram showing how long positive reviews are
pos_ax.hist(len_pos_review, color='green')
pos_ax.set_title('Positive reviews')


# Take only reviews with ratings <= 4 and store the character count in each review
len_neg_review = df[df['Ratings'] <= 4]['Reviews'].str.len()


# Histogram of lengths of negative reviews
neg_ax.hist(len_neg_review, color='red')
neg_ax.set_title('Negative reviews')


# Plot the big title on top and show the plots
figure.suptitle('Number of characters in reviews')
plt.show()
=======
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
>>>>>>> 050280b1b8e3035d23fe347ec0a98704b2d7307b
