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


from matplotlib import pyplot as plt
from data_cleaning import df


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
