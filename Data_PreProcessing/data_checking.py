
from data_cleaning2 import df 

#Data Overview 

#tells you how many missing values are there in each column
# df.isna() -> tells you how many missing values are there in terms of boolean variables i.e. True for Nan and False for filled
#df.isna().sum() -> sums up how many True boolean values there for each column 
print(df.isna().sum())  


#this gives you the basic statistical values about the texts such as mean , median , std
df['Reviews'].describe()


# %s it's a placeholder for a value that will be inserted into the string. 
#df.review is the review column in my dataframe and .nunique() displays the number of unique values in the review column
print('Unique reviews:%s' % df.Reviews.nunique())

 
