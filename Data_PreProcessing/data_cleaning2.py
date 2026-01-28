# ----------------------------------------------Data Cleaning and Preprocessing------------------------------- 

from data_cleaning import STOP_WORDS , df 
import re 

#Removing special characters   
def remove_special_character(content): 
#re.sub(pattern , replacement , text) what you want to find, what you want to replace it with , where you want to do it
# regex pattern -> \[ matches a literal character (only '[' character) 
# [] brackets in regex used to define set of characters
# regex pattern -> [^&@#!] any character except & @ # ! 
# ] -> closes the set of characters 
# * -> matches zero or more characters of the previoulsly defined set
    
    return re.sub(r'\[[^&@#!]]*\]' , '' , content)  


#Removing URL's 
def remove_url(content): 

# re.sub(pattern , replacement , text) what you want to find , what you want to replace it with , where you want to do it 
# regex pattern -> "http" followed by one or more characters , stopping at the first space 
# \S -> matches any non-whitespace character 
# + -> matches one or more of the preceding token 
    return re.sub(r'http\S+' , '' , content)  


#Removing the stopwords from the text 
def remove_stopwords(content):

#clean_data -> Empty list to store the cleaned words 
#content.split() -> splits the strings by spaces and stores it in lists example ["this" , "is" , "just" , "a" , "movie"] 
#i.strip().lower() -> removes unnecessary spaces and coverts into lowercase letters
#i.strip().isalpha() -> checks if a word contains only alphabetical letters

    clean_data = []
    for i in content.split():
        if i.strip().lower() not in STOP_WORDS and i.strip().lower().isalpha(): 
            clean_data.append(i.strip().lower()) 
    return " ".join(clean_data) 

#Expansion of english contractions 

#regex -> sees won't as won\'t so r" " takes raw string 
def contraction_expansion(content):  
    content = re.sub(r"won\'t" , "would not" , content)  
    content = re.sub(r"can\'t" , "can not" , content)
    content = re.sub(r"don\'t" , "do not" , content)
    content = re.sub(r"shouldn\'t" , "should not" , content)
    content = re.sub(r"hasn\'t" , "has not" , content)
    content = re.sub(r"haven\'t" , "have not" , content)
    content = re.sub(r"weren\'t" , "were not" , content)
    content = re.sub(r"mightn\'t" , "might not" , content)
    content = re.sub(r"didn\'t" , "did not" , content)
    content = re.sub(r"n\'t" , "not" , content)
    content = re.sub(r"\'re" , "are" , content)
    content = re.sub(r"\'s" , "is" , content)
    content = re.sub(r"\'d" , "would" , content)
    content = re.sub(r"\'ll" , "will" , content)
    content = re.sub(r"\'t" , "not" , content)
    content = re.sub(r"\'ve" , "have" , content)
    content = re.sub(r"\'m" , "am" , content)     
    return content

#data preprocessing 
def data_cleaning(content): 
    content = remove_special_character(content) 
    content = remove_url(content) 
    content = contraction_expansion(content) 
    content = remove_stopwords(content) 
    return content 


#Data cleaning 
df['Reviews'] = df['Reviews'].apply(data_cleaning)  
 


    