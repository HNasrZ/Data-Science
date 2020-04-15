#############
# Libraries #
#############

import numpy as np
import pandas as pd
import time
from nytimesarticle import articleAPI
import re
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk import FreqDist
import requests
from bs4 import BeautifulSoup
import nltk.stem
from scipy import sparse


##############
# Functions #
#############

# the input for this function is the HTML source of the article
#it returns a dictionary containing the publication date, type of the article(news,letter,opinion,etc), url of the article and the total number of words in that article.

def parse_articles(articles):
    
    news = []
    
    for i in articles['response']['docs']:
        
        dic = {}
        dic['date'] = i['pub_date'][0:10] # cutting time of the day.
        dic['atype'] = i['type_of_material']
        dic['url'] = i['web_url']
        dic['word_count'] = int(i['word_count'])
        news.append(dic)
        
    return news


# the inputs for this function are the API of New York Times and a period of time (start-year and end-year) for which the user whishes to extract the urls of the articles
#the output of this fucntion is a pd.DataFrame containing publication date,url of the articles and the total number of words in each article.

def get_articles_url(api, start_year, end_year):
    
    all_articles = []
    year = start_year
    
    print('Retrieving articles URL...'),
    
    #Loop through all years of interest
    while year <= end_year:
        
       
        # To keep any potential error in obtaining the HTML from stopping the loop a try/except was used.
        # This loop goes through the pages in NYT website (the first 100 pages)
        for i in range(0,100):

            try:
                # Calling API method
                articles = api.search(
                    fq = str({'source':['The New York Times']}), 
                    begin_date = int(str(year) + '0101'), 
                    end_date = int(str(year) + '1231'), 
                    page = i)
                # Checking if the page is empty
                if articles['response']['docs'] == []: break
                
                articles = parse_articles(articles)
                all_articles = all_articles + articles
                
            except Exception:

                pass
            
            # Avoid overwhelming the API
            time.sleep(6)
            
        year += 1
    
    # Copy all articles on the list to a Pandas dataframe
    articles_df = pd.DataFrame(all_articles)
    
    # Filtering out non-news articles and remove 'atype' column
    articles_df = articles_df.drop(articles_df[articles_df.atype != 'News'].index)    # These lines are for cleaning the data
    articles_df.drop('atype', axis = 1, inplace = True)
    
    # Discard non-working links (their number of word_count is 0).
    articles_df = articles_df[articles_df.word_count != 0]
    articles_df = articles_df.reset_index(drop = True)
    
    print('Done!')
    
    return(articles_df)




def scrape_articles_text(articles_df):
    
    # Getting rid of the pandas' warning when the code attempts to make changes to a slice of a DataFrame not all of it
    pd.options.mode.chained_assignment = None
    
    articles_df['article_text'] = 'NaN'
    
    # defining session operator to get the text of the URL's HTML source
    session = requests.Session()
    
    print('Scraping articles body text...'),
    
    
    #loop over all the articles to obtain the text of each article
    for j in range(0, len(articles_df)):
        
        url = articles_df['url'][j]
        req = session.get(url)
        soup = BeautifulSoup(req.text, 'lxml')

        
        # Articles' body were found under different classes
        paragraph_tags = soup.find_all('p', class_= 'story-body-text story-content')
        if paragraph_tags == []:
            paragraph_tags = soup.find_all('p', class_= 'css-1ygdjhk evys1bk0')
            if paragraph_tags == []:
                paragraph_tags = soup.find_all('p', itemprop = 'articleBody')

            

        # Put together all text from HTML p tags
        article = ''
        for p in paragraph_tags:
            article = article + ' ' + p.get_text()

        # Clean article replacing unicode characters.(replacing '‘','’','“'and'”' with ' and '')
        article = article.replace(u'\u2018', u"'").replace(u'\u2019', u"'").replace(u'\u201c', u'"').replace(u'\u201d', u'"')

        # Copying article's content to the dataframe
        articles_df['article_text'][j] = article
        
    print('Done!')
    
    return articles_df


# the input for this function is a string and the output is a list of unique words in that text excludig the stop-words (very common words) 

def tokenize(text):
    
    # Converting all the words to lower case
    words = map(lambda word: word.lower(), word_tokenize(text))
    
    # Removing stop words (very common words such as the, I, you, that,etc)
    cachedStopWords = stopwords.words("english")
    words = [word for word in words
                  if word not in cachedStopWords]
    
    # Stemming the words
    tokens =(list(map(lambda token: PorterStemmer().stem(token), words)))
    
    # Removing non-letters 
    p = re.compile('[a-zA-Z]+')
    
    # Removing all the words whose length is less than 3
    min_length = 3
    filtered_tokens = list(filter(lambda token: p.match(token) and len(token) >= min_length, tokens))
    
    return filtered_tokens



#the inputs of this function are a DataFrame (articles_df) and a string of words (bag_of_words) and the output is the modified DataFrame to which the number of occurances of words in the input "bag_of_words" and their frequency in each article is added. (You can find more explanation in the "How to Use" file.)

def text_mine_articles(articles_df, bag_of_words):
    
    # Getting rid of the pandas' warning when the code attempts to make changes to a slice of a DataFrame not all of it
    pd.options.mode.chained_assignment = None
    
    #Tokenizing the input "bag_of_words"
    bag_of_words = tokenize(bag_of_words)
    
    # initiating a column in the input "articles_df" with a value of zero
    articles_df['num_occurr'] = 0
    
    # 'Set' to get rid of the duplicates in the input "bag_of_words" (and also working with sets is more efficient)
    set_of_words = set(bag_of_words)
    
    print('Processing the text of the articles...')
    
    
#loop over all the articles to calculate the number of occurances of the words in "bag_of_words" in each article and also calculating the term frequency of thw words by dividing the number of occurances by the total number of words in each article.
    for j in range(0, len(articles_df)):
        
        # Convert article body into a list of tokens
        tokenized_article = tokenize(articles_df['article_text'][j])
        
        
        # Counting the occurrences of the words in "bag_of_words" in each article 
        occurrences = [word for word in tokenized_article
                        if word in set_of_words]
        amount = len(occurrences)
        
        # putting the resut in the DataFrame
        articles_df['num_occurr'][j] = amount
        
    # Computing the frequency of occurrence of the words in "bag_of_words" in each article
    articles_df['freq_occurr'] = articles_df['num_occurr']/articles_df['word_count']   
   
    print('Done!')
 
    return articles_df


#the input for this function is the DataFrame from previous function and the period of analysis. The output is a DataFrame containing the total number of articles in each month in the defined period, the frequency of the given word in each month, and the normalized frequency of the word in each month. 

def get_monthly_results(articles_df, start_year, end_year):
    
    print('Arranging the monthly results into a new dataframe...')
    
    # Number of months in range of analysis
    range = 12*(end_year - start_year + 1)
    
    # Initializing some columns for displaying the results
    
    columns = ['month_freq_occurr', 'num_articles','norm_freq_occurr']
    results_df = pd.DataFrame(columns = columns)
    results_df['date'] = pd.date_range(str(start_year) + '-01', periods = range, freq = 'M')
    
    # Set date as index to move data from articles_df to results_df conviniently
    results_df = results_df.set_index(['date'])
    results_df = results_df.fillna(0.0) # with 0s rather than NaNs
    
    # Storing the results on a monthly basis
    i = 0
    #loop over all the articles
    while i < len(articles_df):
        
        # Cut day from date column
        date = articles_df['date'][i][0:7] 
        
        # Group numbers monthly by summing daily results
        results_df.loc[date]['num_articles'] += 1
        results_df.loc[date]['month_freq_occurr'] += articles_df['freq_occurr'][i]
        
        i += 1
    
    # Normalizing the monthly frequency of occurrences by dividing the frequency of the given word in a month by the total number of artices in that month
    results_df['norm_freq_occurr'] = results_df['month_freq_occurr']/results_df['num_articles']
    
    # Detele the NaN cells if any
    results_df['norm_freq_occurr'][results_df['norm_freq_occurr'].isnull()] = 0.0
    
    print('Done!')
    
    return results_df




def visualize_results(results_df, start_year, end_year, dictionary):
    
    # Print the plot on Jupyter Notebook
    get_ipython().magic(u'matplotlib inline')
    
    # Defining plot properties
    plot_title = 'Results for Dictionary: "' + dictionary + '"' + ' from Jan ' + str(start_year) + ' to Dec ' + str(end_year)
    results_df.norm_freq_occurr.plot(legend = True, label = 'Monthly Normalized Frequency of Occurrence', 
                                        figsize = (15, 5), title = plot_title)
    results_df.num_articles.plot(secondary_y = True, style = 'g',
                                    label = 'Articles Published', legend = True)

    
    
def Bag_of_Words(articles_df):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.feature_extraction.text import CountVectorizer

    # all the very common words such as "I, you, have, has, been, be,is",etc.
    cachedStopWords = stopwords.words("english")

    #applying the function TfidfVectorizer on the text of all the articles and creating Tf-IDF matrix (excluding all the stop-words and non-letter words)
    vectorizer = TfidfVectorizer(stop_words=cachedStopWords ,token_pattern='[a-zA-Z]+')
    X = vectorizer.fit_transform(articles_df['article_text'])
  

    Y = pd.DataFrame(X.toarray())
    Y.columns= vectorizer.get_feature_names()
    

    #finding top 7 words in each article whose TF-IDF feature is more than the other words in the vocabulary
    arank = Y.apply(np.argsort, axis=1)
    ranked_cols = Y.columns.to_series()[arank.values[:,::-1][:,:7]]
    new_frame = pd.DataFrame(ranked_cols, index=Y.index)
    
    #applying the function CountVectorizer on the text of all the articles and creating Word-Count matrix (excluding all the stop-words and non-letter words)
    U=CountVectorizer(stop_words=cachedStopWords ,token_pattern='[a-zA-Z]+')
    P=U.fit_transform(articles_df['article_text'])

    #saving tfidf matrix into a data frame
    tfidf=X.toarray(order=None, out=None)
    tfidf_df=pd.DataFrame(tfidf)

    
    #saving Word-Count matrix into a data frame
    count=P.toarray(order=None, out=None)
    count_df=pd.DataFrame(count)

    Vocabulary=vectorizer.get_feature_names()
    

    #print("Unique Words (Vocabulary): ...")
    #print(vectorizer.get_feature_names())
    #print( 'TF-IDF Matrix: ...')
    #print(tfidf_df)
    #print("The Shape of the Vectorizer: ...")
    #print(X.shape)
    #print("TFIDF Table: ...")
    #print(tfidf_df.shape)
    #print("Word-Count Table: ...")
    #print(count_df.shape)
    

    return(new_frame, tfidf_df, count_df, Vocabulary)
