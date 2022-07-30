import pandas as pd
import numpy as np
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim

'''
    This python file contains functions that will be used in the project
'''


def discussion_text_clean(df):
    '''
        Clean the discussion text
    '''

    re = r'(&nbsp)|(<[^>]*>)|(\\n)'
    df['cleaned_discussion_post'] = df['discussion_post_content'].str.replace(re, '', regex=True)
    return df


def utc_to_pts(df):
    '''
        Transfer the UTC timestamps to Pacific timestamps
    '''
    
    df['discussion_topic_posted_at'] = pd.to_datetime(df['discussion_topic_posted_at'], utc=True)
    df['discussion_post_created_at'] = pd.to_datetime(df['discussion_post_created_at'], utc=True)
    
    df['discussion_topic_posted_at'] = df['discussion_topic_posted_at'].dt.tz_convert('US/Pacific')
    df['discussion_post_created_at'] = df['discussion_post_created_at'].dt.tz_convert('US/Pacific')

    return df



#def create_term_col(df):
#    '''
#        1. Remove records in summer session
#        2. Create a column with the term the discussion posts belong to
#    '''
#
#    col = 'discussion_post_created_at'
#
#    # Remove the records in summer session
#    df = df[(df[col] < '06-21-2021') | (df[col] > '09-03-2021')]
#
#    # Transfer the UTC timestamps to Pacific timestamps
#    df = utc_to_pts(df)
#
#    # Time for terms from Fall 2020 to Spring 2022
#    conditions = [((df[col] >= '09-28-2020') & (df[col] <= '12-18-2020')),
#                  ((df[col] >= '01-04-2021') & (df[col] <= '03-19-2021')),
#                  ((df[col] >= '03-24-2021') & (df[col] <= '06-11-2021')),
#                  ((df[col] >= '06-21-2021') & (df[col] <= '09-03-2021')),
#                  ((df[col] >= '09-20-2021') & (df[col] <= '12-10-2021')),
#                  ((df[col] >= '01-03-2022') & (df[col] <= '03-18-2022')),
#                  ((df[col] >= '03-23-2022') & (df[col] <= '06-10-2022'))]
#
#    terms = ['Fall 2020', 'Winter 2021', 'Spring 2021', 'Summer 2021','Fall 2021', 'Winter 2022', 'Spring 2022']
#
#    df['term'] = np.select(conditions, terms)
#    return df
    

def discussion_topic_post_time_gap(df):
    '''
        Calculate the time gap (by day) between the created time for the dicussion topics and that for the post of discussion
    '''
    
    df['discussion_topic_post_time_gap'] = (df['discussion_post_created_at'] - df['discussion_topic_posted_at']) / np.timedelta64(1, 'D')
    
    return df


def standardized_score(df, cols):
    '''
        Standardize the features
    '''
    
    # Ranks of variables within the same discussion topics
    df[cols] = df.groupby(by=['term', 'canvas_course_id', 'discussion_topic_id'])[cols].rank()
    
    # Final feature scores
    cnt = df.groupby(by=['term', 'canvas_course_id', 'discussion_topic_id'])['discussion_post_id'].count().reset_index(name='post_cnt') # counts of numbers of discussion posts in each topics
    
    df = df.merge(cnt, how='left', on=['term', 'canvas_course_id', 'discussion_topic_id'])
    
    df[cols] = df[cols].div(df['post_cnt'], axis=0)
    
    df = df.drop('post_cnt', axis=1)
    
    return df
    


def discussion_post_count(df):
    '''
        Calculate the the number of dicussion posts for each student
    '''

    df = df.groupby(by=['term', 'mellon_id', 'canvas_course_id'])['discussion_post_id'].count().reset_index()
    df = df.groupby(by=['term'])['discussion_post_id'].mean().reset_index(name = 'discussion_post_cnt_avg')
    return df


def calculate_features_avg(df, cols):
    '''
        Calculate all features average including features from direct calculations and NLP models
    '''

    rename_dic = {}

    for i in range(len(cols)):
        rename_dic[cols[i]] = cols[i] + '_avg'

    df = df.groupby(by=['term', 'mellon_id', 'canvas_course_id'])[cols].mean().reset_index()
    df = df.groupby(by='term')[cols].mean().reset_index().rename(columns = rename_dic, errors = 'raise')
    return df


def discussion_reply_rate(df):
    '''
        Calculate the reply rate for each student by terms
    '''

    df_reply = df.groupby(by=['parent_discussion_post_id'])['discussion_post_id'].count().reset_index(name='reply_rate')
    df = df.merge(df_reply, left_on='discussion_post_id', right_on='parent_discussion_post_id', how='inner')
    return df


def lda_model(df, num_topics, passes):
    '''
        Generate the LDA models given the dataframe with cleaned discussion post
    '''
    
    tokenizer = RegexpTokenizer(r"[\w']+")

    # create English stop words list
    en_stop = get_stop_words('en')

    # Create p_stemmer of class PorterStemmer
    p_stemmer = PorterStemmer()
    
    # compile the cleaned_discussion_post column into a list
    doc_set = [str(x) for x in list(df['cleaned_discussion_post'].values)]
    #print(len(doc_set))
    
    # list for tokenized documents in loop
    texts = []
    
    # loop through document list
    for content in doc_set:
    
        # clean and tokenize document string
        raw = content.lower()
        tokens = tokens = tokenizer.tokenize(raw)

        # remove stop words from tokens
        stopped_tokens = [w for w in tokens if w not in en_stop]

        # stem tokens
        stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]

        # add tokens to list
        texts.append(stemmed_tokens)
    
    # turn our tokenized documents into a id <-> term dictionary
    dictionary = corpora.Dictionary(texts)

    # convert tokenized documents into a document-term matrix
    corpus = [dictionary.doc2bow(text) for text in texts]
    
    # generate the LDA model
    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=num_topics, id2word = dictionary, passes=passes)
    
    return ldamodel