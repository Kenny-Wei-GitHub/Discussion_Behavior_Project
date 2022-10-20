import pandas as pd
import numpy as np
import re, nltk, gensim
import spacy
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from pprint import pprint
from scipy.stats import ttest_ind
from IPython.display import Audio, display


'''
    This python file contains functions that will be used in the project
'''


def discussion_text_clean(df):
    '''
        Clean the discussion text
    '''

    re = r'(&nbsp)|(<[^>]*>)|(\\n)'
    df['cleaned_discussion_post'] = df.loc[:, 'discussion_post_content'].str.replace(re, '', regex=True)
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
    df[cols] = df.groupby(by=['term', 'canvas_course_id', 'discussion_topic_id'])[cols].rank(method='min')
    
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

    df = df.groupby(by=['term', 'mellon_id', 'canvas_course_id'])['discussion_post_id'].count().reset_index(name = 'discussion_post_cnt_avg')
    return df


def calculate_features_avg(df, cols):
    '''
        Calculate all features average including features from direct calculations and NLP models
    '''

    rename_dic = {}

    for i in range(len(cols)):
        rename_dic[cols[i]] = cols[i] + '_avg'

    df = df.groupby(by=['term', 'mellon_id', 'canvas_course_id'])[cols].mean().reset_index().rename(columns = rename_dic, errors = 'raise')
    return df


def term_code(row):
    if row['term'] == 'Fall 2020':
        return 1
    elif row['term'] == 'Winter 2021':
        return 2
    elif row['term'] == 'Spring 2021':
        return 3
    elif row['term'] == 'Fall 2021':
        return 4
    elif row['term'] == 'Winter 2022':
        return 5
    elif row['term'] == 'Spring 2022':
        return 6


def t_test_group(term, df1, df2, col, alter='two-sided'):
    '''
        Conduct t-test on two student groups in one specific term/quarter for some variables
    '''
    
    temp1 = df1.loc[df1['term'] == term]
    temp2 = df2.loc[df2['term'] == term]
    return ttest_ind(temp1[col], temp2[col], alternative=alter)


def t_test_term(df1, df2, col, alter='two-sided'):
    '''
        Conduct t-test on two terms/quarters for some variables
    '''
    
    return ttest_ind(df1[col], df2[col], alternative=alter)


def discussion_reply_rate(df):
    '''
        Calculate the reply rate for each student by terms
    '''

    df_reply = df.groupby(by=['parent_discussion_post_id'])['discussion_post_id'].count().reset_index(name='reply_rate')
    df = df.merge(df_reply, left_on='discussion_post_id', right_on='parent_discussion_post_id', how='inner')
    return df


def sent_to_words(sentences):
    '''
        Tokenization
    '''
    for s in sentences:
        yield gensim.utils.simple_preprocess(s, deacc=True)

        
def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    '''
        Lemmatization
    '''
    
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    texts_out = []
    for s in texts:
        doc = nlp(" ".join(s))
        texts_out.append(" ".join([token.lemma_ if token.lemma_ not in ['-PRON-'] else '' for token in doc if token.pos_ in allowed_postags]))
        
    return texts_out
    

def print_topic(model, vectorizer):
    terms = vectorizer.get_feature_names()
    lst = []
    
    for i, c in enumerate(model.components_):
        zipped = zip(terms, c)
        top_term_key = sorted(zipped, key=lambda x: x[1], reverse=True)[:6]
        top_term_lst = list(dict(top_term_key).keys())
        print('Topic '+str(i)+':', top_term_lst)
        lst.append(top_term_lst)
    
    return lst


def lda_preprocess(df):
    '''
        Preprocess the discussion posts
    '''

    # compile the cleaned_discussion_post column into a list
    doc_set = [str(x) for x in list(df['cleaned_discussion_post'].values)]
    #print(len(doc_set))
    
    # list for tokenized documents
    data_words = list(sent_to_words(doc_set))
    
    # list for lemmatized documents
    data_lemmatized = lemmatization(data_words, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
    
    # Create the Document-Word matrix
    vectorizer = CountVectorizer(analyzer='word', min_df=10, stop_words='english', lowercase=True, token_pattern='[a-zA-Z0-9]{3,}', max_features=10000)
    data_vectorized = vectorizer.fit_transform(data_lemmatized)
    
    return vectorizer, data_vectorized
    
    
def lda_model(vectorizer, data_vectorized):
    '''
        Generate the LDA models given preprocessd discussion posts
    '''
    
    # generate the LDA model
    #if ldamodel is None:
    #    ldamodel = LatentDirichletAllocation(n_components=15, learning_decay=0.9, max_iter=10, learning_method='online', random_state=100, batch_size=128, evaluate_every = -1, n_jobs = -1, total_samples=1168344).partial_fit(data_vectorized)
    #else:
    #    ldamodel = ldamodel.partial_fit(data_vectorized)
    
    ldamodel = LatentDirichletAllocation(n_components=14, learning_decay=0.9, learning_method='online', random_state=100, batch_size=128, evaluate_every = -1).fit(data_vectorized)
    
    topic_lst = print_topic(ldamodel, vectorizer)
    
    return ldamodel, topic_lst
    

def allDone():
  display(Audio(url='https://sound.peal.io/ps/audios/000/000/537/original/woo_vu_luvub_dub_dub.wav', autoplay=True))