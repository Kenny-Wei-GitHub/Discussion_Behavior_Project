import pandas as pd
import numpy as np

'''
    This python file contains functions that will be used in the project
'''


def discussion_text_clean(df):
    '''
        Clean the discussion text
    '''

    regex = r'(&nbsp)|(<[^>]*>)|(\\n)'
    df['cleaned_discussion_post'] = df['discussion_post_content'].str.replace(regex, '')
    return df


def utc_to_pts(df):
    '''
        Transfer the UTC timestamps to Pacific timestamps
    '''
    
    df['discussion_topic_posted_at'] = pd.to_datetime(df['discussion_topic_posted_at'], utc=True)
    df['discussion_post_created_at'] = pd.to_datetime(df['discussion_post_created_at'], utc=True)
    
    df['discussion_topic_posted_at'].dt.tz_convert('US/Pacific')
    df['discussion_post_created_at'].dt.tz_convert('US/Pacific')

    return df


def create_term_col(df):
    '''
        1. Remove records in summer session
        2. Create a column with the term the discussion posts belong to
    '''

    col = 'discussion_post_created_at'

    # Remove the records in summer session
    df = df[(df[col] < '06-21-2021') | (df[col] > '09-03-2021')]

    # Transfer the UTC timestamps to Pacific timestamps
    df = utc_to_pts(df)

    # Time for terms from Fall 2020 to Spring 2022
    conditions = [((df[col] >= '09-28-2020') & (df[col] <= '12-18-2020')),
                  ((df[col] >= '01-04-2021') & (df[col] <= '03-19-2021')),
                  ((df[col] >= '03-24-2021') & (df[col] <= '06-11-2021')),
                  ((df[col] >= '06-21-2021') & (df[col] <= '09-03-2021')),
                  ((df[col] >= '09-20-2021') & (df[col] <= '12-10-2021')),
                  ((df[col] >= '01-03-2022') & (df[col] <= '03-18-2022')),
                  ((df[col] >= '03-23-2022') & (df[col] <= '06-10-2022'))]

    terms = ['Fall 2020', 'Winter 2021', 'Spring 2021', 'Summer 2021','Fall 2021', 'Winter 2022', 'Spring 2022']

    df['term'] = np.select(conditions, terms)
    return df


def discussion_topic_post_time_gap(df):
    '''
        Calculate the time gap (by day) between the create time for the dicussion topics and the post of discussion
    '''

    df['discussion_topic_post_time_gap'] = (df['discussion_post_created_at'] - df['discussion_topic_posted_at']) / np.timedelta64(1, 'D')
    return df


def discussion_post_count(df):
    '''
        Calculate the the number of dicussion posts for each student
    '''

    df = df.groupby(by=['term', 'mellon_id'])['discussion_post_id'].count().reset_index(name = 'discussion_post_cnt')
    return df


def calculate_features_avg(df, cols):
    '''
        Calculate all features average including features from direct calculations and NLP models
    '''

    rename_dic = {}

    for i in range(len(cols)):
        rename_dic[cols[i]] = cols[i] + '_avg'

    df = df.groupby(by=['term'])[cols].mean().reset_index().rename(columns = rename_dic, errors = 'raise')
    return df


def discussion_reply_rate(df):
    '''
        Calculate the reply rate for each student by terms
    '''

    df = df.groupby(by=['parent_discussion_post_id'])['discussion_post_id'].count().reset_index(name='reply_rate')
    return df
