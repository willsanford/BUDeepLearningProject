import os
import time
import pandas as pd
from tqdm import tqdm 
import pickle
import openai
'''
company_tweet = pd.read_csv('../data/dataset1/Company_Tweet.csv')
company = pd.read_csv('../data/dataset1/Company.csv')
tweets1 = pd.read_csv('../data/dataset1/Tweet.csv')
tweets1['Date'] = pd.to_datetime(tweets1['post_date'] * 1e9)

start_date = pd.to_datetime('2019-01-01 00:00:00')
end_date = pd.to_datetime('2019-12-31 23:59:58')

t = tweets1.merge(company_tweet, on='tweet_id', how='inner')[['tweet_id', 'body', 'Date', 'ticker_symbol']]

t = t[(t.Date < end_date) & (t.Date > start_date)]

count_per_day = t.groupby(t['Date'].dt.date).size()

sampled_df = t.groupby(t['Date'].dt.date).apply(lambda x: x.sample(min(len(x), 100))).reset_index(drop=True)
sampled_df = sampled_df.sample(frac=1)

t.to_csv('../data/cleaned/full.csv')
sampled_df.to_csv('../data/cleaned/sampled.csv')
'''

t = pd.read_csv('../data/cleaned/full.csv', index_col=0)
sampled_df = pd.read_csv('../data/cleaned/sampled.csv', index_col=0)

sentiment = [ '1(neg)' , '2' , '3' , '4' , '5(pos)']
relation = [ 'Mostly related' , 'Somewhat related' , 'Unrelated']
advantage = [ 'Advantage' , 'Disadvantage']

def sentiment_function(Message):
    openai.api_key = "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"
    ## openai.api_key = "sk-gYvQOh4jIIb6robTxMtT3BlbkFJz2ouvCowounfAoaP8AbA"    
    response = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo",
        request_timeout=30,
        messages =[
        { 
            "role": "system", 
            "content" : """ 
                You are a financial analyst , who
                analyzes if news could have some benefits for the S&P500 index.
                """ 
        } ,
        { 
            "role" : "user",
            "content" : f"""
                Determine from { sentiment }: probabilities , from { advantage }:
                probabilities , from { relation }: probabilities .
                Format : [ Sentiment : Probabilities for each sentiment , Advantage :
                Probabilities , Relation : Probabilities ].
                Alternatively , state " NA ".
                 '''{Message}'''"""
        } ,

         ] ,
        temperature = 0
    )
    ans = response['choices'][0]['message']['content']
    return ans

i = 0
k = 0
data = []
bad_data = []
for ind, (ind_, (tweet_id , body, Date, ticker_symbol)) in tqdm(enumerate(sampled_df.iterrows()), total=sampled_df.shape[0]):
    if k > 841: # Resume
        time.sleep(1)
        try:
            d = sentiment_function(body)
            data.append((tweet_id, d))
        except Exception as e:
            print(f'Error for {tweet_id}:')
            print(e)        
            bad_data.append(tweet_id)
    i += 1

    if i % 20 == 0:
        i = 0

        if k > 841:
            with open(f"../data/processed/{k}.pkl", 'wb') as fh:
                pickle.dump(data, fh)
            data = []
            with open(f"../data/processed/{k}_b.pkl", 'wb') as fh:
                pickle.dump(bad_data, fh)
            bad_data = []
        k += 1
    
