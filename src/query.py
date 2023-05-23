from tweet_query.tweet_db_query import assemble_ambient_tweets
from tweet_query.utils import save_json, load_json, valid_date, text_reader
import os
import sys
import argparse
from bson import json_util
import gzip
import json
import time
import datetime
import pandas as pd
from pprint import pprint


def parse_args(args):
    parser = argparse.ArgumentParser(
        description="A command line interface for creating interactive ambient tweet embedding plots",
    )
    parser.add_argument(
        '-w', '--word',
        type=str,
        help='anchor to plot'
    )
    parser.add_argument(
        '-l', '--word_list',
        type=text_reader,
        default=None,
        help='textfile of words to plot'
    )
    parser.add_argument(
        '-r', '--high_res',
        help="Flag to plot high res",
        action='store_true',
    )
    parser.add_argument(
        '--location',
        type=str,
        default=None,
        help="Include a location field, either state or city_state",
    )
    parser.add_argument(
        '-b', '--begin_date',
        help="beginning of date range -- formate YYYY-MM-DD",
        type=valid_date,
        default='2020-01-01',
    )
    parser.add_argument(
        '-e', '--end_date',
        help="end of date range -- format YYYY-MM-DD",
        type=valid_date,
        default='2020-12-31'
    )
    parser.add_argument(
        '-p', '--pth',
        help="path to save embedding data",
        default="../data/"
    )
    parser.add_argument(
        '-c', '--count_type',
        type=str,
        default='pure_text',
        help="Choose between 'pure_text' for originally authored tweets or 'rt_text' for retweeted text."
    )

    return parser.parse_args(args)


def grab_ambient_tweets(anchor, args):
    """ Get tweets"""
    start_date, end_date, high_res = (args.begin_date, args.end_date, args.high_res)
    start_time = time.time()

    freq = 'D'  # daily frequency
    dates = pd.date_range(start_date, end_date, freq=freq)
    formatting = lambda x: f'_{x}' if x is not None else ''
    loc_str = f'{formatting(args.location)}'
    fname = f"{args.pth}/{anchor}/{anchor}_{start_date.strftime('%Y-%m-%d')}_{end_date.strftime('%Y-%m-%d')}_{freq}_{high_res}" \
            f"{loc_str}.json.gz"
    print(f"Data stored at: {fname}")
    data_dir = f'{args.pth}/{anchor}'
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)
        print('Created data directory.')

    # load cached data if it exists
    if os.path.isfile(fname):
        print('Loading cached data.')
        tweets = load_json(fname)
    # else query from database
    else:
        if args.word_list:
            print(args.word_list)
            print(type(args.word_list))
            query = query = {'$text': {'$search':'"' + '" "'.join(args.word_list) + '"',
                                       '$caseSensitive': False},
                             'fastText_lang': 'en'}
            print(query)
        else:
            query = None
        if args.location:
            tweets = assemble_ambient_tweets(anchor, dates, high_res=high_res, project=False, limit=0,
                                             db='tweet_segmented_location',
                                             hostname='serverus.cems.uvm.edu',
                                             port=27015,
                                             query=query,
                                             )

        else:
            tweets = assemble_ambient_tweets(anchor, dates, high_res=high_res, project=False, limit=0, query=query,)

        print(f" Num Tweets: {len(tweets)}")
        save_json(fname, tweets)

    print(f"Data acquired for '{anchor}' in {time.time() - start_time:.2f} s")

    fname = fname.split('.')[-3]
    return tweets, fname




def preprocess(tweets, count_type):
    """ Select either rt_text or pure_text containing tweets"""

    if count_type in ['pure_text', 'rt_text']:
        # filter retweets
        tweets = [tweet for tweet in tweets if count_type in tweet]
    else:
        raise ValueError("Please choose between --count_type='pure_text' and --count_type='rt_text'.")

    return tweets

def get_user_bio(tweet):
    if 'actor' in tweet:
        return tweet['actor']['summary']
    elif 'user' in tweet:
        return tweet['user']['description']


def get_tweet_link(tweet):
    if 'link' in tweet:
        return tweet['link']
    elif 'user' in tweet:
        try:
            return f"http://twitter.com/{tweet['user']['screen_name']}/statuses/{str(tweet['id'])}"
        except KeyError as e:
            print(e)
            print("KeyError")
            pprint(tweet)

    else:
        pprint(tweet)
        return None

def save_tsv(tweets, fname, args):
    """ Saves relevant tweet data in TSV format"""
    _id = [tweet['id'] for tweet in tweets]
    df = pd.DataFrame(_id, columns=['id'])
    df['lang'] = [tweet['fastText_lang'] for tweet in tweets]
    df['link'] = [get_tweet_link(tweet) for tweet in tweets]
    df['tweet_created_at'] = [tweet['tweet_created_at'] for tweet in tweets]
    if args.location:
        df[args.location] = [tweet[args.location] if args.location in tweet else None for tweet in tweets]
    df['bio'] = [get_user_bio(tweet) for tweet in tweets]
    df['Text'] = [tweet[args.count_type] for tweet in tweets]
    df['Label'] = None

    # save all tweets
    if args.word is not None:
        label = args.word
    else:
        label = "_".join(args.word_list)

    # save all data
    df.to_csv(args.pth + f"{label}/" + os.path.basename(fname) + '.tsv', sep='\t')

    # shuffle and retain only top 40000 for interactive plots
    max_tweets = 40000
    frac = min(max_tweets / df.shape[0], 1)
    df = df.sample(frac=frac).reset_index(drop=True)

    df.to_csv(args.pth + f"{label}/" + os.path.basename(fname) + '_top40000.tsv', sep='\t')
    df.loc[:1000, :].to_csv(args.pth + f"{label}/" + os.path.basename(fname) + '_top1000.tsv', sep='\t')

    print(df.head())
    print(f"Saved tsv files in {args.pth + label}/")

    return


def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    # load data from tweetDB or local file
    tweets, fname = grab_ambient_tweets(args.word, args)
    print(fname)
    # choose between analyzing original tweets or retweeted text
    tweets = preprocess(tweets, args.count_type)

    save_tsv(tweets, fname, args)


    return tweets



if __name__ == '__main__':
    tweets_ = main()
