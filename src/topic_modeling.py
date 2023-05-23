from bertopic import BERTopic
from tweet_query.tweet_db_query import get_ambient_tweets, assemble_ambient_tweets
import os
import sys
import argparse
from bson import json_util
import gzip
import json
import time
import datetime
import pandas as pd
from sentence_transformers import SentenceTransformer
from umap import UMAP

import matplotlib.pyplot as plt
from pprint import pprint

def save_json(fname, tweets):
    """ Save tweets to disk as .json.gz"""
    with gzip.open(fname,
                   'wt') as f:
        json.dump(tweets, f, default=json_util.default)


def load_json(fname):
    """ Load a gzipped json object of the tweets"""
    with gzip.open(fname,
                   'rt') as f:
        dict_list = json.load(f, object_hook=json_util.object_hook)
        return dict_list

def valid_date(s):
    try:
        return datetime.datetime.strptime(s, "%Y-%m-%d")
    except ValueError:
        msg = "Not a valid date: '{0}'.".format(s)
        raise argparse.ArgumentTypeError(msg)

def text_reader(fname):
    with open(fname, 'r') as f:
        word_list = [i.strip() for i in f.readlines()]
    return word_list

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
    print(fname)

    if os.path.isfile(fname):
        tweets = load_json(fname)
    else:
        if args.location:
            tweets = assemble_ambient_tweets(anchor, dates, high_res=high_res, project=False, limit=0,
                                             db='tweet_segmented_location',
                                             hostname='serverus.cems.uvm.edu',
                                             port=27015,
                                             )

        else:
            tweets = assemble_ambient_tweets(anchor, dates, high_res=high_res, project=False, limit=0)
        print(f" Num Tweets: {len(tweets)}")
        save_json(fname, tweets)

    print(f"Data acquired for '{anchor}' in {time.time() - start_time:.2f} s")

    fname = fname.split('.')[0]
    return tweets, fname

def preprocess_tweets(tweets, count_type='pure_text'):

    tweets = [tweet for tweet in tweets if 'verb' in tweet and tweet['verb'] == 'post']
    docs = [tweet[count_type] for tweet in tweets if count_type in tweet]

    return tweets, docs


def tweet_timestamps(tweets, count_type='pure_text'):

    return [tweet['tweet_created_at'] for tweet in tweets if count_type in tweet]


def model_topics(tweets):
    """ Create embeddings and run topic model."""
    tweets, docs = preprocess_tweets(tweets)

    # create embeddings
    sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = sentence_model.encode(docs, show_progress_bar=False)

    topic_model = BERTopic()
    topics, probs = topic_model.fit_transform(docs, embeddings=embeddings)
    return tweets, topics, topic_model, embeddings


def main(args=None):
    """Create"""
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)
    print(args)

    tweets, fname = grab_ambient_tweets(args.word, args)
    print(f" Num Tweets: {len(tweets)}")

    return model_topics(tweets)



if __name__ == "__main__":
    tweets, topics, topic_model, embeddings = main()