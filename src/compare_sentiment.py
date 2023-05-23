import datetime
import pandas as pd
import pprint
import time
import shutil
import sys, os
import re
import tweet_query
from bson.json_util import dumps
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import logging
import argparse
import json
import gzip
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, DateTime, asc
from sqlalchemy.orm import Session
from multiprocessing import Pool
from dateutil import tz
from pathlib import Path
from tweet_query.ambient_rd_figures import ambient_rank_divergence, run_rd_figure
from tweet_query.tweet_db_query import parse_labeled_df_to_counters
from tweet_query.counters import Counters, combine
from tweet_query.sentiment import load_happs_scores, counter_to_dict, filter_by_scores, get_weighted_score_np
from tweet_query.sentiment_plot import general_sentiment_shift
from utils import valid_date, time_tag
import subprocess

fname_dict = {False: "",
              True: "_test"}

def date_range(dates):
    delta = dates[-1] - dates[0]
    return delta

def parse_args(args):
    parser = argparse.ArgumentParser(
        description="A command line interface for creating interactive ambient tweet embedding plots",
    )
    parser.add_argument(
        '-a', '--anchor',
        type=str,
        help='Anchor string from initial query'
    )
    parser.add_argument(
        '-s', '--start_date',
        type=valid_date,
        default='2016-01-01'
    )
    parser.add_argument(
        '-e', '--end_date',
        type=valid_date,
        default='2022-08-15'
    )
    parser.add_argument(
        '-d', '--data',
        type=str,
        help='data with pth for sentiment plots'
    )
    parser.add_argument(
        '-l', '--label',
        type=str,
        help='True label string from classifier'
    )
    parser.add_argument(
        '--agg_level',
        type=str,
        help='string aggregation level',
        default='W',
    )
    parser.add_argument(
        '-t', '--test',
        action='store_true',
        help='Just make plot with first 10000 tweets'
    )
    
    return parser.parse_args(args)


def get_fname(args):
    fname = str(Path(args.data).parent) + '/' + str(Path(args.data).stem)
    return fname


def load_data(fname, args):
    if args.test:
        df = pd.read_csv(fname, sep='\t', lineterminator='\n', nrows=1000)
    else:
        df = pd.read_csv(fname, sep='\t', lineterminator='\n')
    return df

def load_counters(fname, args, ngrams='1grams'):

    dates = pd.date_range(*extract_date_from_fname(fname))
    counters_R = Counters(dates, args.anchor)
    counters_NR = Counters(dates, args.anchor)
    counters_all = Counters(dates, args.anchor)

    counters_R.fname = fname + f"_{str(args.agg_level)}_relevant_{ngrams}.json"
    counters_NR.fname = fname + f"_{str(args.agg_level)}_not_relevant_{ngrams}.json"
    counters_all.fname = fname + f"_{str(args.agg_level)}_all_{ngrams}.json"

    counters_R.load_json()
    counters_NR.load_json()
    counters_all.load_json()

    # removing counters to plotting range
    dates = pd.date_range(args.start_date, args.end_date, freq=args.agg_level)
    counters_R.dates = dates
    counters_R.counters = counters_R.counters[:len(dates)]
    counters_NR.dates = dates
    counters_NR.counters = counters_NR.counters[:len(dates)]
    counters_all.dates = dates
    counters_all.counters = counters_all.counters[:len(dates)]
    print(len(counters_all.counters))
    print(len(counters_all.dates))

    return counters_R, counters_NR, counters_all


def extract_date_from_fname(fname):
    """ Grab first two dates from filename"""
    pattern = '\d{4}[-]\d{2}[-]\d{2}'
    dates = re.findall(pattern, fname)
    print(dates)
    return datetime.datetime.strptime(dates[0], "%Y-%m-%d"), datetime.datetime.strptime(dates[1], "%Y-%m-%d")


def create_counters(df, args, fname, agg_level='W', scheme=1):
    
    dates = pd.date_range(*extract_date_from_fname(fname))
    counters_R = Counters(dates, args.anchor)
    counters_NR = Counters(dates, args.anchor)
    
    print(args.label)
    print(df.head())
    counters_R.counters, counters_NR.counters = parse_labeled_df_to_counters(df, dates, args.label, scheme=scheme)

    if agg_level != 'D':
        counters_R.aggregate(agg_level)
        counters_NR.aggregate(agg_level)

    counters_all = combine([counters_R, counters_NR], omit_anchor=False)
    counters_all.anchor = args.anchor

    counters_R.fname = fname + f"_{agg_level}_relevant_{scheme}grams.json"
    counters_NR.fname = fname + f"_{agg_level}_not_relevant_{scheme}grams.json"
    counters_all.fname = fname + f"_{agg_level}_all_{scheme}grams.json"

    counters_R.save(json=True, localdb=False)
    counters_NR.save(json=True, localdb=False)
    counters_all.save(json=True, localdb=False)

    return counters_R, counters_NR, counters_all


def plot_sentiment_timeseries(counters, agg_level, args, fig=None):
    counters_R, counters_NR, counters_all = counters

    if fig is None:
        fig = plt.subplots(nrows=3, ncols=1, sharex=True,
                       figsize=(6.75,6), dpi=500,
                       gridspec_kw={'height_ratios':[1,3,2]},
                       )
    counters_R.plot_sentiment_timeseries(unsafe=True, fig=fig, lang='en',
            count_type='count_no_rt', color='#1f77b4', label="- R")
    axsub,ax,ax2 = counters_NR.plot_sentiment_timeseries(unsafe=True, fig=fig, lang='en', 
            count_type='count_no_rt', hedonometer=False, label=' - NR')
    axsub,ax,ax2 = counters_all.plot_sentiment_timeseries(unsafe=True, fig=fig, lang='en', 
            count_type='count_no_rt', hedonometer=False, label=' - R + NR', color='#9b5eab')
    
    if agg_level[-1] == 'D':
        locator = mdates.DayLocator(interval=7)
        formatter = mdates.DateFormatter('%b %d ')
    elif agg_level[-1] == 'W':
        if len(counters_R.counters) < 25:
            locator = mdates.MonthLocator([i + 1 for i in range(12)])
        else:
            locator = mdates.MonthLocator([i * 4 + 1 for i in range(3)])

        formatter = mdates.DateFormatter('%b %Y')
    elif agg_level[-1] == 'M':
        formatter = mdates.DateFormatter('%b %Y')

        if date_range(counters_R.dates) < datetime.timedelta(weeks=52):
            locator = mdates.MonthLocator([i+ 1 for i in range(12)])
        elif date_range(counters_R.dates) < datetime.timedelta(weeks=4*52):
            locator = mdates.MonthLocator([i * 6 + 1 for i in range(2)])
        else:
            locator = mdates.MonthLocator([i * 6 + 1 for i in range(1)])
            formatter = mdates.DateFormatter('%Y')
    
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    for _ax in [ax, axsub, ax2]:
        _ax.grid(axis='x', which='major', color='#666666', linestyle='-', alpha=0.3)
    ax.set_ylabel("Ambient\n Sentiment", fontsize=14)
    ax2.set_ylabel("Ambient\n Sentiment \n $\sigma$", fontsize=14)
    ax.tick_params(axis='x', labelsize=12)
    plt.setp(ax.get_xticklabels(), ha='right')
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    ax.legend(fancybox=False, ncol=2)
    axsub.set_title(args.anchor, fontsize=20)
    ax.set_axisbelow(True)

    ax.set_ylim(5, 7.2)
    print(f"Min happs: {counters_R.happs_values.min()}")
    print(f"Max happs: {counters_R.happs_values.max()}")

    if counters_R.happs_values.min() < 5 or counters_R.happs_values.max() > 6.25:
        ax.set_ylim(counters_R.happs_values.min()-0.5,7.2)
    plt.tight_layout()
    figname = f"../figures/{args.anchor}_sentiment_{args.start_date.strftime('%Y-%m-%d')}_{args.end_date.strftime('%Y-%m-%d')}{fname_dict[args.test]}"
    plt.savefig(figname+".pdf")
    plt.savefig(figname+".png")
    plt.savefig(figname+".tif")



def plot_sentiment_shift(counters, args, ax=None, count_type='count_no_rt', reference_value=None, stop_lens=[(4,6)]):
    """ Plot sentiment shifts for the whole time period """

    counters_R, counters_NR, counters_all = counters
    type2score = load_happs_scores(lang='english_v2')

    type2freq_1 = counter_to_dict(counters_NR.collapse(), count_type)
    type2freq_2 = counter_to_dict(counters_R.collapse(), count_type)

    type2freq_1, type2score_new1, stop_words = filter_by_scores(type2freq_1, type2score, stop_lens)
    type2freq_2, type2score_new2, stop_words = filter_by_scores(type2freq_2, type2score, stop_lens)

    if reference_value is None:
        reference_value = get_weighted_score_np(type2freq_1, type2score)[0]

    titles = [ f"Anchor: {counters_NR.anchor} - NR", f"Anchor: {counters_R.anchor} - R",]
    shift = general_sentiment_shift(type2freq_1, type2freq_2, titles=titles, type2score_1=type2score,
                                    reference_value=reference_value, stop_lens=stop_lens, top_n=20, ax=ax)
    figname = f"../figures/{args.anchor}_sentiment_shift_{args.start_date.strftime('%Y-%m-%d')}_{args.end_date.strftime('%Y-%m-%d')}{fname_dict[args.test]}"
    plt.savefig(figname+".pdf")
    plt.savefig(figname+".png")
    plt.savefig(figname+".tif")


def plot_sentiment_shift_relevant(counters, args, ax=None, count_type='count_no_rt', reference_value=None, stop_lens=[(4,6)]):
    """ Plot sentiment shifts for the whole time period """

    counters_1, counters_2 = counters
    type2score = load_happs_scores(lang='english_v2')

    type2freq_1 = counter_to_dict(counters_1.collapse(), count_type)
    type2freq_2 = counter_to_dict(counters_2.collapse(), count_type)

    type2freq_1, type2score_new1, stop_words = filter_by_scores(type2freq_1, type2score, stop_lens)
    type2freq_2, type2score_new2, stop_words = filter_by_scores(type2freq_2, type2score, stop_lens)

    if reference_value is None:
        reference_value = get_weighted_score_np(type2freq_1, type2score)[0]

    titles = [ f"Anchor: {counters_1.anchor} - R", f"Anchor: {counters_2.anchor} - R",]
    shift = general_sentiment_shift(type2freq_1, type2freq_2, titles=titles, type2score_1=type2score,
                                    reference_value=reference_value, stop_lens=stop_lens, top_n=20, ax=ax)
    if ax is not None:
        figname = f"../figures/{counters_1.anchor}_{counters_2.anchor}_sentiment_shift_{args.start_date.strftime('%Y-%m-%d')}_{args.end_date.strftime('%Y-%m-%d')}{fname_dict[args.test]}"
        plt.savefig(figname+".pdf")
        plt.savefig(figname+".png")
        plt.savefig(figname + ".tif")


def plot_allotax_shift(counters,
                       args,
                       count_type='count',
                       matlab='/Users/michael/MATLAB_R2022b.app/bin/matlab -nosplash -nodesktop',
                       ):

    counters_R, counters_NR, counters_all = counters
    ambient_rank_divergence(counters_NR.collapse(),
                            counters_R.collapse(),
                            f"{counters_NR.anchor}-Relevant",
                            f"{counters_R.anchor}-NotRelevant",
                            (args.start_date, args.end_date),
                            (args.start_date, args.end_date),
                            count_type=count_type,
                            matlab=matlab,)

    # copy to figures folder
    fig_fname = f"figallotaxonometer9000/figallotaxonometer9000-2022-08-31-2022-08-31-rank-div-alpha-third-{args.anchor}-Relevant_{args.anchor}-NotRelevant_noname.pdf"
    print(fig_fname)
    # convert pdf to png
    #subprocess.call()

def standard_year_compare(counters,
                          args,
                          year='2019',
                          ngrams='1grams',
                          lang='en',
                          subsample=True,
                          count_type='count',
                          scheme=1,
                          matlab='/Users/michael/MATLAB_R2022b.app/bin/matlab -nosplash -nodesktop',
                          ):
    subsample_dict = {
        True: "_subsample",
        False: "",
    }
    year_dict = {
        '2019': f'{ngrams}_2019-01-01_2019-12-31_{lang}{subsample_dict[subsample]}.tsv',
        '2020': f'{ngrams}_2020-01-01_2021-01-01_{lang}{subsample_dict[subsample]}.tsv',
        '2021': f'{ngrams}_2021-01-01_2021-12-31_freq_<Day>_{lang}{subsample_dict[subsample]}.tsv',
        '2022': f'{ngrams}_2022-01-01_2022-05-09_freq_<Day>_{lang}{subsample_dict[subsample]}.tsv'
    }
    date_dict = {
        '2019': (datetime.datetime(2019, 1, 1), datetime.datetime(2020, 1, 1)),
        '2020': (datetime.datetime(2020, 1, 1), datetime.datetime(2021, 1, 1)),
        '2021': (datetime.datetime(2021, 1, 1), datetime.datetime(2021, 12, 31)),
        '2022': (datetime.datetime(2022, 1, 1), datetime.datetime(2022, 5, 9)),
    }
    # initialize
    start_date = date_dict[year][0]
    end_date = date_dict[year][1]
    dates = pd.date_range(start_date, end_date, freq='D')

    # load reference dataset
    pth = '~/tweet_query/data/reference/'
    reference_df = pd.read_csv(pth + year_dict[year],
                               sep='\t',
                               index_col='ngram')


    # split Relevant counters into date range?
    counters_R, counters_NR, counters_all = counters




    # run matlab plotting script

    word_1 = f"All-Twitter-{year}"
    word_2 = f"{counters_NR.anchor}-Relevant"
    dates1 = (date_dict[year][0], date_dict[year][1])
    dates2 = (args.start_date, args.end_date)
    data_pth = '../data/'
    dist_dict1 = reference_df[[count_type, 'rank', 'freq']]
    fname1 = f'test1_{dates1[0].strftime("%Y-%m-%d")}_{dates1[1].strftime("%Y-%m-%d")}.tsv'
    dist_dict1.to_csv(data_pth + fname1, sep="\t")

    dist_dict2 = tweet_query.tweet_db_query.ambient_ngrams_dataframe_from_counter(counters_R.collapse(),
                                                                                  word_2,
                                                                                  scheme=scheme,
                                                                                  count_type=count_type)
    fname2 = f'test2_{dates2[0].strftime("%Y-%m-%d")}_{dates2[1].strftime("%Y-%m-%d")}.tsv'
    dist_dict2.to_csv(data_pth + fname2, sep="\t")
    return run_rd_figure(word_1, word_2, dates1, dates2, fname1, fname2, matlab, data_pth,tags=f"_{ngrams}")



def plot_sentiment_final(counters, agg_level, args):

    pass


def main(args=None):
    
    if args is None:
        args = sys.argv[1:]
        args = parse_args(args)

    testing = True

    fname = get_fname(args)
    print(fname)

    # load counters if already saved, else parse for the first time.
    if os.path.isfile(fname + f"_{args.agg_level}_relevant_1grams.json.gz"):
        print("Loading counters...")
        counters = load_counters(fname, args)
    else:
        print("Parsing counters...")
        df = load_data(args.data, args)
        counters = create_counters(df, args, fname)


    if testing:  # plot individual panels
        plot_sentiment_timeseries(counters, args.agg_level, args)
        plot_sentiment_shift(counters, args)
        plot_allotax_shift(counters, args)
        standard_year_compare(counters, args)
        if os.path.isfile(fname + f"_{args.agg_level}_relevant_2grams.json.gz"):
            print("Loading counters...")
            counters = load_counters(fname, args)
        else:
            print("Parsing counters...")
            df = load_data(args.data, args)
            counters = create_counters(df, args, fname)
        df = load_data(args.data, args)
        counters = create_counters(df, args, fname, scheme=2)
        standard_year_compare(counters, args, scheme=2, ngrams='2grams')


    else:  # plot combined panels
        plot_sentiment_final(counters, args.agg_level, args)


if __name__ == "__main__":
    main()
