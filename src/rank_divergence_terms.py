import sys
import argparse
import pandas as pd
import tweet_query
import subprocess
from datetime import datetime
from utils import valid_date, time_tag
from pprint import pprint


def create_counters(df, args, ):
    dates = pd.date_range(args.start_date, args.end_date)
    counters = tweet_query.counters.Counters(dates, args.anchor)

    counters.counters = tweet_query.tweet_db_query.parse_df_to_counters(df, dates, scheme=int(args.ngrams[0]))
    agg_level = 'D'

    if agg_level != 'D':
        counters.aggregate(agg_level)

    print(f"Finished loading {args.anchor}")
    return counters


def standard_year_compare(counters,
                          year,
                          filter=None,
                          high_res=False,
                          caching=False,
                          ngrams='1grams',
                          count_type='count',
                          subsample=True,
                          pth='../data/reference/',
                          data_pth='../data/',
                          matlab='/Users/michael/MATLAB_R2022b.app/bin/matlab -nosplash -nodesktop',
                          tags='',
                          plot=True,
                          print_head=False,
                          alpha=1/4,
                          lang='en',
                          labeled=False,):
    subsample_dict = {
        True: "_subsample",
        False: "",
    }

    year_dict = {
        '2019': f'{ngrams}_2019-01-01_2020-01-01_freq_<Day>_{lang}{subsample_dict[subsample]}.tsv',
        '2020': f'{ngrams}_2020-01-01_2021-01-01_freq_<Day>_{lang}{subsample_dict[subsample]}.tsv',
        '2021': f'{ngrams}_2021-01-01_2021-12-31_freq_<Day>_{lang}{subsample_dict[subsample]}.tsv',
        '2022': f'{ngrams}_2022-01-01_2022-05-09_freq_<Day>_{lang}{subsample_dict[subsample]}.tsv'
    }
    date_dict = {
        '2019': (datetime(2019, 1, 1), datetime(2020, 1, 1)),
        '2020': (datetime(2020, 1, 1), datetime(2021, 1, 1)),
        '2021': (datetime(2021, 1, 1), datetime(2021, 12, 31)),
        '2022': (datetime(2022, 1, 1), datetime(2022, 5, 9)),
    }
    start_date = date_dict[year][0]
    end_date = date_dict[year][1]
    dates = pd.date_range(start_date, end_date, freq='D')
    anchor = counters.anchor

    ambient_counter = counters.collapse()
    ambient_df = pd.DataFrame(ambient_counter).T

    reference_df = pd.read_csv(pth + year_dict[year],
                               sep='\t',
                               index_col='ngram')
    print(f"Finished loading Reference")

    # compute ranks
    ambient_df['rank'] = ambient_df[count_type].rank(ascending=False)
    reference_df['rank'] = reference_df[count_type].rank(ascending=False)

    divergence = tweet_query.measurements.rank_divergence(reference_df, ambient_df, alpha=alpha).sort_values(
        ascending=False)
    fname = f"query_terms/{anchor}_{ngrams}_{year}_divergence_{labeled}.txt"
    divergence.to_csv(fname, sep='\t')

    if plot:
        fname1 = f'reference_{lang}_{dates[0].strftime("%Y-%m-%d")}_{dates[-1].strftime("%Y-%m-%d")}.tsv'
        reference_df.to_csv(data_pth + fname1, sep="\t")

        fname2 = f'ambient_{lang}_{" ".join(anchor.split("_"))}_{dates[0].strftime("%Y-%m-%d")}_{dates[-1].strftime("%Y-%m-%d")}.tsv'
        ambient_df.to_csv(data_pth + fname2, sep="\t")

        print('Plotting...')
        tweet_query.ambient_rd_figures.run_rd_figure('Twitter', " ".join(anchor.split("_")), date_dict[year],
                                                     dates, fname1, fname2,
                                                     matlab, data_pth,
                                                     tags=tags+f"_high_res_{high_res}_{ngrams}")




    return divergence


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
    )
    parser.add_argument(
        '-e', '--end_date',
        type=valid_date,
    )
    parser.add_argument(
        '-f', '--fname',
        type=str,
        help='data with pth'
    )
    parser.add_argument(
        '-t', '--test',
        action='store_true',
        help='Just make plot with first 10000 tweets'
    )
    parser.add_argument(
        '-n', '--ngrams',
        default='1grams',
        type=str,
        help='Compare 1grams, 2grams, or 3grams'
    )
    parser.add_argument(
            '-l','--labeled',
            action='store_true',
            )

    return parser.parse_args(args)


def load_data(args):
    """ Load tweet tsv"""
    if args.test:
        df = pd.read_csv(args.fname, sep='\t', lineterminator='\n', nrows=10000)
    else:
        df = pd.read_csv(args.fname, sep='\t', lineterminator='\n')
    return df


def main(args=None):
    if args is None:
        args = sys.argv[1:]
        args = parse_args(args)

    # load list of words
    # todo: filter search terms from list?

    df = load_data(args)
    counters = create_counters(df, args)


    divergence = standard_year_compare(counters, str(args.start_date.year),
            subsample=False,
            ngrams=args.ngrams,
            labeled=args.labeled,
            )
    pprint([i for i in divergence[:10].iteritems()])
    fname = f"query_terms/{args.anchor}_divergence.txt"



if __name__ == "__main__":
    main()
