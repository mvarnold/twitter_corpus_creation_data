import datetime
import pandas as pd
import pprint
import time
import shutil
import sys, os
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
from tweet_query.ambient_rd_figures import ambient_rank_divergence
from tweet_query.tweet_db_query import parse_labeled_df_to_counters
from tweet_query.counters import Counters, combine
from tweet_query.sentiment import load_happs_scores, counter_to_dict, filter_by_scores, get_weighted_score_np
from tweet_query.sentiment_plot import general_sentiment_shift
from utils import valid_date, time_tag
import subprocess
from itertools import combinations
from compare_sentiment import *


data_dict = {
    'Solar': "../data/Solar/Solar_2016-01-01_2022-12-31_D_False_stateSolar_2016-01-01_2022-08-30_D_False_state_top1000_labeled_all-mpnet-base-v2_labeled",
    "Wind": "../data/Wind/Wind_2016-01-01_2022-12-31_D_False_stateWind_2020-01-01_2021-01-01_D_False_state_top1000_labeled_all-mpnet-base-v2_labeled",
             "Nuclear": "../data/Nuclear/Nuclear_2016-01-01_2022-12-31_D_False_stateNuclear_2016-01-01_2022-03-15_D_False_state_top1000_labeled_all-mpnet-base-v2_labeled",
}
args = parse_args(sys.argv[1:])

f,axes = plt.subplots(1, 3, figsize=(19,12), dpi=150)

i = 0
textstr = 'ABC'
counters_dict = {}
for anchor, data_fname in data_dict.items():
    args.anchor = anchor
    counters = load_counters(data_fname, args=args)
    plot_sentiment_shift(counters, args, axes[i])
    props = dict(boxstyle='round', facecolor='w', alpha=0.5)
    # place a text box in upper left in axes coords
    axes[i].text(0.04, 0.98, textstr[i], transform=axes[i].transAxes, fontsize=24,
            verticalalignment='top', bbox=props)
    i += 1
    counters_dict[anchor] = counters[0]  # add data to dict


plt.tight_layout()
plt.savefig("../figures/combined_shifts.png")
plt.savefig("../figures/combined_shifts.pdf")
plt.close()

f,axes = plt.subplots(1, 3, figsize=(19,12), dpi=150)

i = 0
for counters_pair in combinations(counters_dict.values(),2):
    plot_sentiment_shift_relevant(counters_pair, args, axes[i]);
    i += 1

plt.tight_layout()
plt.savefig("../figures/combined_shifts_relevant.png")
plt.savefig("../figures/combined_shifts_relevant.pdf")
plt.close()
