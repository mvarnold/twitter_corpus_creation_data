import datetime
import argparse

def valid_date(s):
    try:
        return datetime.datetime.strptime(s, "%Y-%m-%d")
    except ValueError:
        msg = "Not a valid date: '{0}'.".format(s)
        raise argparse.ArgumentTypeError(msg)


def time_tag(time_ranges, time_i):
    """given a time and time series, return a unique timestamp label"""
    for i, date in enumerate(time_ranges):
        if time_i < date:
            break
    return time_ranges[i - 1]