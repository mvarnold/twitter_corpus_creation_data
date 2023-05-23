import pandas as pd
import sys
import argparse
from pathlib import Path

def parse_args(args):
    parser = argparse.ArgumentParser(
        description="Tweet id extractor",
    )
    parser.add_argument(
        '-f', '--filename',
        type=Path
    )
    return parser.parse_args(args)

def split_id(raw_id):
    """ Try to process id to interger"""
    if type(raw_id) is str:
        return str.split(raw_id, ':')[-1]
    else:
        return raw_id


def main(args=None):
    """ Load tweet file and save with only id and classification column"""
    if args is None:
        args = parse_args(sys.argv[1:])

    try:
        df = pd.read_json(args.filename)
    except ValueError:
        df = pd.read_csv(args.filename, sep='\t', lineterminator='\n')
        pass
    df['ID'] = df['id'].apply(split_id)

    try:
        labels = ['ID', 'Label', 'Label_score']
        print(df[labels].columns)

    except KeyError:
        labels = ['ID', 'Label']
        pass

    fname = args.filename
    df[labels].to_csv(fname.parent.joinpath(fname.stem+'_IDs_only'+fname.suffix), sep='\t')
    print("Saving to: ")
    print(fname.parent.joinpath(fname.stem+'_IDs_only'+fname.suffix) )



if __name__ == "__main__":
    main()