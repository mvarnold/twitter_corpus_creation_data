import sys
import torch
import random
from transformers.file_utils import is_tf_available, is_torch_available, is_torch_tpu_available
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, pipeline
from datasets import load_metric
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from tqdm.auto import tqdm
from pprint import pprint

def parse_args(args):
    parser = argparse.ArgumentParser(
        description="A command line interface for creating interactive ambient tweet embedding plots",
    )
    parser.add_argument(
        '-u', '--unlabeled_data',
        type=str,
        help='unlabeled data with pth to classify'
    )
    parser.add_argument(
        '-m', '--model_name',
        type=str,
        help='Transformer Model',
        default='sentence-transformers/all-mpnet-base-v2',
    )
    parser.add_argument(
        '-n', '--num_labels',
        type=int,
        help='Number of labels',
        default=2,
    )
    parser.add_argument(
        '-d', '--device',
        type=str,
        help='Choose from {"cpu", "cuda", "mps"} for cpu, gpu or apple silicon',
        default='cpu',
    )
    parser.add_argument(
        '-b', '--batch_size',
        type=int,
        help='Number of tweets to run inference on per batch',
        default=8,
    )
    return parser.parse_args(args)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    if is_torch_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if is_tf_available():
        import tensorflow as tf
 
        tf.random.set_seed(seed)


def load_model(args):
    """ Load fine-tuned models"""

    tokenizer = AutoTokenizer.from_pretrained(args.model_name+'_tokenizer',
                do_lower_case=True,
                local_files_only=True,
                )

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name+'_model', num_labels = args.num_labels).to(args.device)
    
    return model, tokenizer


def text_classifier(sbert_model, sbert_tokenizer, args):
    """ Classifier function to run binary classification and save the resulting labels """
    device_dict = {
        'cpu': -1,
        'cuda': 0,
        'mps':'mps',
    }
    batch_dict = {
        'cpu': None,
        'cuda': args.batch_size,
        'mps':args.batch_size,
    }


    # create classification pipeline
    pipe = pipeline("text-classification", model=sbert_model, tokenizer=sbert_tokenizer, device=device_dict[args.device])

    # load text to classify
    to_classify_df = pd.read_csv(args.unlabeled_data, sep='\t', lineterminator='\n')

    to_classify_df['Text'] = to_classify_df['Text'].astype(str)

    # filter nans
    to_classify_df = to_classify_df[to_classify_df['Text'].notnull()]


    batch_size = batch_dict[args.device]
    print("-" * 30)
    print(f"Streaming batch_size={batch_size}")
    dataset = (row.Text for i, row in to_classify_df.iterrows())
    labels = []
    for out in tqdm(pipe(dataset, batch_size=batch_size), total=len(to_classify_df)):
        labels.append(out)

    to_classify_df['Label'] = [i['label'] for i in labels]
    to_classify_df['Label_score'] = [i['score'] for i in labels]
    
    fname = str(Path(args.unlabeled_data).parent)+ '/' + str(Path(args.unlabeled_data).stem) + f"{args.model_name.split('/')[-1]}_labeled.tsv"
    print(fname)
    to_classify_df.to_csv(fname, sep='\t')    

    return to_classify_df


set_seed(42)

args = sys.argv[1:]
args = parse_args(args)

sbert_model, sbert_tokenizer = load_model(args)

df = text_classifier(sbert_model, sbert_tokenizer, args)

pd.set_option('display.width', 1000)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', -1)
print(df[['Text','Label']])

