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


def parse_args(args):
    parser = argparse.ArgumentParser(
        description="A command line interface for creating interactive ambient tweet embedding plots",
    )
    parser.add_argument(
        '-l', '--labeled_data',
        type=str,
        help='labeled data with pth'
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
        '-s', '--save',
        action='store_true',
        help='Flag to save model in same directory as data'
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



def TextClassification_with_Transformer(model_name: str,
                                        Data: pd.Series,
                                        Target: pd.Series,
                                        test_size: np.float64,
                                        max_length: int,
                                        num_labels: int,
                                        num_epochs: int,
                                        metrics_name: str,
                                        ):
    # Make data
    X = Data
    y = Target
    y = pd.factorize(y)[0]

    # Load Metrics
    metric = load_metric(metrics_name)

    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X.tolist(), y, test_size=test_size)

    # Call the Tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=True)
    except (OSError, ValueError):
        tokenizer = AutoTokenizer.from_pretrained(model_name + '_tokenizer',
                                                  do_lower_case=True,
                                                  local_files_only=True,
                                                  )

    # Encode the text
    train_encodings = tokenizer(X_train, truncation=True, padding=True, max_length=max_length)
    valid_encodings = tokenizer(X_test, truncation=True, padding=True, max_length=max_length)

    class MakeTorchData(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
            item["labels"] = torch.tensor([self.labels[idx]])
            return item

        def __len__(self):
            return len(self.labels)

    # convert our tokenized data into a torch Dataset
    train_dataset = MakeTorchData(train_encodings, y_train.ravel())
    valid_dataset = MakeTorchData(valid_encodings, y_test.ravel())

    # Call Model
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels).to("mps")
    except OSError:
        model = AutoModelForSequenceClassification.from_pretrained(model_name + '_model', num_labels=num_labels).to(
            "cuda")

    # Create Metrics
    def compute_metrics(eval_pred):

        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)

        # 'micro', 'macro', etc. are for multi-label classification. 
        # If you are running a binary classification, leave it as default or specify "binary" for average
        return metric.compute(predictions=predictions, references=labels)

    # Specifiy the arguments for the trainer
    training_args = TrainingArguments(
        output_dir='./results',  # output directory
        num_train_epochs=num_epochs,  # total number of training epochs
        per_device_train_batch_size=10,  # batch size per device during training
        per_device_eval_batch_size=10,  # batch size for evaluation
        warmup_steps=100,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir='./logs',  # directory for storing logs
        load_best_model_at_end=True,  # load the best model when finished training (default metric is loss)
        metric_for_best_model=metrics_name,  # select the base metrics
        logging_steps=100,  # log & save weights each logging_steps
        save_steps=100,
        evaluation_strategy="steps",  # evaluate each `logging_steps`
    )

    # Call the Trainer
    trainer = Trainer(
        model=model,  # the instantiated Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=valid_dataset,  # evaluation dataset
        compute_metrics=compute_metrics,  # the callback that computes metrics of interest
    )

    # Train the model
    trainer.train()

    # Call the summary
    trainer.evaluate()

    return trainer, model, tokenizer


def text_classifier(sbert_model, sbert_tokenizer, args):
    """ Classifier function to run binary classification and save the resulting labels """

    # create classification pipeline
    pipe = pipeline("text-classification", model=sbert_model, tokenizer=sbert_tokenizer, device=0)

    # load text to classify
    to_classify_df = pd.read_csv(args.unlabeled_data, sep='\t')

    to_classify_df = to_classify_df[to_classify_df['Text'].notnull()]

    batch_size = 8
    print("-" * 30)
    print(f"Streaming batch_size={batch_size}")
    dataset = (row.Text for i, row in to_classify_df.iterrows())
    labels = []
    for out in tqdm(pipe(dataset, batch_size=batch_size), total=len(to_classify_df)):
        labels.append(out)

    to_classify_df['Label'] = [i['label'] for i in labels]
    to_classify_df['Label_score'] = [i['score'] for i in labels]

    fname = str(Path(args.unlabeled_data).parent) + '/' + str(Path(args.unlabeled_data).stem) + "_labeled.tsv"
    print(fname)
    to_classify_df.to_csv(fname, sep='\t')

    return to_classify_df


set_seed(42)

args = sys.argv[1:]
args = parse_args(args)

df = pd.read_csv(args.labeled_data, sep='\t')
data = df.Text
target = df.Label

sbert_trainer, sbert_model, sbert_tokenizer = TextClassification_with_Transformer(
    model_name=args.model_name,
    Data=data,
    Target=target,
    test_size=0.33,
    max_length=512,
    num_labels=2,
    num_epochs=5,
    metrics_name='f1')
if args.save:
    model_pth = Path(args.labeled_data).stem + '_model'
    sbert_model.save_pretrained(model_pth)
    tokenizer_pth = Path(args.labeled_data).stem + '_tokenizer'
    sbert_tokenizer.save_pretrained(tokenizer_pth)

print(text_classifier(sbert_model, sbert_tokenizer, args))
