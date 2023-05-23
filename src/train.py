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
from pprint import pprint
import json
from tqdm.auto import tqdm

TOKENIZERS_PARALLELISM = True

def parse_args(args):
    parser = argparse.ArgumentParser(
        description="A command line interface for creating interactive ambient tweet embedding plots",
    )
    parser.add_argument(
        '-l', '--labeled_data',
        type=str,
        help='labeled data with pth',
        required=True
    )
    parser.add_argument(
        '-m', '--model_name',
        type=str,
        help='Transformer Model to use: \n'
             '{sentence-transformers/all-mpnet-base-v2,\n'
             'sentence-transformers/all-MiniLM-L6-v2,'
             'etc,}',
        default='sentence-transformers/all-mpnet-base-v2',
    )
    parser.add_argument(
        '-d', '--device',
        type=str,
        help='Choose from {"cpu", "cuda", "mps"} for cpu, gpu or apple silicon',
        default='cpu',
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
                                        device: str = 'cpu',
                                        dataname = None,
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
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels).to(device)
    except OSError:
        model = AutoModelForSequenceClassification.from_pretrained(model_name + '_model', num_labels=num_labels).to(
            device)

    # Create Metrics
    def compute_metrics(eval_pred):

        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)

        # 'micro', 'macro', etc. are for multi-label classification.
        # If you are running a binary classification, leave it as default or specify "binary" for average
        return metric.compute(predictions=predictions, references=labels)

    # Specifiy the arguments for the trainer
    training_args = TrainingArguments(
        output_dir=f'./results_{dataname}',  # output directory
        num_train_epochs=num_epochs,  # total number of training epochs
        per_device_train_batch_size=10,  # batch size per device during training
        per_device_eval_batch_size=10,  # batch size for evaluation
        warmup_steps=100,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir='./logs',  # directory for storing logs
        load_best_model_at_end=True,  # load the best model when finished training (default metric is loss)
        metric_for_best_model=metrics_name,  # select the base metrics
        logging_steps=50,  # log & save weights each logging_steps
        save_steps=50,
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
    metrics = trainer.evaluate()
    print(metrics)

    training_args.logging_dir = 'logs/'
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

    return trainer, model, tokenizer

def main():
    """ Train classifier with labeled data"""
    set_seed(42)

    args = sys.argv[1:]
    args = parse_args(args)

    df = pd.read_csv(args.labeled_data, sep='\t')

    data = df.Text
    target = df.Label

    if len(target.unique()) > 2:
        print(target.unique())
        print(target[target.isna()])
    assert len(target.unique()) == 2, "Should only contain two unique values"

    if args.model_name == "all":
        models_list = ['sentence-transformers/all-mpnet-base-v2',
                       'sentence-transformers/all-MiniLM-L12-v2',
                       'sentence-transformers/all-MiniLM-L6-v2',
                       'sentence-transformers/all-distilroberta-v1',
                       'sentence-transformers/paraphrase-MiniLM-L6-v2',
                       'sentence-transformers/paraphrase-MiniLM-L3-v2',
                       'sentence-transformers/distiluse-base-multilingual-cased-v1',
                       'intfloat/e5-base',
                       'intfloat/e5-large',
                       ]
        results_dict = {}
        for model_name in models_list:
            sbert_trainer, sbert_model, sbert_tokenizer = TextClassification_with_Transformer(
                model_name=model_name,
                Data=data,
                Target=target,
                test_size=0.33,
                max_length=512,
                num_labels=2,
                num_epochs=5,
                metrics_name='f1',
                device=args.device,
                dataname=args.labeled_data.split('/')[-1].split('.')[0])
            results_dict[model_name] = sbert_trainer.evaluate()
            model_pth = "models/" + Path(args.labeled_data).stem + f"_{model_name.split('/')[-1]}_model"
            sbert_model.save_pretrained(model_pth)
            tokenizer_pth = "models/" + Path(args.labeled_data).stem + f"_{model_name.split('/')[-1]}_tokenizer"
            sbert_tokenizer.save_pretrained(tokenizer_pth)

        print("Finished!")
        print(args.labeled_data)
        pprint(results_dict)
        with open(f"{args.labeled_data.split('/')[-1]}.json", 'w') as f:
            f.write(json.dumps(results_dict))

    else:
        sbert_trainer, sbert_model, sbert_tokenizer = TextClassification_with_Transformer(
            model_name=args.model_name,
            Data=data,
            Target=target,
            test_size=0.33,
            max_length=512,
            num_labels=2,
            num_epochs=5,
            metrics_name='f1',
            device=args.device)

        model_pth = "models/" + Path(args.labeled_data).stem + f"_{args.model_name.split('/')[-1]}_model"
        sbert_model.save_pretrained(model_pth)
        tokenizer_pth = "models/" + Path(args.labeled_data).stem + f"_{args.model_name.split('/')[-1]}_tokenizer"
        sbert_tokenizer.save_pretrained(tokenizer_pth)

if __name__ == "__main__":
    main()
