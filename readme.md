# TweetDB Corpus Curation
Tools to rapidly train tweet classifiers powered by pre-trained, transformer based sentence embeddings.

## Installation

To do:


## Query

Querying for relevant tweets can be done with single anchor keywords
or lists of keywords. 
A typical query might look something like this: 

```bash
python query.py -w "Cancer" -b "2021-01-01" -e "2022-08-30"
```

Matching tweets will be cached locally in this project's data directory by default,
saving both the raw json files as well as tsv files with only selected text fields and metadata.

## Labeling

Without committing to a labeling platform,
the fastest way to label data is to grab the top1000.tsv subsample in Excel,
and label non-relevant tweets with a 0 and relevant tweets with a 1.
Make sure to re-export the file in a .tsv format when finished with "_labeled" post-pended to the filename.

Another option is to set up a labeling project on computationalstorylab.lighttag.io. This provides support for multi


## Fine-tuning

To train and save a model run the train script. You can specify a pretrained model to 
download from the [huggingface repository](https://www.sbert.net/docs/pretrained_models.html).
For most cases, I'd use `all-mpnet-base-v2` for the best accuracy,
and `all-MiniLM-L6-v2` for faster training and inference speed,
with slightly lower accuracy.

To run the script with labeled data file, `-l` and sentence transformer model,`-m`, use: 

```bash
python train.py 
-l "../data/Cancer/Cancer_2021-01-01 00:00:00_2022-08-30 00:00:00_D_False_top1000_labeled.tsv"
-m "sentence-transformers/all-MiniLM-L6-v2"
-d "cpu"
```

The model will be saved by default in the `src/models/` directory with the same filename stem as the datafile, so you know what it was trained on.

## Using the model for inference
To classify tweets with the fine-tuned model, we need to run classify with a datafile of unlabeled tweets, and a model to apply.
The model should be the same directory name as in your `models/` directory, but without the `_model` suffix.
```bash
python classify.py 
-u "../data/Cancer/Cancer_2021-01-01 00:00:00_2022-08-30 00:00:00_D_False.tsv" 
-m "models/Cancer_2021-01-01 00:00:00_2022-08-30 00:00:00_D_False_top1000_labeled_all-MiniLM-L6-v2" 
-d "cpu"
```
To speed things up, you can specify `-d "cuda"` to run on a gpu if your machine has a cuda enabled version of pytorch installed.

## Compare 



## Transferring from local machine to remote
Any point after labeling is a good point to send the labeled data to the vacc if you have an environment set up there.
You can train the pretrained models in under ten minutes on a CPU, 
it will be much faster to deploy the model to label a full tweet dataset on a machine with GPUs.

To send things over, run the `send_data_to_HPC.sh` script with the anchor and base filename.
```bash
zsh send_data_to_HPC.sh Cancer "Cancer_2016-01-01 00:00:00_2022-08-30 00:00:00_D_False_state"
```

Keep in mind that this assumes the labeled data is matches the filename structure which is printed to screen and the data directory structure on the vacc. Importantly, it's moving both labeled data, unlabeled data, and a model if one already exists.