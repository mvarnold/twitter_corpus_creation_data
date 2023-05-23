#!/bin/zsh

python tweet_id_extractor.py -f '../data/Nuclear/Nuclear_2016-01-01_2022-03-15_D_False_state_top1000_labeled.tsv'
python tweet_id_extractor.py -f '../data/Nuclear/Nuclear_2016-01-01_2022-12-31_D_False_stateNuclear_2016-01-01_2022-03-15_D_False_state_top1000_labeled_all-mpnet-base-v2_labeled.tsv'
echo Wind
python tweet_id_extractor.py -f '../data/Wind/Wind_2020-01-01_2021-01-01_D_False_state_top1000_labeled.tsv'
echo Wind2
python tweet_id_extractor.py -f '../data/Wind/Wind_2020-01-01_2021-01-01_D_False_stateWind_2020-01-01_2021-01-01_D_False_state_top1000_labeled_all-mpnet-base-v2_labeled.tsv'
echo Solar
python tweet_id_extractor.py -f '../data/Solar/Solar_2016-01-01_2022-08-30_D_False_state_labeled.tsv'
python tweet_id_extractor.py -f '../data/Solar/Solar_2016-01-01_2022-12-31_D_False_stateSolar_2016-01-01 00:00:00_2022-08-30 00:00:00_D_False_state_top1000_labeled_all-mpnet-base-v2_labeled.tsv'
