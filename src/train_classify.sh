python train.py -l "../data/Hom"

python classify.py -u "../data/Nuclear/Nuclear_2016-01-01_2022-12-31_D_False_state.tsv" -m "models/Nuclear_2016-01-01_2022-03-15_D_False_state_top1000_labeled_all-mpnet-base-v2" -d 'mps'