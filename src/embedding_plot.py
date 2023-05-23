import os
import sys
import argparse
from bson import json_util
import gzip
import json
import time
import datetime
import pandas as pd
from sentence_transformers import SentenceTransformer
from umap import UMAP

import matplotlib.pyplot as plt
from pprint import pprint