import csv
import os
import sys
import pickle 
import pandas as pd
import numpy as np
import argparse
from dim_reduction_utils import load_balanced_cryptic_and_regular_data
from emmaemb.core import Emma
from emmaemb.vizualisation import get_knn_alignment_scores
from constants import DATA_PATH, EMBEDDINGS_PATH, IMG_OUTPUT_PATH, EMB_SPACES, CRYPTOBENCH_TRAIN_DATASET, SCPDB_DATASET

parser = argparse.ArgumentParser()
parser.add_argument('--metric', type=str, default='euclidean')
args = parser.parse_args()

METRIC = args.metric
N_TREES = 50
RANDOM_SHUFFLE = False
K = [3, 5, 10, 50, 100, 200]

# emb_spaces = EMB_SPACES
emb_spaces = EMB_SPACES

datasets = [CRYPTOBENCH_TRAIN_DATASET, SCPDB_DATASET]

with open(f'{DATA_PATH}/protein_ids.pkl', 'rb') as f:
    protein_ids = pickle.load(f)

for emb_space in emb_spaces:
    embeddings_name = emb_space[0]
    emma = load_balanced_cryptic_and_regular_data(emb_space, datasets, DATA_PATH, protein_ids=protein_ids)

    if RANDOM_SHUFFLE:
        emma.metadata["binding_site"] = np.random.permutation(emma.metadata["binding_site"].to_numpy())

    emma.build_annoy_index(emb_space=embeddings_name, metric=METRIC, n_trees=N_TREES)

    for k in K:
        df = get_knn_alignment_scores(emma, feature='binding_site', k=k, metric=METRIC, use_annoy=True, n_trees=N_TREES, annoy_metric=METRIC)
        heatmap_data = (
            df.groupby(["Class", "Embedding"])["Fraction"]
            .mean()
            .unstack()  # Reshape to have Classes as rows and Embeddings as columns
        )

        class_counts = df.groupby("Class").size()

        heatmap_data.index = [
            f"{feature_class} (n = {int(count / len(df['Embedding'].unique()))})"
            for feature_class, count in zip(
                heatmap_data.index, class_counts[heatmap_data.index]
            )
        ]

        heatmap_data.to_csv(f'{IMG_OUTPUT_PATH}/knn-binding-sites/{METRIC}/{embeddings_name},k={k}.csv')
