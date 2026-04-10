import csv
import os
import sys
import pickle 
import pandas as pd
import numpy as np
import argparse
from sklearn.utils import shuffle
sys.path.append('/home/unix/vkrhk/EmmaEmb/EmmaEmb/analysis')
from dim_reduction_utils import load_dataset_with_all_balanced_classes
from emmaemb.core import Emma
from emmaemb.vizualisation import get_knn_alignment_scores
from constants import DATA_PATH, EMBEDDINGS_PATH, IMG_OUTPUT_PATH, EMB_SPACES, CRYPTOBENCH_TRAIN_DATASET, SCPDB_DATASET


parser = argparse.ArgumentParser()
parser.add_argument('--metric', type=str, default='euclidean')
args = parser.parse_args()

METRIC = args.metric
N_TREES = 100
emb_spaces = EMB_SPACES
emma = load_dataset_with_all_balanced_classes()

emma.metadata, emma.emb['ESM2']['emb'], emma.emb['ANKH']['emb'], emma.emb['ProtT5']['emb'], emma.emb['ProstT5']['emb'] = shuffle(\
                                          emma.metadata, emma.emb['ESM2']['emb'], \
                                                          emma.emb['ANKH']['emb'],
                                                           emma.emb['ProtT5']['emb'],
                                                            emma.emb['ProstT5']['emb'], random_state=42)
emma.metadata.reset_index(inplace=True, drop=True)

results = []
for random_shuffle in [0, 0.1, 0.2, 0.5, 1.0]:
    number_of_elements = round(len(emma.metadata["binding_site"]) * random_shuffle)
    print(f'Randomly shuffling {number_of_elements} elements of the binding_site feature')
    emma.metadata.loc[0:number_of_elements, "binding_site"] = shuffle(emma.metadata.loc[0:number_of_elements, "binding_site"].to_numpy())

    for embeddings_name, _ in emb_spaces:
        emma.build_annoy_index(emb_space=embeddings_name, metric=METRIC, n_trees=N_TREES)

    df = get_knn_alignment_scores(emma, feature='binding_site', k=10, metric=METRIC, use_annoy=True, n_trees=N_TREES, annoy_metric=METRIC)
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
    
    with open(f'{IMG_OUTPUT_PATH}/shuffle-dataset/{METRIC}/{random_shuffle}.pkl', 'wb') as f:
        pickle.dump(heatmap_data, f)