import csv
import os
import sys
import pickle 
import pandas as pd
import numpy as np
from dim_reduction_utils import load_balanced_cryptic_and_regular_data
from emmaemb.core import Emma
from emmaemb.vizualisation import get_knn_alignment_scores
from constants import DATA_PATH, EMBEDDINGS_PATH, IMG_OUTPUT_PATH, EMB_SPACES, CRYPTOBENCH_TRAIN_DATASET, SCPDB_DATASET

METRIC = 'euclidean'
N_TREES = 50
RANDOM_SHUFFLE = True

# emb_spaces = EMB_SPACES
emb_spaces = [("ESM2", f"{EMBEDDINGS_PATH}/esm2_t33_650M_UR50D/layer_33/chopped_1022_overlap_300"),
# ("ANKH", f"{EMBEDDINGS_PATH}/ankh_base/layer_None/chopped_1022_overlap_300"),
# ("ProstT5", f"{EMBEDDINGS_PATH}/Rostlab/ProstT5/layer_None/chopped_1022_overlap_300/"),
# ("ProtT5", f"{EMBEDDINGS_PATH}/Rostlab/prot_t5_xl_uniref50/layer_None/chopped_1022_overlap_300"),
# ("ESM1", f"{EMBEDDINGS_PATH}/esm1_t34_670M_UR100/layer_34/chopped_1022_overlap_300/"),
# ("ESMC", f"{EMBEDDINGS_PATH}/esmc-300m-2024-12/layer_None/chopped_1022_overlap_300/"),
]

datasets = [CRYPTOBENCH_TRAIN_DATASET, SCPDB_DATASET]

with open(f'{DATA_PATH}/protein_ids.pkl', 'rb') as f:
    protein_ids = pickle.load(f)

for emb_space in emb_spaces:
    embeddings_name = emb_space[0]
    emma = load_balanced_cryptic_and_regular_data(emb_space, datasets, DATA_PATH, protein_ids=protein_ids)
    
    if RANDOM_SHUFFLE:
        emma.metadata["binding_site"] = np.random.permutation(emma.metadata["binding_site"].to_numpy())

    emma.build_annoy_index(emb_space=embeddings_name, metric=METRIC, n_trees=N_TREES)
    df = get_knn_alignment_scores(emma, feature='binding_site', k=3, metric=METRIC, use_annoy=True, n_trees=N_TREES, annoy_metric=METRIC)
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

    heatmap_data.to_csv(f'{IMG_OUTPUT_PATH}/knn-binding-sites-RANDOM/{embeddings_name}.csv')
