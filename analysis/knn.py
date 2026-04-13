import os

import numpy as np
import argparse
from dim_reduction_utils import load_dataset_with_all_balanced_classes, load_imbalanced_cryptic_and_regular_data, mean_center
from emmaemb.vizualisation import get_knn_alignment_scores
from constants import IMG_OUTPUT_PATH, EMB_SPACES, CRYPTOBENCH_TRAIN_DATASET, SCPDB_DATASET


parser = argparse.ArgumentParser()
parser.add_argument('--metric', type=str, default='euclidean')
parser.add_argument('--mean_centering', action='store_true')
parser.add_argument('--imbalanced', action='store_true')
args = parser.parse_args()

METRIC = args.metric
N_TREES = 100
K = [5, 10, 20, 30, 50, 100]

emb_spaces = EMB_SPACES

datasets = [CRYPTOBENCH_TRAIN_DATASET, SCPDB_DATASET]

def run_imbalanced():
    for emb_space in emb_spaces:
        emb_space_name = emb_space[0]
        print(f"Processing embedding space: {emb_space_name}")
        emma = load_imbalanced_cryptic_and_regular_data(emb_space, datasets)

        print(f"loaded dataset with {len(emma.metadata)} samples.")

        if args.mean_centering:
            print('Applying mean-centering to embedding spaces...')
            mean_center(emma, emb_spaces=[emb_space_name])
        print(f"Building Annoy index for embedding space: {emb_space_name} with metric: {METRIC} and n_trees: {N_TREES}")

        annoy_index_path = f'/media/drive2/vkrhk/annoy-ranks/{emb_space_name}_{METRIC}_{N_TREES}{'_mean-centering' if args.mean_centering else ""}.npy'
        if METRIC == "cosine":
            distance_metric_annoy = "angular"
        elif METRIC == "cityblock":
            distance_metric_annoy = "manhattan"
        elif METRIC == "euclidean":
            distance_metric_annoy = "euclidean"

        if os.path.exists(annoy_index_path):
            print(f"Loading precomputed Annoy ranks from {annoy_index_path}...")
            if "annoy_ranks" not in emma.emb[emb_space_name]:
                emma.emb[emb_space_name]["annoy_ranks"] = {}
            if distance_metric_annoy not in emma.emb[emb_space_name]["annoy_ranks"]:
                emma.emb[emb_space_name]["annoy_ranks"][distance_metric_annoy] = {}
            if N_TREES not in emma.emb[emb_space_name]["annoy_ranks"][distance_metric_annoy]:
                emma.emb[emb_space_name]["annoy_ranks"][distance_metric_annoy][N_TREES] = {}
            emma.emb[emb_space_name]["annoy_ranks"][distance_metric_annoy][N_TREES] = np.load(annoy_index_path)# , allow_pickle=True)
        else:
            emma.build_annoy_index(emb_space=emb_space_name, metric=METRIC, n_trees=N_TREES)
            np.save(annoy_index_path, emma.emb[emb_space_name]["annoy_ranks"][distance_metric_annoy][N_TREES])
        for k in K:
            print(f"\tCalculating k-NN alignment scores for k={k}...")
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
            heatmap_data.to_csv(f'{IMG_OUTPUT_PATH}/knn-binding-sites/imbalanced/{METRIC}/{emb_space_name},k={k}{",mean_centered" if args.mean_centering else ""}.csv')
            
print(f"Using distance metric: {METRIC}")
print(f"Using mean-centering: {args.mean_centering}")
print(f"Using imbalanced dataset: {args.imbalanced}")

if args.imbalanced:
    run_imbalanced()
    exit()

else:
    emma = load_dataset_with_all_balanced_classes()

if args.mean_centering:
    print('Applying mean-centering to embedding spaces...')
    mean_center(emma, emb_spaces=['ESM2', 'ANKH', 'ProstT5', 'ProtT5'])

for embeddings_name, _ in emb_spaces:
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
    heatmap_data.to_csv(f'{IMG_OUTPUT_PATH}/knn-binding-sites/{METRIC}/k={k}{",mean_centered" if args.mean_centering else ""}{\
        ".imbalanced" if args.imbalanced else ""}.csv')