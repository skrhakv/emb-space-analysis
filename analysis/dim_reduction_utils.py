import csv
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from emmaemb.core import Emma
from constants import EMBEDDINGS_PATH, EMB_SPACES


def prepare_data(emma, embedding_space):
    embeddings = emma.emb[embedding_space]["emb"]
    labels = emma.metadata["binding_site"].apply(lambda x: 1 if x == 'BINDING' else 2 if x == 'CRYPTIC-BINDING' else 0).values
    return embeddings, labels

def run_PCA(n_components, embeddings):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_components)
    embeddings = pca.fit_transform(embeddings)
    variance_explained = pca.explained_variance_ratio_
    return embeddings, variance_explained

def run_tSNE(n_components, embeddings):
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=n_components)
    embeddings = tsne.fit_transform(embeddings)
    return embeddings

def run_UMAP(n_components, embeddings):
    from umap import UMAP
    umap = UMAP(n_components=n_components, n_neighbors=50, min_dist=0.8, metric='euclidean', random_state=42)
    embeddings = umap.fit_transform(embeddings)
    return embeddings

def run_PHATE(n_components, embeddings):
    from phate import PHATE
    phate = PHATE(n_components=n_components)
    embeddings = phate.fit_transform(embeddings)
    return embeddings

def plot_kde(ax, x, y, cmap, label):
    from scipy.stats import gaussian_kde

    if len(x) == 0:
        return
    # Calculate the point density
    xy = np.vstack([x, y])
    kde = gaussian_kde(xy)
    
    # Create grid
    x_padding = (x.max() - x.min()) * 0.25
    y_padding = (y.max() - y.min()) * 0.25
    xmin, xmax = x.min() - x_padding, x.max() + x_padding
    ymin, ymax = y.min() - y_padding, y.max() + y_padding
    xi, yi = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    zi = kde(np.vstack([xi.flatten(), yi.flatten()]))
    
    # Plot contours
    ax.contour(xi, yi, zi.reshape(xi.shape), colors=cmap, linewidths=1.5)
    # Add a representative legend entry
    ax.plot([], [], color=cmap, label=label)

def get_silhouette_score(embeddings, labels):
    score = silhouette_score(embeddings, labels)
    return score

def plot_scatter(embeddings, labels, x_idx, y_idx, emb_space, method, path=None):
    if path is not None and not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    
    pc_x = embeddings[:, x_idx]
    pc_y = embeddings[:, y_idx]
    fig, axes = plt.subplots(1,2,figsize=(12, 5))

    ax = axes[0]

    # Plot each class with KDE contours (binding - blue, non binding - red, cryptic binding - green)
    plot_kde(ax, pc_x[labels == 0], pc_y[labels == 0], 'blue', 'Non-binding')
    plot_kde(ax, pc_x[labels == 1], pc_y[labels == 1], 'red', 'Binding')
    plot_kde(ax, pc_x[labels == 2], pc_y[labels == 2], 'green', 'Cryptic Binding')

    x_padding = (pc_x.max() - pc_x.min()) * 0.1
    y_padding = (pc_y.max() - pc_y.min()) * 0.1
    ax.set_xlim(pc_x.min() - x_padding, pc_x.max() + x_padding)
    ax.set_ylim(pc_y.min() - y_padding, pc_y.max() + y_padding)
    ax.set_xlabel(f'PC{x_idx + 1}')
    ax.set_ylabel(f'PC{y_idx + 1}')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)

    ax = axes[1]
    # Add scatter points with low alpha to see individual residues
    ax.scatter(pc_x[labels == 0], pc_y[labels == 0], alpha=0.2, color='blue', s=5, label='Non-binding')
    ax.scatter(pc_x[labels == 1], pc_y[labels == 1], alpha=0.2, color='red', s=5, label='Binding')
    ax.scatter(pc_x[labels == 2], pc_y[labels == 2], alpha=0.2, color='green', s=5, label='Cryptic Binding')

    x_padding = (pc_x.max() - pc_x.min()) * 0.1
    y_padding = (pc_y.max() - pc_y.min()) * 0.1
    ax.set_xlim(pc_x.min() - x_padding, pc_x.max() + x_padding)
    ax.set_ylim(pc_y.min() - y_padding, pc_y.max() + y_padding)
    ax.set_xlabel(f'PC{x_idx + 1}')
    ax.set_ylabel(f'PC{y_idx + 1}')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    if path:
        plt.savefig(path)
    plt.show()

def plot_scatter1(embeddings, labels, x_idx, y_idx, emb_space, method, path=None):
    if path is not None and not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    
    pc_x = embeddings[:, x_idx]
    pc_y = embeddings[:, y_idx]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    x_padding = (pc_x.max() - pc_x.min()) * 0.1
    y_padding = (pc_y.max() - pc_y.min()) * 0.1

    # Plot for Non-binding (labels == 0)
    ax = axes[0]
    ax.scatter(pc_x[labels == 0], pc_y[labels == 0], alpha=0.2, color='blue', s=5, label='Non-binding')
    ax.set_xlim(pc_x.min() - x_padding, pc_x.max() + x_padding)
    ax.set_ylim(pc_y.min() - y_padding, pc_y.max() + y_padding)
    ax.set_xlabel(f'PC{x_idx + 1}')
    ax.set_ylabel(f'PC{y_idx + 1}')
    ax.set_title('Non-binding')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)

    # Plot for Binding (labels == 1)
    ax = axes[1]
    ax.scatter(pc_x[labels == 1], pc_y[labels == 1], alpha=0.2, color='red', s=5, label='Binding')
    ax.set_xlim(pc_x.min() - x_padding, pc_x.max() + x_padding)
    ax.set_ylim(pc_y.min() - y_padding, pc_y.max() + y_padding)
    ax.set_xlabel(f'PC{x_idx + 1}')
    ax.set_ylabel(f'PC{y_idx + 1}')
    ax.set_title('Binding')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)

    # Plot for Cryptic Binding (labels == 2)
    ax = axes[2]
    ax.scatter(pc_x[labels == 2], pc_y[labels == 2], alpha=0.2, color='green', s=5, label='Cryptic Binding')
    ax.set_xlim(pc_x.min() - x_padding, pc_x.max() + x_padding)
    ax.set_ylim(pc_y.min() - y_padding, pc_y.max() + y_padding)
    ax.set_xlabel(f'PC{x_idx + 1}')
    ax.set_ylabel(f'PC{y_idx + 1}')
    ax.set_title('Cryptic Binding')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)

    fig.suptitle(f'Silhouette Score: {get_silhouette_score(embeddings[:, [x_idx, y_idx]], labels):.4f}\n{emb_space} - {method}', fontsize=12)

    plt.tight_layout()
    if path:
        plt.savefig(path)
    plt.show()

def load_balanced_cryptic_and_regular_data(emb_space, datasets, protein_ids=None):
    '''Balance number of cryptic residues and regular binding residues, but keep all non-binding residues. 
    Actually this is a huge technical dept and if protein_ids is None then there is no balancing'''
    (embeddings_name, embeddings_path) = emb_space
    
    embeddings = []
    feature_data = []

    for dataset in datasets:
        
        BINDING_FLAG = 'CRYPTIC-BINDING' if (dataset[-9:] == 'train.txt' or dataset[-8:] == 'test.txt') else 'BINDING'
        binding_type = 'CRYPTIC' if BINDING_FLAG == 'CRYPTIC-BINDING' else 'REGULAR'
        with open(dataset, 'r') as f:
            reader = csv.reader(f, delimiter=';')

            for ii, row in enumerate(reader):
                protein_id = row[0] + row[1]

                if protein_ids is not None and f'{protein_id}_{binding_type}' not in protein_ids:
                    continue

                annotation = row[3].split(' ')
                annotation = [int(i[1:]) for i in annotation]
                sequence = row[4]

                path = f"{embeddings_path}/{protein_id}.npy"
                if not os.path.exists(path):
                    continue
                
                embedding = np.load(path)

                if embedding.shape[0] != len(sequence):
                    continue

                embeddings.append(embedding)
                
                for i in range(len(sequence)):
                    feature_data.append([sequence[i], BINDING_FLAG if i in annotation else 'NON-BINDING'])

    concatenated_embeddings_path = f"{EMBEDDINGS_PATH}/concatenated-embeddings/{embeddings_name}_binding_site_embeddings.npy"
    embeddings = np.concatenate(embeddings, axis=0)
    np.save(concatenated_embeddings_path, embeddings) 

    del embeddings

    feature_data = pd.DataFrame.from_records(feature_data, columns=["amino acid", "binding_site"])

    emma = Emma(feature_data=feature_data)
    emma.add_emb_space(
        embeddings_source=f"{EMBEDDINGS_PATH}/concatenated-embeddings/{embeddings_name}_binding_site_embeddings.npy",
        emb_space_name=embeddings_name)
    return emma

def load_row(row):
    protein_id = row[0] + row[1]
    annotation = row[3].split(' ')
    annotation = [int(i[1:]) for i in annotation]
    sequence = row[4]

    # load embeddings for this protein from all embedding spaces
    these_embeddings = {}
    length, have_same_length = 0, True
    for i, (embeddings_name, embeddings_path) in enumerate(EMB_SPACES):
        path = f"{embeddings_path}/{protein_id}.npy"
        if not os.path.exists(path):
            have_same_length = False
            break
        embedding = np.load(path)
        these_embeddings[embeddings_name] = embedding
        # check that all embedding spaces have the same length for this protein
        if i == 0:
            length = embedding.shape[0]
        else:
            if embedding.shape[0] != length:
                have_same_length = False
                break
            
    # check that all embedding spaces have the same length for this protein
    if not have_same_length:
        return None, None, None, None

    return protein_id, annotation, sequence, these_embeddings

def load_dataset_with_all_balanced_classes():
    import csv
    import os
    import sys
    import pandas as pd
    import numpy as np
    from emmaemb.core import Emma
    sys.path.append('/home/unix/vkrhk/EmmaEmb/EmmaEmb/analysis')
    from constants import EMBEDDINGS_PATH, EMB_SPACES, CRYPTOBENCH_TRAIN_DATASET, SCPDB_DATASET
    import random

    def shuffle_residues(length):
        randomly_ordered_residues = list(range(length))
        random.shuffle(randomly_ordered_residues)
        return randomly_ordered_residues

    def get_embeddings(embeddings, these_embeddings, embedding_indices):
        for embeddings_name in these_embeddings:
            if embeddings_name not in embeddings:
                embeddings[embeddings_name] = []

            collected_embeddings = these_embeddings[embeddings_name][embedding_indices]
            embeddings[embeddings_name].append(collected_embeddings)

    def add_protein_id(protein_id, protein_ids, type):
        assert protein_id not in protein_ids, f"Protein {protein_id} is already added"
        protein_ids.add(f'{protein_id}_{type}')

    # collect data from the cryptobench dataset to count the number of binding residues in it
    protein_ids = set()
    feature_data = []
    embeddings = {}
    number_of_cryptic_residues = 0
    number_of_non_binding_residues = 0

    # with open(CRYPTOBENCH_TEST_DATASET, 'r') as f:
    with open(CRYPTOBENCH_TRAIN_DATASET, 'r') as f:
        reader = csv.reader(f, delimiter=';')
        for row in reader:
            protein_id, annotation, sequence, these_embeddings = load_row(row)
            if protein_id is None:
                continue
            for i in annotation:
                feature_data.append([sequence[i], 'CRYPTIC-BINDING'])
                number_of_cryptic_residues += 1
            # add NON-BINDING residues until we have the same number of NON-BINDING and BINDING residues in the embeddings (loop randomly the non-binding residues)
            randomly_ordered_residues = shuffle_residues(len(sequence))
            embedding_indices = annotation.copy()  # start with the binding residues
            for i, residue_index in enumerate(randomly_ordered_residues):
                if residue_index in annotation: # skip binding residues
                    continue
                if i > round(len(annotation) / 2):
                    break
                feature_data.append([sequence[residue_index], 'NON-BINDING'])
                embedding_indices.append(i)
                number_of_non_binding_residues += 1

            add_protein_id(protein_id, protein_ids, 'CRYPTIC')

            get_embeddings(embeddings, these_embeddings, embedding_indices)

    # import pickle
    # this is a list of scPDB protein ids that have seq similarity > 10 % with the LIGYSIS dataset
    # with open('/home/unix/vkrhk/EmmaEmb/data/banned_protein_ids.pkl', 'rb') as f:
    #     banned_protein_ids = pickle.load(f)

    number_of_regular_binding_residues = 0

    # with open(LIGYSIS_DATASET, 'r') as f:
    with open(SCPDB_DATASET, 'r') as f:
        reader = csv.reader(f, delimiter=';')
        for row in reader:
            if number_of_regular_binding_residues >= number_of_cryptic_residues:
                break

            protein_id = row[0] + row[1]
            protein_id, annotation, sequence, these_embeddings = load_row(row)
            if protein_id is None:
                continue
            # if protein_id in banned_protein_ids:
            #     continue

            for i in annotation:
                feature_data.append([sequence[i], 'BINDING'])
                number_of_regular_binding_residues += 1

            randomly_ordered_residues = shuffle_residues(len(sequence))
            embedding_indices = annotation.copy()  # start with the binding residues
            this_number_of_non_binding_residues = 0
            for residue_index in randomly_ordered_residues:

                if residue_index in annotation: # skip binding residues
                    continue
                if this_number_of_non_binding_residues > round(len(annotation) / 2) or \
                    number_of_cryptic_residues <= number_of_non_binding_residues:
                    break
                
                feature_data.append([sequence[residue_index], 'NON-BINDING'])
                embedding_indices.append(residue_index)
                number_of_non_binding_residues += 1
                this_number_of_non_binding_residues += 1

            add_protein_id(protein_id, protein_ids, 'REGULAR')
            get_embeddings(embeddings, these_embeddings, embedding_indices)

    for embeddings_name in embeddings:
        concatenated_embeddings_path = f"{EMBEDDINGS_PATH}/concatenated-embeddings/{embeddings_name}_binding_site_embeddings.npy"
        embeddings[embeddings_name] = np.concatenate(embeddings[embeddings_name], axis=0)
        np.save(concatenated_embeddings_path, embeddings[embeddings_name])  

    feature_data = pd.DataFrame.from_records(feature_data, columns=["amino acid", "binding_site"])

    # initiate Emma object and load embedding spaces
    emma = Emma(feature_data=feature_data)
    for embeddings_name, _ in EMB_SPACES:
        emma.add_emb_space(
            embeddings_source=f"{EMBEDDINGS_PATH}/concatenated-embeddings/{embeddings_name}_binding_site_embeddings.npy",
            emb_space_name=embeddings_name)
        
    return emma


def load_imbalanced_dataset():
    from constants import SCPDB_DATASET, CRYPTOBENCH_TRAIN_DATASET

    feature_data = []
    embeddings = {}
    for dataset, binding_type in [(SCPDB_DATASET, 'BINDING'), (CRYPTOBENCH_TRAIN_DATASET, 'CRYPTIC-BINDING')]:
        with open(dataset, 'r') as f:
            reader = csv.reader(f, delimiter=';')
            for row in reader:
                protein_id, annotation, sequence, these_embeddings = load_row(row)
                if protein_id is None:
                    continue
                for i in annotation:
                    feature_data.append([sequence[i], binding_type])
                for emb_space in these_embeddings.keys():
                    if emb_space not in embeddings:
                        embeddings[emb_space] = []
                    embeddings[emb_space].append(these_embeddings[emb_space])
    for emb_space in embeddings.keys():
        embeddings[emb_space] = np.concatenate(embeddings[emb_space], axis=0)
        np.save(f"{EMBEDDINGS_PATH}/concatenated-embeddings/{emb_space}_binding_site_embeddings.npy", embeddings[emb_space])
    feature_data = pd.DataFrame.from_records(feature_data, columns=["amino acid", "binding_site"])
    emma = Emma(feature_data=feature_data)
    for embeddings_name, _ in EMB_SPACES:
        emma.add_emb_space(
            embeddings_source=f"{EMBEDDINGS_PATH}/concatenated-embeddings/{embeddings_name}_binding_site_embeddings.npy",
            emb_space_name=embeddings_name)
    return emma