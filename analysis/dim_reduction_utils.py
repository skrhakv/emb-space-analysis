import csv
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from emmaemb.core import Emma


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

SUBSET_SIZE = 308 # number of proteins in the scPDB subset so there is the same number of cryptic binding residues and regular binding residues 
def load_balanced_cryptic_and_regular_data(emb_space, datasets, DATA_PATH):
    '''Balance number of cryptic residues and regular binding residues, but keep all non-binding residues.'''
    (embeddings_name, embeddings_path) = emb_space
    
    embeddings = []
    feature_data = []

    for dataset in datasets:

        with open(f"{DATA_PATH}/{dataset}", 'r') as f:
            reader = csv.reader(f, delimiter=';')

            for ii, row in enumerate(reader):
                protein_id = row[0] + row[1]
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
                BINDING_FLAG = 'CRYPTIC-BINDING' if dataset == 'train.txt' else 'BINDING'
                for i in range(len(sequence)):
                    feature_data.append([sequence[i], BINDING_FLAG if i in annotation else 'NON-BINDING'])

                # take only first 'SUBSET_SIZE' binding residues from scPDB
                if ii > SUBSET_SIZE and dataset == 'scPDB_enhanced_binding_sites_translated.csv':
                    break

    print(len(feature_data), len(embeddings))
    concatenated_embeddings_path = f"{DATA_PATH}/concatenated-embeddings/{embeddings_name}_binding_site_embeddings.npy"
    embeddings = np.concatenate(embeddings, axis=0)
    np.save(concatenated_embeddings_path, embeddings) 

    del embeddings

    feature_data = pd.DataFrame.from_records(feature_data, columns=["amino acid", "binding_site"])

    emma = Emma(feature_data=feature_data)
    emma.add_emb_space(
        embeddings_source=f"{DATA_PATH}/concatenated-embeddings/{embeddings_name}_binding_site_embeddings.npy",
        emb_space_name=embeddings_name)
    return emma