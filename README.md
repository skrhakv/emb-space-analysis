# EmmaEmb Analysis
This repository contains supporting code for the [EmmaEmb](https://github.com/broadinstitute/EmmaEmb) framework, as described in the [preprint](https://www.biorxiv.org/content/10.1101/2024.06.21.600139v2.full) *"Decoding protein language models: insights from embedding space analysis"*.

## Overview
This project provides tools for analyzing protein language model (pLM) embeddings using the following datasets:
- [CryptoBench](https://osf.io/pz4a9/overview): Utilized for cryptic binding site annotations.
- [scPDB-enhanced](https://zenodo.org/records/18271517): Utilized for regular binding site annotations.


The analysis specifically focuses on:
- kNN Scoring: Calculating k-Nearest Neighbor scores within the embedding space.
- Classification: Training Multi-Layer Perceptron (MLP) classifiers for multi-class ligand binding site datasets.
- Visualizations: Generating the manuscript's figures and Plotly visualizations.

The training data together with trained model can be [downloaded from this link](https://s3.cl4.du.cesnet.cz/a93fcece52e6da0dd335b4459d47b0aebb74836b:share/emmaemb.classifier-data.tar.gz).
## Structure
- `src`: Scripts for kNN calculations and MLP training.
- `src/figs`: Scripts for generating manuscript visualizations.
## Credits
If you found the repository useful, please cite:
```
@article {Rissom2024.06.21.600139,
	author = {Rissom, Pia Francesca and Sarmiento, Paulo Yanez and Safer, Jordan and Coley, Connor W. and Renard, Bernhard Y. and Heyne, Henrike O. and Iqbal, Sumaiya},
	title = {Decoding protein language models: insights from embedding space analysis},
	elocation-id = {2024.06.21.600139},
	year = {2025},
	doi = {10.1101/2024.06.21.600139},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2025/02/11/2024.06.21.600139},
	eprint = {https://www.biorxiv.org/content/early/2025/02/11/2024.06.21.600139.full.pdf},
	journal = {bioRxiv}
}
```
Special thanks to [Pia Francesca Rissom](https://github.com/pia-francesca) for providing the Plotly visualization scripts and core Emma utilities.

## License
This repository is licensed under the [MIT license](https://github.com/skrhakv/emb-space-analysis/blob/master/LICENSE).
