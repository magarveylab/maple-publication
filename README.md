# maple-publication
Metabolome-genome Alignment and Predictive Learning Engine (MAPLE)

## Overview
MAPLE is an AI-driven framework designed to integrate LC-MS/MS metabolomic profiles with bacterial genomic data for targeted metabolite discovery. This repository contains the full implementation of MAPLE, including data preprocessing, inference pipelines and model training.

## System Requirements

### Hardware Requirements

This system was developed and tested on a high-performance server with the following configuration:

- Dual Intel Gold 5218 CPUs @ 2.30GHz

- 8× NVIDIA Quadro RTX 5000 GPUs (16 GiB VRAM each)

- 250 GiB DDR4 RAM

- Ubuntu Linux 20.04

However, the software can be run on any modern Linux system equipped with an NVIDIA GPU that supports CUDA 12 or higher. Performance will scale with available GPU memory and compute capacity.

### Software Requirements

- Linux (Ubuntu 20.04 or compatible)

- NVIDIA GPU with CUDA 12+

- Conda (recommended for environment management)


### Conda Installation

Different modules in this package require different Python versions. To ensure compatibility, we provide dedicated Conda environments for each MAPLE module.

#### Installation via Pip Symlinks:

1. Create and activate the appropriate Conda environment for each MAPLE module.

2. Then, install the package in editable mode using pip:

Conda environment for peak picking (e.g., adduct analysis, molecular formula prediction) and <sup>13</sup>C isotope feeding analysis 
```
    conda env create -f envs/MaplePeakPicker.yml
    conda activate MaplePeakPicker
    pip install -e .
```

Conda environment for in silico MS2 fragmentation
```
    conda env create -f envs/MapleFragmenter.yml
    conda activate MapleFragmenter
    pip install -e .
```

Conda environment for embedding MS<sup>1</sup>/MS<sup>2</sup> data
```
    conda env create -f envs/MapleDL.yml
    conda activate MapleDL
    pip install -e .
```

### Inference-Only Installation
1. Install the Package via Pip Symlinks:
    - Create and activate the Conda environments for different Maple modules, then install the package in editable mode

Conda environment for peak picking (e.g., adduct analysis, molecular formula prediction) and <sup>13</sup>C isotope feeding analysis 
```
    conda env create -f envs/MaplePeakPicker.yml
    conda activate MaplePeakPicker
    pip install -e .
```

Conda environment for in silico MS2 fragmentation
```
    conda env create -f envs/MapleFragmenter.yml
    conda activate MapleFragmenter
    pip install -e .
```

Conda environment for embedding MS<sup>1</sup>/MS<sup>2</sup> data
```
    conda env create -f envs/MapleDL.yml
    conda activate MapleDL
    pip install -e .
```

2. Download the necessary data from the accompanying [Zenodo repository](https://zenodo.org/doi/10.5281/zenodo.15285195). 

3. Set Up Qdrant
    - Install Qdrant and restore the Qdrant reference databases from the provided snapshots. Look under **Qdrant Setup** for more details.

## Supplementary Package Installation for Genomic and Molecular Analysis

The following packages were used to support various analysis including strain prioritization, computation of the polyketide molecular universe, and large-scale analysis of encoded metabolite landscapes. Outputs are integrated with metabolite-level analyses for comprehensive multi-omic interpretation.

| Package | Description | Publication Link
|---|---| ---|
| [IBIS](https://github.com/magarveylab/ibis-publication/tree/main)          | Integrated Biosynthetic Inference Suite (IBIS) - AI-based platform for high-throughput identification and comparison of bacterial metabolism from genomic data   | [Here](https://www.pnas.org/doi/10.1073/pnas.2425048122) | 
| [BLOOM](https://github.com/magarveylab/bloom-publication/tree/main)         | Biosynthetic Learning from Ontological Organizations of Metabolism (BLOOM) - Chemoinformatics platform for biosynthetic pathway inference from molecular structures via substructure matching. Utilizes AI-based embeddings for organizing metabolites within a biosynthetic ontology, and incorporates knowledge graph reasoning to associate BGCs with molecules. |  In Review | 


## Qdrant Setup
MAPLE inference piplelines utilize [Qdrant](https://qdrant.tech/) embedding databases for approximate nearest neighbor (ANN) lookups. We provide a hosted cloud service for vector similarity searches. However, in the event of downtime or for local deployment, Qdrant can be easily run in a Docker container by following the the [official quickstart guide](https://qdrant.tech/documentation/quickstart/).

### Restoring Qdrant Databases
To restore the Qdrant databases, download and extract `QdrantSnapshots.yml` and place the contents in [this directory](https://github.com/magarveylab/maple-publication/tree/main/Maple/Embedder/QdrantSnapshots). Since the MS1-Qdrant database is too large to store directly, you will need to recreate it from the raw embeddings `ms1_embeddings.zip`. Run the following command to do so. The script requires approximately 12 GB of memory and takes about 1 hour to complete. This step is only necessary for Qdrant-related functions: `annotate_mzXML_with_chemotypes` and  `annotate_mzXML_with_tax_scores`.
```
conda activate MapleDL
python restore_qdrant.py -ms1_embedding_dir ms1_embedding.zip
```

## Graphormer Training
Training scripts for both MS1Former and MS2Former are [provided](https://github.com/magarveylab/maple-graphormer-training/tree/main) to support model development, pretraining, and task-specific fine-tuning.

## Inference
Refer to the Jupyter notebooks below for example inference workflows that can be adapted to your own data:

1. Peak-Picking-Modules.ipynb
2. Insilico-Fragmentation-Modules.ipynb
3. MS-Embedding-Modules.ipynb


