# maple-publication
MAPLE package for external use (released with publication)

## Overview
MAPLE is an AI-driven framework designed to integrate LC-MS/MS metabolomic profiles with bacterial genomic data for targeted metabolite discovery. This repository contains the full implementation of MAPLE, including data preprocessing, inference pipelines and model training.

## Installation

### Inference-Only Installation
1. Install the Package via Pip Symlinks:
    - Create and activate the Conda environments for different Maple modules, then install the package in editable mode \

Conda environment for peak picking (e.g., adduct analysis, molecular formula prediction) and <sup>13</sup>C isotope feeding analysis 
```
    conda env create -f envs/MaplePeakPicker.yml
    conda activate MaplePeakPicker
    pip install -e .
```

Conda environment for _in silico_ MS<sup>2</sup> fragmentation
```
    conda env create -f envs/MapleFragmenter.yml
    conda activate MapleFragmenter
    pip install -e .
```

Conda environment for embedding MS<sup>1</sup> and MS<sup>2</sup> data
```
    conda env create -f envs/MapleEmbedder.yml
    conda activate MapleEmbedder
    pip install -e .
```

2. Set Up Qdrant
    - Install Qdrant and restore the Qdrant reference databases from the provided snapshots. Look under **Qdrant Setup** for more details.

## Inference

### Preprocessing Raw mzXML Files

MAPLE provides high-level inference functions for streamlined LC–MS/MS data preprocessing. Raw WIFF files should be converted to the mzXML format using [ProteoWizard](https://hub.docker.com/r/proteowizard/pwiz-skyline-i-agree-to-the-vendor-licenses).

The preprocessing pipeline consists of the following steps:
1. Data Loading – mzXML files are parsed using [pyOpenMS](https://pyopenms.readthedocs.io/en/latest/)
2. EIC Decomposition – Spectral data is decomposed into extracted ion chromatograms (EICs)
3. Isotopic Pattern Analysis – MS<sup>1</sup> signals are grouped into isotopic distributions, and charge states are assigned.
4. Adduct Annotation - [Common adducts](https://github.com/magarveylab/maple-publication/blob/main/Maple/PeakPicker/database/adducts.csv) are identified to compute accrurate neutral monoisotopic mass.
5. Molecular Formula Prediction - Candidate molecular formulas within 5 ppm of the observed mass are generated using [ChemCalc](https://www.chemcalc.org/), limited to common natural product elements (C, H, O, N, S, Cl, Br). [Brainpy](https://github.com/mobiusklein/brainpy) is used to compute theoretical isotopic distributions, which are compared to experimental data to prioritize high-confidence candidates.
```python
import json

from Maple.PeakPicker.MSAnalysis import MSAnalysis

mzxml_fp = "sample_data/103350.mzXML"
analysis = MSAnalysis(mzxml_fp)
analysis.run_complete_process()

out = analysis.MASTERbook
json.dump(out, open("sample_output/103350.json", "w"))
```