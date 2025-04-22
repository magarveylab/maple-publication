# maple-publication
MAPLE package for external use (released with publication)

## Overview
MAPLE is an AI-driven framework designed to integrate LC-MS/MS metabolomic profiles with bacterial genomic data for targeted metabolite discovery. This repository contains the full implementation of MAPLE, including data preprocessing, inference pipelines and model training.

## Installation

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
    conda env create -f envs/MapleEmbedder2.yml
    conda activate MapleEmbedder2
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
from Maple.PeakPicker import run_peak_picker

run_peak_picker(
    mzXML_fp="sample_data/20109_Chitinophaga__C408L_Czapek-Dox-1perstarch_HP20-XAD7bags_C12_1.mzXML", # input data
    output_fp="sample_output/20109_peaks.json"
)
```

Run the following code to predict molecular formulas independently, with faster inference achieved by prioritizing high-intensity signals.
```python
from Maple.PeakPicker import run_formula_predictor
import json

query_peaks = [
    {"peak_id": 31365361,
     "mass": 283.084,
     "charge": 1,
     "adduct_type": "2MpH",
     "iso_mz": [567.1758, 568.1783, 569.1757, 570.1852],
     "iso_intens": [1, 0.324, 0.112, 0.039]}
]

run_formula_predictor(
    peaks=query_peaks,
    output_fp="sample_output/example_formula_predictions.json",
    cpu=10 # the number of cpu cores to use
)
```

### Analyzing Isotope Feeding Studies

MAPLE includes inference functions for analyzing isotope feeding experiments by comparing control and labeled samples. It detects changes in isotopic distributions (measured using skewness) and identifies mass shifts in MS<sup>2</sup> fragmentation data. <sup>12</sup>C mzXML files must first be processed using the peak picking module.

```python
from Maple.FeedingAnalysis import run_feeding_analysis

run_feeding_analysis(
    c13_mzXML_fp="sample_data/20111_Chitinophaga__C408L_Czapek-Dox-1perstarch_HP20-XAD7bags_C13_1.mzXML", # input data
    c12_peaks_fp="sample_output/20109_peaks.json", # input data
    output_fp="sample_output/20111_feeding_results.json"
)
```

Run the following code to calculate the skewness of an isotopic distribution
```python
from Maple.FeedingAnalysis import get_isot_dist_skewness

iso_mz = [937.684, 938.687, 939.689, 940.694, 941.694]
iso_intens = [1, 0.558, 0.188, 0.046, 0.009]
skew = iso_dist_skewness(iso_mz=iso_mz, iso_intens=iso_intens)

```

Run the following code to assess significant labeled isotope incorporation, using the Dixon Q test to determine whether <sup>13</sup>C skewness is a statistical outlier.

```python
from Maple.FeedingAnalysis import does_peak_shift

c12_skew_values = [0.3098, 0.2913, 0.273, 0.2825, 0.2695, 0.2822, 0.2668]
c13_skew_value = 0.1537
result = does_peak_shift(c12_skew_values, c13_skew_value, alpha=0.01)
```

### _In Silico_ MS<sup>2</sup> Fragmentation Prediction
MAPLE computes theoretical fragmentation trees using a curated set of chemical reactions derived from [literature](https://pubs.rsc.org/en/content/articlelanding/2016/np/c5np00073d). The current implementation supports positive ion mode only.
```python
from Maple.Fragmenter import run_ms2_fragmenter

run_ms2_fragmenter(
    smiles="CNC[C@H](O)C1=CC=C(O)C(O)=C1", # input data
    output_fp="sample_output/example_insilico_fragmentation.json"
)
```

### Embedding MS<sup>1</sup> Signals
MAPLE generates embeddings that capture contextual relationships between co-eluting metabolites, enabling the construction of MS<sup>1</sup>-level similarity networks. These networks can be used to assess metabolomic uniqueness across taxa and prioritize lineage-specific chemical signatures.

Run the following code to compute a global spectral embedding and individual MS<sup>1</sup> embeddings for a given mzXML file. Note: mzXML files must first be processed using the peak picking module.

```python
from Maple.Embedder import run_MS1Former

run_MS1Former_on_mzXML(
    peaks_fp="sample_output/20109_peaks.json", # input data
    output_fp="sample_output/20109_MS1Former_embeddings.pkl",
    gpu_id=0
)
```

After generating MS<sup>1</sup> embeddings, run the following command to compute taxonomy consistency scores. Query Taxonomic labels must correspond to the naming conventions provided in the following [reference tables](https://github.com/magarveylab/maple-publication/tree/main/Maple/Embedder/dat/taxonomy_tables). If a score returns None, it indicates that no overlapping signals were detected in the current LC–MS/MS database, and therefore a reliable score could not be computed. 

```python
from Maple.Embedder import annotate_mzXML_with_tax_scores

annotate_mzxml_with_tax_scores(
    peaks_fp="sample_output/20109_peaks.json", # input data
    ms1_emb_fp="sample_output/20109_MS1Former_embeddings.pkl", # input data
    output_fp="sample_output/20109_MS1Former_taxscores.csv",
    query_phylum="bacteroidetes",
    query_class="sphingobacteriia",
    query_order="sphingobacteriales",
    query_family="chitinophagaceae",
    query_genus="chitinophaga",
)
```

### Embedding MS<sup>2</sup> Signals

MAPLE generates two MS<sup>2</sup> embeddings: the **chemotype embedding** for ANN-based biosynthetic class prediction, and the **analog embedding** for clustering structural derivatives within biosynthetic classes.

```python
from Maple.Embedder import run_MS2Former_on_mzXML

run_MS2Former_on_mzXML(
    peaks_fp="sample_output/20109_peaks.json", # input data
    output_fp="sample_output/20109_MS2Former_chemotype_embeddings.pkl",
    embedding_type="chemotype",
    gpu_id=0,
    min_ms2=5,
)

run_MS2Former_on_mzXML(
    peaks_fp="sample_output/20109_peaks.json", # input data
    output_fp="sample_output/20109_MS2Former_analog_embeddings.pkl",
    embedding_type="analog",
    gpu_id=0,
    min_ms2=5,
)
```
Run the following code to perform ANN-based biosynthetic class prediction.
```python
from Maple.Embedder import annotate_mzXML_with_chemotypes

# Directly from peak data (includes embedding generation)
annotate_mzXML_with_chemotypes(
    peaks_fp="sample_output/20109_peaks.json", # input data
    output_fp="sample_output/20109_MS2Former_chemotype_predictions.csv",
)

# From precomputed embeddings
annotate_mzXML_with_chemotypes(
    ms2_emb_fp="sample_output/20109_MS2Former_chemotype_embeddings.pkl", # input data
    output_fp="sample_output/20109_MS2Former_chemotype_predictions.csv",
)
```
Run the following command to perform density-based MS<sup>2</sup> embedding clustering across multiple mzXML files. The method supports comparison of millions of peaks simultaneously. For optimal performance, we recommend tuning the clustering parameters (`min_cluster_size` and `n_neighbors`). Default parameters used in the study are provided.
```python

from Maple.Embedder import compute_ms2_networks_from_mzXMLs

out = compute_ms2_networks_from_mzXMLs(
    ms2_emb_fps=[
        'sample_output/20109_MS2Former_analog_embeddings.pkl'
    ],
    output_fp="example_MS2Former_analog_predictions.csv",
    n_neighbors=15,
    min_cluster_size=5,
)

```