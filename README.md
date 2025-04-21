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
from Maple.PeakPicker.MSAnalysis import MSAnalysis
import json

mzxml_fp = "sample_data/20109_Chitinophaga__C408L_Czapek-Dox-1perstarch_HP20-XAD7bags_C12_1.mzXML"
analysis = MSAnalysis(mzxml_fp)
# disable predict_formulas (for faster preprocessing)
analysis.run_complete_process(predict_formulas=False)

out = analysis.MASTERbook
json.dump(out, open("sample_output/20109_peaks.json", "w"))
```

Run the following code to predict molecular formulas independently.
```python
from Maple.PeakPicker.FormulaAnalysis import FormulaAnalysis
import json

query_peaks = [
    {'peak_id': 31365361,
     'mass': 283.084,
     'charge': 1,
     'adduct_type': '2MpH',
     'iso_mz': [567.1758, 568.1783, 569.1757, 570.1852],
     'iso_intens': [1, 0.324, 0.112, 0.039]}
]

formula_analysis = FormulaAnalysis(query_peaks, cores=10)
out = formula_analysis.get_predictions()
json.dump(out, open("sample_output/example_formula_predictions.json", "w"))
```

### Processing Isotope Feeding Studies

MAPLE includes inference functions for analyzing isotope feeding experiments by comparing control and labeled samples. It detects changes in isotopic distributions (measured using skewness) and identifies mass shifts in MS<sup>2</sup> fragmentation data. C<sup>12</sup> mzXML files must first be processed using the peak picking module.

```python
from Maple.FeedingAnalysis.FeedingAnalysis import feeding_analysis
import json

c12_peaks_fp = "sample_output/20109_peaks.json
c13_mzxml_fp = "sample_data/20111_Chitinophaga__C408L_Czapek-Dox-1perstarch_HP20-XAD7bags_C13_1.mzXML"

out = feeding_analysis(
    mzXML_fp=c13_mzxml_fp,
    query_peaks=json.load(open(peaks_fp)),
)

json.dump(out, open("sample_output/20111_feeding_results.json", "w"))
```

Run the following code to calculate the skewness of an isotopic distribution
```python
from Maple.FeedingAnalysis.Spectra import iso_dist_skewness

# format -> [relative_intensity, monoisotopic_mass]
isotopic_distribution =[
    [1, 937.684],
    [0.558, 938.687],
    [0.188, 939.689],
    [0.046, 940.694],
    [0.009, 941.694]
]
skew = iso_dist_skewness(isotopic_distribution)

```

Run the following code to assess significant labeled isotope incorporation, using the Dixon Q test to determine whether C<sup>13</sup> skewness is a statistical outlier.

```python
from Maple.FeedingAnalysis.Stats import does_peak_shift

c12_skew_values = [0.3098, 0.2913, 0.273, 0.2825, 0.2695, 0.2822, 0.2668]
c13_skew_value = 0.1537

does_peak_shift(c12_skew_values, c13_skew_value, alpha=0.01)
```