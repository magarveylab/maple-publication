import json
import logging
import math
import os
import re
import sys
import time
import urllib.parse
import urllib.request

import pandas as pd
from brainpy import isotopic_variants
from tqdm import tqdm

from Maple.PeakPicker.isotopepreds.e_ratio_preds import e_ratio_preds

# set logging
logging.basicConfig(
    format="%(asctime)s %(message)s", level=logging.INFO, stream=sys.stdout
)

# load formula adjustment database
curdir = os.path.abspath(os.path.dirname(__file__))
adducts_library_fp = os.path.join(curdir, "database/formula_adjustments.csv")
formula_adjustments = pd.read_csv(adducts_library_fp).to_dict("records")
adduct_dict = {adduct["adduct_type"]: adduct for adduct in formula_adjustments}

formula_parse = re.compile("([A-Z][a-z]?)(\d*)")


def batchify(l, bs=500):  # group calls for database
    return [l[x : x + bs] for x in range(0, len(l), bs)]


def calc_ppm(x, y):
    return (x - y) / y * 1000000


def mass2formulas(
    mass,
    mfRange="C5-100 H1-100 N0-50 O0-50 Cl0-3 Br0-3 S0-3",
    ppm_tol=5,
    delay: int = 1,
):
    chemcalcURL = "https://www.chemcalc.org/chemcalc/em"
    params = {"mfRange": mfRange, "monoisotopicMass": mass}
    response = urllib.request.urlopen(
        chemcalcURL, urllib.parse.urlencode(params).encode("utf-8")
    )
    formulas = [
        f["mf"]
        for f in json.loads(response.read())["results"]
        if abs(f["ppm"]) <= ppm_tol
    ]
    time.sleep(delay)
    return formulas


def formula2counts(formula):
    # return elemental counts for each formula
    element_possibilities = ["C", "H", "N", "O", "Cl", "Br", "S"]
    all_elements = {}
    for element_name, count in formula_parse.findall(formula):
        if count == "":
            count = 1
        else:
            count = int(count)
        all_elements[str(element_name)] = count
    # populate dictionary with missing elements
    for element_name in element_possibilities:
        if element_name not in all_elements:
            all_elements[element_name] = 0
    return all_elements


def formula2distribution(formula, adduct_type, npeaks, charge):
    # calculate elemental counts
    stor = formula2counts(formula)
    # adjust elemental count to correspond to adduct
    adduct = adduct_dict[adduct_type]
    stor["C"] = stor["C"] * adduct["M_multiply"] + adduct["C_add"]
    stor["H"] = (
        stor["H"] * adduct["M_multiply"]
        + adduct["H_add"]
        - adduct["H_subtract"]
    )
    stor["N"] = stor["N"] * adduct["M_multiply"] + adduct["N_add"]
    stor["O"] = stor["O"] * adduct["M_multiply"] + adduct["O_add"]
    stor["S"] = stor["S"] * adduct["M_multiply"] + adduct["S_add"]
    stor["Na"] = adduct["Na_add"]
    stor["K"] = adduct["K_add"]
    # calculate theoretical distribution using brainpy
    theoretical_isotopic_cluster = isotopic_variants(
        stor, npeaks=npeaks, charge=charge
    )
    # normalize intensities of isotopic distributions
    high_intens = max([p.intensity for p in theoretical_isotopic_cluster])
    stor["theor_intens"] = [
        round(p.intensity / high_intens, 3)
        for p in theoretical_isotopic_cluster
    ]
    stor["theor_mz"] = [p.mz for p in theoretical_isotopic_cluster]
    return stor


def compare_distributions(
    exp_data, theor_data, calibration_ppm=2, allowed_ppm=5
):
    # The following method to compare isotopic distributions is used by TraceFinder 4.1
    # https://assets.thermofisher.com/TFS-Assets/CMD/manuals/Man-XCALI-97834-TraceFinder-41-Lab-Director-Quan-ManXCALI97834-EN.pdf
    # initialize variables
    exp_intens = exp_data["iso_intens"]
    theor_intens = theor_data["theor_intens"]
    exp_mz = exp_data["iso_mz"]
    theor_mz = theor_data["theor_mz"]
    # calculate normalized intensity deviation (normalize by multiplication with 10)
    intens_deviation = [
        abs(x - y) * 10 for x, y in zip(exp_intens, theor_intens)
    ]
    # calculate normalized mass deviation (normalize by multiplication with 10)
    mass_deviation = []
    for x, y in zip(exp_mz, theor_mz):
        ppm = calc_ppm(x, y)
        if ppm < calibration_ppm:
            mass_deviation.append(0)
        else:
            mass_deviation.append(
                (ppm - calibration_ppm) / (allowed_ppm - calibration_ppm)
            )
    # calculate vector sum using pythagorean theorem
    vector_sum = []
    for m, i in zip(mass_deviation, intens_deviation):
        combined_deviation = math.sqrt(m**2 + i**2)
        if combined_deviation > 1:
            combined_deviation = 1
        vector_sum.append(combined_deviation)
    # calculate weight factor
    weight_factor = [intens / sum(exp_intens) for intens in exp_intens]
    # return isotopic pattern score
    return round(
        (1 - sum([v * w for v, w in zip(vector_sum, weight_factor)])) * 100, 2
    )


def formula_finder(peak, ppm_tol=5, ips=50, t=0.1):
    # peak properties
    peak_id = peak["peak_id"]
    mass = peak["mass"]
    charge = peak["charge"]
    adduct_type = peak["adduct_type"]
    iso_mz = peak["iso_mz"]
    iso_intens = peak["iso_intens"]
    iso_count = len(iso_mz)
    theor_nc = peak["nc"]
    theor_sc = peak["sc"]
    # return formulas fitted to mass (using ChemCalc), then filtered by set ppm tolerance
    inital_formulas = mass2formulas(mass, ppm_tol=ppm_tol)
    # further filter formulas using isotope pattern scores
    filtered_formulas = []
    for formula in inital_formulas:
        theor_data = formula2distribution(
            formula, adduct_type, iso_count, charge
        )
        # remove formulas with lower C count relative to N and O count
        if (
            theor_data["O"] >= theor_data["C"]
            or theor_data["N"] >= theor_data["C"]
        ):
            continue
        # remove formulas with no O or N
        if theor_data["O"] + theor_data["N"] == 0:
            continue
        # filter for Cl, Br, S based on intensity of third component
        if theor_data["Cl"] > 0 or theor_data["Br"] > 0 or theor_data["S"] > 0:
            if abs(theor_data["theor_intens"][2] - iso_intens[2]) > t:
                continue
        # filter for S and N using ratios:
        nc_ratio = theor_data["N"] / theor_data["C"]
        sc_ratio = theor_data["S"] / theor_data["C"]
        if nc_ratio < theor_nc["lower"] or nc_ratio > theor_nc["upper"]:
            continue
        if sc_ratio < theor_sc["lower"] or sc_ratio > theor_sc["upper"]:
            continue
        # calculate isotopic pattern score
        isotopic_pattern_score = compare_distributions(peak, theor_data)
        # remove any isotope patter score under 10
        if isotopic_pattern_score > 10:
            filtered_formulas.append(
                {"formula": formula, "score": isotopic_pattern_score}
            )
    # determine highest isotopic pattern score
    final_formulas = []
    if len(filtered_formulas) > 0:
        max_score = max([i["score"] for i in filtered_formulas])
        for formula in filtered_formulas:
            if max_score - formula["score"] <= ips:
                final_formulas.append([formula["formula"], formula["score"]])
    return {peak_id: final_formulas}


class FormulaAnalysis:

    def __init__(self, peaks, ppm_tol=5, ips=50, t=0.1, cores=10):
        """
        mandatory arguments:
            peaks is list of dictionaries with the following keys
                peak_id,
                mass,
                charge,
                adduct_type,
                iso_mz,
                iso_intens
        optional arguments:
            ppm_tol -> error tolerance in calculating initial formulas
            ips -> isotopic pattern score (default minimum is 50)
            t -> error tolerance in the difference of the third peak
        """
        self.peaks = peaks
        self.ppm_tol = ppm_tol
        self.ips = ips
        self.t = t
        self.cores = cores
        self.predictions = {}

    def predict_elemental_ratios(self):
        logging.info("Predict Elemental Ratios")
        peaks = []
        batches = batchify(self.peaks, bs=500)
        for result in tqdm(batches):
            peaks.extend(e_ratio_preds(result))
        self.peaks = peaks

    def predict_formulas(self):
        logging.info("Predict Formulas")
        for p in tqdm(self.peaks):
            r = formula_finder(p, ppm_tol=self.ppm_tol, ips=self.ips, t=self.t)
            self.predictions.update(r)
