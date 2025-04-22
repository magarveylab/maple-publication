from multiprocessing import Pool

import networkx as nx
import numpy as np
from pyopenms import *
from pyopenms import MSExperiment, MzXMLFile
from tqdm import tqdm

from Maple.PeakPicker.AdductAnalysis import AdductAnalysis
from Maple.PeakPicker.eicAnalysis import eicAnalysis
from Maple.PeakPicker.FormulaAnalysis import FormulaAnalysis
from Maple.PeakPicker.IsotopeAnalysis import IsotopeAnalysis
from Maple.PeakPicker.Logger import logging
from Maple.PeakPicker.utils import (
    overlap_masses,
    ppm_lower_end,
    ppm_upper_end,
    quick_precursor_search,
)


class MSAnalysis:

    ########################################################################
    # Initialization
    ########################################################################

    def __init__(self, mzXML_fp: str):
        # load arguments
        self.mzXML_fp = mzXML_fp
        # set up documents
        self.signal_dict = {}
        self.scan_trace = {}
        self.MS2book = {}
        self.MASTERbook = []

    ########################################################################
    # Pipeline
    ########################################################################

    def run_complete_process(
        self,
        cores=10,
        isotope_abs_tol=0.1,
        isotope_min_charge=1,
        isotope_max_charge=3,
        isotope_min_isopeaks=2,
        isotope_hal_sim_cutoff=0.90,
        isotope_allow_bromine=False,
        noise_intensity_cutoff=1000,
        eic_ppm_tol=10,
        eic_scan_count_cutoff=5,
        ms2_ppm_tol=10,
        ms2_rt_tol=10,
        ms2_limit=50,
        adduct_rt_tol=10,
        adduct_ppm_tol=10,
        adduct_atr=True,
        adduct_awr=0.8,
        adduct_aip=True,
        adduct_aic=3000,
        adduct_cc=0.6,
        single_scan_adduct_restrict=False,
        predict_formulas=True,
        formula_ppm_tol=5,
        formula_ips=50,
        formula_t=0.1,
        final_intensity_cutoff=10000,
        final_rt_cutoff=150,
    ):
        # complete extraction procedure
        logging.info("Read mzXML data")
        self.read_data()
        logging.info("Parse mzXML data")
        self.parse_data()
        logging.info("Decompose spectral data into EICs")
        self.eic_deconvolution(
            eic_ppm_tol=eic_ppm_tol,
            eic_scan_count_cutoff=eic_scan_count_cutoff,
        )
        logging.info("Determine isotopic distributions")
        self.isotope_deconvolution(
            noise_intensity_cutoff=noise_intensity_cutoff,
            isotope_abs_tol=isotope_abs_tol,
            isotope_min_charge=isotope_min_charge,
            isotope_max_charge=isotope_max_charge,
            isotope_min_isopeaks=isotope_min_isopeaks,
            isotope_hal_sim_cutoff=isotope_hal_sim_cutoff,
            isotope_allow_bromine=isotope_allow_bromine,
            cores=cores,
        )
        logging.info("Assign ms2 data")
        self.assign_ms2(
            ms2_ppm_tol=ms2_ppm_tol, ms2_rt_tol=ms2_rt_tol, ms2_limit=ms2_limit
        )
        logging.info("Detect Adducts")
        self.adduct_deconvolution(
            adduct_rt_tol=adduct_rt_tol,
            adduct_ppm_tol=adduct_ppm_tol,
            adduct_atr=adduct_atr,
            adduct_awr=adduct_awr,
            adduct_aip=adduct_aip,
            adduct_aic=adduct_aic,
            adduct_cc=adduct_cc,
            single_scan_adduct_restrict=single_scan_adduct_restrict,
            final_intensity_cutoff=final_intensity_cutoff,
            final_rt_cutoff=final_rt_cutoff,
            cores=cores,
        )
        if predict_formulas:
            logging.info("Predict Formulas for Base Peaks")
            self.formula_prediction(
                cores=cores,
                formula_ppm_tol=formula_ppm_tol,
                formula_ips=formula_ips,
                formula_t=formula_t,
            )
        logging.info("Cleaning MASTERbook values for JSON output")
        self.clean_values()
        logging.info("Finished")

    ########################################################################
    # Read data from mzXML with pyopenMS
    ########################################################################

    def read_data(self):
        # load data from mzXML
        self.experiment = MSExperiment()
        MzXMLFile().load(self.mzXML_fp, self.experiment)

    def parse_data(self):
        self.num_scans = self.experiment.size()
        signal_id = 1
        for scan_id, scan in tqdm(
            enumerate(self.experiment), total=self.num_scans
        ):
            # skip if noting is in the scan
            if scan.size() == 0:
                continue
            # determine MS1 vs MS2
            ms_level = scan.getMSLevel()
            # determine retention time
            rt = scan.getRT()
            # cache MS1 scans
            if ms_level == 1:
                signals = []
                for mz, i in zip(*scan.get_peaks()):
                    signals.append([signal_id, mz, i])
                    self.signal_dict[signal_id] = {
                        "scan_id": scan_id,
                        "mz": mz,
                        "intensity": i,
                        "rt": rt,
                    }
                    signal_id += 1
                self.scan_trace[scan_id] = np.array(signals)
            # cache MS2 scans
            if ms_level == 2:
                scan_peaks = [
                    {"mz": mz, "intensity": i}
                    for mz, i in zip(*scan.get_peaks())
                ]
                # len of precursor is always 1 for this analysis
                mz = scan.getPrecursors()[0].getMZ()
                key = (scan_id, mz, rt)
                self.MS2book[key] = scan_peaks

    ########################################################################
    # Combine signals into EIC to select representatives
    ########################################################################

    def eic_deconvolution(
        self, cores=10, eic_ppm_tol=10, eic_scan_count_cutoff=5
    ):
        self.scan_trace = self.peak_dereplication(
            self.scan_trace,
            cores=cores,
            eic_ppm_tol=eic_ppm_tol,
            eic_scan_count_cutoff=eic_scan_count_cutoff,
            get_new_scan_trace=True,
            apply_intensity_threshold=True,
            allow_singletons=False,
        )

    def peak_dereplication(
        self,
        scan_trace,
        cores=10,
        eic_ppm_tol=10,
        eic_scan_count_cutoff=5,
        get_new_scan_trace=True,
        apply_intensity_threshold=True,
        allow_singletons=False,
        min_singelton_intensity=10000,
    ):
        # note this function is used to dereplicate peaks before
        # and after isotopic deconvolution
        # initial count of signals
        initial_count = 0
        for scan_id, signals in scan_trace.items():
            initial_count += len(signals)
        # set up process
        logging.info("Overlapping signals and organizing into network")
        eic_analysis = eicAnalysis(
            scan_trace,
            eic_scan_count_cutoff=eic_scan_count_cutoff,
            eic_ppm_tol=eic_ppm_tol,
        )
        scans_to_run = eic_analysis.scans_organized
        pool = Pool(cores)
        process = pool.imap_unordered(
            eic_analysis.process_scan, scans_to_run, chunksize=100
        )
        # construct network where nodes are peaks
        # and edges connect peaks belonging to same EIC
        eic_graph = nx.Graph()
        for result in tqdm(process, total=len(scans_to_run)):
            eic_graph.add_edges_from(result)
        pool.close()
        if allow_singletons:
            for scan_id, signals in scan_trace.items():
                eic_graph.add_nodes_from(
                    [n[0] for n in signals if n[2] >= min_singelton_intensity]
                )
        eic_analysis.signal_dict = self.signal_dict
        eic_analysis.eic_graph = eic_graph
        # find all connected regions in the graph (these represent different EIC)
        # chose most intense peak for each EIC
        logging.info("Find repersentative signals")
        new_scan_trace = {}
        signal_ids_to_keep = set()
        components = list(nx.connected_components(eic_graph))
        for nodes in tqdm(components):
            scan_condition, scan_count = eic_analysis.get_scan_count(nodes)
            if scan_condition or allow_singletons:
                signal_ids = eic_analysis.get_top_eic_signals(
                    nodes, apply_threshold=apply_intensity_threshold
                )
                signal_ids_to_keep.update(signal_ids)
                if get_new_scan_trace:
                    for signal_id in signal_ids:
                        self.signal_dict[signal_id]["scan_count"] = scan_count
                        scan_id = self.signal_dict[signal_id]["scan_id"]
                        mz = self.signal_dict[signal_id]["mz"]
                        intensity = self.signal_dict[signal_id]["intensity"]
                        if scan_id not in new_scan_trace:
                            new_scan_trace[scan_id] = []
                        new_scan_trace[scan_id].append(
                            [signal_id, mz, intensity]
                        )
        logging.info("Pre: {} signals".format(initial_count))
        logging.info("Post: {} signals".format(len(signal_ids_to_keep)))
        if get_new_scan_trace:
            # cast array - scan trace used for isotope deconvolution
            for scan_id, scan_signals in new_scan_trace.items():
                new_scan_trace[scan_id] = np.array(scan_signals)
            return new_scan_trace
        else:
            return signal_ids_to_keep

    ########################################################################
    # Combine isotopes
    ########################################################################

    def isotope_deconvolution(
        self,
        cores=10,
        isotope_abs_tol=0.1,
        isotope_min_charge=1,
        isotope_max_charge=3,
        isotope_min_isopeaks=2,
        isotope_hal_sim_cutoff=0.7,
        noise_intensity_cutoff=1000,
        isotope_allow_bromine=False,
    ):
        # decelare isotope process
        isotope_analysis = IsotopeAnalysis(
            abs_tol=isotope_abs_tol,
            min_charge=isotope_min_charge,
            max_charge=isotope_max_charge,
            min_isopeaks=isotope_min_isopeaks,
            hal_sim_cutoff=isotope_hal_sim_cutoff,
            noise_intensity_cutoff=noise_intensity_cutoff,
            allow_bromine=isotope_allow_bromine,
        )
        # find all the possible isotope distributions
        logging.info("Find all possible isotopic distributions")
        isotope_nets = []
        pool = Pool(cores)
        process = pool.imap_unordered(
            isotope_analysis.build_networks,
            self.scan_trace.items(),
            chunksize=100,
        )
        for result in tqdm(process, total=len(self.scan_trace)):
            isotope_nets.extend(result)
        pool.close()
        # prioritization and selection of likely isotope
        derep_isotopes = []
        logging.info("Priortization of isotopic distributions")
        pool = Pool(cores)
        process = pool.imap_unordered(
            isotope_analysis.dereplicate_isotopes, isotope_nets, chunksize=100
        )
        for result in tqdm(process, total=len(isotope_nets)):
            derep_isotopes.extend(result)
        pool.close()
        # organize results into MS1book
        scan_trace = {}
        for signal in derep_isotopes:
            # update peak with charge and isotopic distribution
            signal_id = signal["peak_id"]
            peak = self.signal_dict[signal_id]
            peak.update(signal)
            # signal row
            scan_id = peak["scan_id"]
            mz = peak["mz"]
            intensity = peak["intensity"]
            signal_row = [signal_id, mz, intensity]
            if scan_id not in scan_trace:
                scan_trace[scan_id] = []
            scan_trace[scan_id].append(signal_row)
        # cast as numpy arrays for peak dereplication
        for scan_id in scan_trace:
            scan_trace[scan_id] = np.array(scan_trace[scan_id])
        # dereplicate signals
        signals_to_keep = self.peak_dereplication(
            scan_trace,
            cores=cores,
            eic_ppm_tol=10,
            eic_scan_count_cutoff=5,
            get_new_scan_trace=False,
            apply_intensity_threshold=False,
            allow_singletons=True,
        )
        for signal_id in signals_to_keep:
            self.MASTERbook.append(self.signal_dict[signal_id])

    ########################################################################
    # Add fragmentation data
    ########################################################################

    def assign_ms2(self, ms2_ppm_tol=10, ms2_rt_tol=10, ms2_limit=50):
        # cast ms2 keys as array for fast search
        ms2_lookup_array = np.array(list(self.MS2book.keys()))
        for base in tqdm(self.MASTERbook):
            # filter precursors
            max_rt = base["rt"] + ms2_rt_tol
            min_rt = base["rt"] - ms2_rt_tol
            max_mz = ppm_upper_end(base["mz"], ms2_ppm_tol)
            min_mz = ppm_lower_end(base["mz"], ms2_ppm_tol)
            precursors = quick_precursor_search(
                ms2_lookup_array, max_mz, min_mz, max_rt, min_rt
            )
            # find closest precursor
            if precursors.size != 0:
                index = np.abs(ms2_lookup_array[:, 1] - base["mz"]).argmin()
                ms2 = self.MS2book[tuple(ms2_lookup_array[index])]
            else:
                ms2 = []
            del precursors
            # sort ms2 ions by intensity and capture top n ions
            if len(ms2) > 0:
                # intensity dict
                intensity_dict = {i["mz"]: i["intensity"] for i in ms2}
                mass_groups = overlap_masses(list(intensity_dict.keys()))
                filtered_ms2 = []
                for group in mass_groups:
                    top_ion = max(group, key=lambda m: intensity_dict[m])
                    filtered_ms2.append(
                        {"mz": top_ion, "intensity": intensity_dict[top_ion]}
                    )
                ms2 = sorted(
                    filtered_ms2, key=lambda i: i["intensity"], reverse=True
                )
                ms2 = ms2[:ms2_limit]
            # cache ms2
            base["ms2"] = ms2

    ########################################################################
    # Deduce adducts and true mass
    ########################################################################

    def adduct_deconvolution(
        self,
        adduct_rt_tol=10,
        adduct_ppm_tol=10,
        adduct_atr=True,
        adduct_awr=0.8,
        adduct_aip=True,
        adduct_aic=3000,
        adduct_cc=0.6,
        single_scan_adduct_restrict=False,
        final_intensity_cutoff=10000,
        final_rt_cutoff=150,
        cores=10,
    ):
        # calculate adducts
        adduct_analysis = AdductAnalysis(
            self.MASTERbook,
            rt_tol=adduct_rt_tol,
            ppm_tol=adduct_ppm_tol,
            atr=adduct_atr,
            awr=adduct_awr,
            aip=adduct_aip,
            aic=adduct_aic,
            cc=adduct_cc,
            cores=cores,
        )
        adduct_analysis.adduct_networking()
        self.MASTERbook = adduct_analysis.MASTERbook
        # filter peaks
        filtered_MASTERbook = []
        for peak in self.MASTERbook:
            # remove peaks under 10000 intensity and lacks adduct
            # and peaks that appear less than 150 seconds
            if peak["intensity"] < final_intensity_cutoff:
                if peak["adduct_cluster_id"] == None:
                    continue
                if peak["rt"] <= final_rt_cutoff:
                    continue
            if single_scan_adduct_restrict:
                if (
                    peak["scan_count"] == 1
                    and peak["adduct_cluster_id"] == None
                ):
                    continue
            filtered_MASTERbook.append(peak)
        self.MASTERbook = filtered_MASTERbook

    ########################################################################
    # Molecular formula prediction
    ########################################################################

    def formula_prediction(
        self, cores=10, formula_ppm_tol=5, formula_ips=50, formula_t=0.1
    ):
        # format data for formula analysis
        input_data = []
        for peak in self.MASTERbook:
            peak["formulas"] = None
            if peak["base_adduct"] and len(peak["isotopic_dist_mz"]) >= 2:
                input_data.append(
                    {
                        "peak_id": peak["peak_id"],
                        "mass": peak["monoisotopic_mass"],
                        "charge": peak["charge"],
                        "adduct_type": peak["adduct_type"],
                        "iso_mz": list(peak["isotopic_dist_mz"]),
                        "iso_intens": list(peak["isotopic_dist_intensity"]),
                    }
                )
        # prepare formula analysis
        formula_analysis = FormulaAnalysis(
            input_data,
            ppm_tol=formula_ppm_tol,
            ips=formula_ips,
            t=formula_t,
            cores=cores,
        )
        formula_analysis.predict_elemental_ratios()
        formula_analysis.predict_formulas()
        # add formula predictions to peaks
        predictions = formula_analysis.predictions
        for peak in self.MASTERbook:
            f = predictions.get(peak["peak_id"])
            if f != None:
                peak["formulas"] = sorted(f, key=lambda k: k[1], reverse=True)
            else:
                peak["formulas"] = None

    ########################################################################
    # Format Data for clean export
    ########################################################################

    def clean_values(self):
        new_MASTERbook = []
        for peak in self.MASTERbook:
            # reformat variables
            peak["scan_id"] = int(peak["scan_id"])
            peak["mz"] = round(float(peak["mz"]), 3)
            peak["rt"] = round(float(peak["rt"]), 1)
            peak["charge"] = int(peak["charge"])
            peak["intensity"] = int(peak["intensity"])
            peak["monoisotopic_mass"] = round(
                float(peak["monoisotopic_mass"]), 3
            )
            for ion in peak["ms2"]:
                ion["mz"] = round(float(ion["mz"]), 3)
                ion["intensity"] = round(float(ion["intensity"]), 3)
            peak["isotopic_dist_intensity"] = [
                round(float(i), 3) for i in peak["isotopic_dist_intensity"]
            ]
            peak["isotopic_dist_mz"] = [
                round(float(i), 3) for i in peak["isotopic_dist_mz"]
            ]
            peak["isotopic_distribution"] = list(
                zip(peak["isotopic_dist_intensity"], peak["isotopic_dist_mz"])
            )
            new_MASTERbook.append(peak)
            del peak["isotopic_dist_intensity"]
            del peak["isotopic_dist_mz"]
        # replace book
        self.MASTERbook = new_MASTERbook
        del new_MASTERbook
