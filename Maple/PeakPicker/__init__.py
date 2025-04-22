import json


def run_peak_picker(mzXML_fp: str, output_fp: str):
    from Maple.PeakPicker.MSAnalysis import MSAnalysis

    # Wrapper function to run peak picking process
    analysis = MSAnalysis(mzXML_fp=mzXML_fp)
    analysis.run_complete_process(predict_formulas=False)
    out = analysis.MASTERbook
    json.dump(out, open(output_fp, "w"))


def run_formula_predictor(peaks: list, output_fp: str, cpu: int = 10):
    from Maple.PeakPicker.FormulaAnalysis import FormulaAnalysis

    formula_analysis = FormulaAnalysis(peaks, cores=cpu)
    out = formula_analysis.get_predictions()
    json.dump(out, open(output_fp, "w"))
