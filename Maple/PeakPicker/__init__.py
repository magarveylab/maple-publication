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
    response = formula_analysis.get_predictions()
    out = []
    for peak_id, formulas_raw in response.items():
        formulas = [{"formula": f[0], "score": f[1]} for f in formulas_raw]
        out.append(
            {
                "peak_id": peak_id,
                "formulas": sorted(
                    formulas, key=lambda x: x["score"], reverse=True
                ),
            }
        )
    json.dump(out, open(output_fp, "w"))
