import json
import os
import pickle

from dotenv import load_dotenv

curdir = os.path.abspath(os.path.dirname(__file__))
dotenv_path = os.path.join(curdir, ".env")
load_dotenv(dotenv_path)

# inference functions


def run_MS1Former(peaks_fp: str, output_fp: str):
    from Maple.Embedder.inference.MS1Pipeline import MS1Pipeline

    pipe = MS1Pipeline(gpu_id=0)
    ms1_peaks = json.load(open(peaks_fp))
    out = pipe.embed_ms1_spectra_from(ms1_peaks=ms1_peaks)
    pickle.dump(out, open(output_fp, "wb"))
