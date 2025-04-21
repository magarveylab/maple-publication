import os

import pandas as pd


class InverseDict:

    def __init__(self):
        self.forward = {}
        self.reverse = {}

    def add(self, x, y):
        self.forward[x] = y
        self.reverse[y] = x

    def delete(self, dict_type, x):
        if dict_type == "forward":
            if x in self.forward:
                to_remove = self.forward.pop(x)
                del self.reverse[to_remove]
        elif dict_type == "reverse":
            if x in self.reverse:
                to_remove = self.reverse.pop(x)
                del self.forward[to_remove]


# define location path of package
curdir = os.path.abspath(os.path.dirname(__file__))

# load reactions
reactions_fp = os.path.join(curdir, "data", "reactions.csv")
reactions_df = pd.read_csv(reactions_fp)

# load extra smarts
terms_fp = os.path.join(curdir, "data", "extra_smarts_terms.csv")
terms = {
    r["term"]: r["smarts"] for r in pd.read_csv(terms_fp).to_dict("records")
}

# load electronegativity chart
en_chart = {}
en_fp = os.path.join(curdir, "data", "electronegativity.csv")
for row in pd.read_csv(en_fp).to_dict("records"):
    en_chart[row["atomic_number"]] = row["pauling_electronegativity"]
