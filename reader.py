import csv

import pandas as pd
import numpy as np
import sys

def read_annotated_file(path, index="index"):
    indices = []
    originals = []
    translations = []
    z_means = []
    score_stds = []
    with open(path, mode="r", encoding="utf-8-sig") as csvfile:
        reader = csv.DictReader(csvfile, delimiter="\t", quoting=csv.QUOTE_NONE)
        for row in reader:
            indices.append(row[index])
            originals.append(row["original"])
            translations.append(row["translation"])
            z_means.append(float(row["z_mean"]))
            temp=row["scores"].strip('][').split(', ')
            score_stds.append(float(round(np.std([float(i) for i in temp]),6)))

    return pd.DataFrame(
        {'index': indices,
         'original': originals,
         'translation': translations,
         'z_mean': z_means,
         'score_stds':score_stds
         })


def read_test_file(path, index="index"):
    indices = []
    originals = []
    translations = []
    with open(path, mode="r", encoding="utf-8-sig") as csvfile:
        reader = csv.DictReader(csvfile, delimiter="\t", quoting=csv.QUOTE_NONE)
        for row in reader:
            indices.append(row[index])
            originals.append(row["original"])
            translations.append(row["translation"])

    return pd.DataFrame(
        {'index': indices,
         'original': originals,
         'translation': translations,
         })
