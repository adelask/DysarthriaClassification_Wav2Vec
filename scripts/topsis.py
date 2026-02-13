import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean


def topsis(table):
    values = table.iloc[:, 2:].values

    # normalisation between 0 and 1
    norm_values = values / np.sqrt((values ** 2).sum(axis=0))

    # equal weights to all metrics
    weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

    # weighted values
    weighted_values = norm_values * weights

    # compute PIS a NIS
    PIS = weighted_values.max(axis=0)
    NIS = weighted_values.min(axis=0)

    # calculate Euclidean distance
    dist_pos = np.array([euclidean(row, PIS) for row in weighted_values])
    dist_neg = np.array([euclidean(row, NIS) for row in weighted_values])

    # relative closeness
    relative_closeness = dist_neg / (dist_pos + dist_neg)

    table2 = pd.DataFrame({
        "Extraction method": table["Metoda extrakce"],
        "TOPSIS": relative_closeness,
    })

    table2["rank"] = table2["TOPSIS"].rank(ascending=False, method="min").astype(int)
    print(table2)
    print("\n\n")


df = pd.read_csv('mean_performance_metrics.csv', sep=";")
#mean metrics

for dataset_value, group in df.groupby("dataset"):
    print(f"dataset: {dataset_value}")
    topsis(group)