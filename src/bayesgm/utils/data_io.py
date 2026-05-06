import numpy as np


def save_data(fname, data, delimiter="\t"):
    """Save an array as .npy, .txt, or .csv."""
    if fname.endswith(".npy"):
        np.save(fname, data)
    elif fname.endswith(".txt") or fname.endswith(".csv"):
        np.savetxt(fname, data, fmt="%.6f", delimiter=delimiter)
    else:
        raise ValueError("Wrong saving format, please specify .npy, .txt, or .csv")
