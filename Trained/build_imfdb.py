# import the necessary packages
import pandas as pd
import numpy as np

EXTENDED_LABELS_PATH = "imfdb_dataset/data_desc.csv"

# read the age labels into memory
extended_labels = pd.read_csv(EXTENDED_LABELS_PATH, header = None)
age = extended_labels.iloc[:, 1]

age_labels = np.array(age)
age_labels = age_labels.reshape((age_labels.shape[0], 1))

np.save("dataset/imfdb_age_labels.npy", age_labels)