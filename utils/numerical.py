import numpy as np

# average value for a DataFrame column, ignoring NaN's
def nan_average(df, column):
    not_nan_df = df[np.isnan(df[column]) == False]
    avg = sum(not_nan_df[column]) / len(not_nan_df)
    return avg