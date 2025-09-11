def range_analysis(df, min=0, max=-1):
    """calculates the following stats for the dataframe: number of rows with value within the input range, percentage of rows within that range, sum of values of the rows within that range, percentage of that sum in relation to the whole column

    Args:
        df (pandas.DataFrame): dataframe containing an address and an integer representing the value to be ranged
        min (int, optional): lower bound of range. Defaults to 0.
        max (int, optional): upper bound of range. Defaults to -1: no upper bound.

    Returns:
        _list_: list of stats regarding the dataframe in input
    """

    if min == max:
        mints = f"n = {max}"
    else:
        mints = f"{min} <= n <= {max}"
        
    if max == -1:
        mints = f"{min} <= n"
        minters_in_range_df = df[df.iloc[:,1] >= min]
    else:
        minters_in_range_df = df[(df.iloc[:,1] >= min) & (df.iloc[:,1] <= max)]

    num_minters = len(minters_in_range_df)
    
    total_minters = len(df)
    percentage_minters = "{}%".format(round((num_minters / total_minters) * 100, 2)) 

    num_mints = minters_in_range_df.iloc[:,1].sum()

    total_mints = df.iloc[:,1].sum()
    percentage_mints = "{}%".format(round((num_mints / total_mints) * 100, 2))

    return [mints, num_minters, percentage_minters, num_mints, percentage_mints]