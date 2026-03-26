def infer_type_series(series):
    """
    Infers dominant data type of a pandas Series.
    Returns 'int', 'float', or 'str'.
    """
    sample = series.dropna().astype(str).head(10)
    inferred = []

    for val in sample:
        val = val.strip()
        if val.replace('.', '', 1).replace('-', '', 1).isdigit():
            if '.' in val:
                inferred.append('float')
            else:
                inferred.append('int')
        else:
            inferred.append('str')

    return max(set(inferred), key=inferred.count)