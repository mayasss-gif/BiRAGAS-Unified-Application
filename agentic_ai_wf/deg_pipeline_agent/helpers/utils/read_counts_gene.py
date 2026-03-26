import pandas as pd

def read_counts_with_gene_column(file_path, sep='\t'):
    # Read the file without assuming headers
    df_raw = pd.read_csv(file_path, sep=sep, header=None)

    print(f"🔍 First row sample: {df_raw.iloc[0].tolist()}")

    # Heuristically detect if first row is actually the header (by checking if >1 string values)
    first_row = df_raw.iloc[0]
    num_strings = sum(isinstance(val, str) and not val.replace('.', '', 1).isdigit() for val in first_row)

    if num_strings >= 2:
        # Likely a header row: promote it to columns
        df = pd.read_csv(file_path, sep=sep)
        df.rename(columns={df.columns[0]: 'gene'}, inplace=True)
        print("✅ First row used as header, first column renamed to 'gene'")
    else:
        # No header present, assign manually
        df = df_raw
        df.columns = ['gene'] + [f'Sample_{i}' for i in range(1, df.shape[1])]
        print("✅ No header in file, assigned 'gene' and sample column names")

    return df