import pandas as pd
from typing import Dict, Union

def detect_header_data_mismatch(
    file_path: str,
    separator: str = '\t',
    encoding: str = 'utf-8',
    sample_rows: int = 10
) -> Dict[str, Union[Dict, pd.DataFrame]]:
    """
    Detect header/data column mismatch and return corrected DataFrame.
    
    Returns:
        {
            'has_mismatch': bool,
            'header_columns': int,
            'data_columns': int,
            'difference': int,
            'sample_data_column_counts': list,
            'recommended_action': str,
            'details': str,
            'missing_header_columns': dict,
            'df': pd.DataFrame (corrected DataFrame)
        }
    """
    result = {
        'has_mismatch': False,
        'header_columns': 0,
        'data_columns': 0,
        'difference': 0,
        'sample_data_column_counts': [],
        'recommended_action': '',
        'details': '',
        'missing_header_columns': {},
        'df': None
    }

    try:
        # Read raw lines
        with open(file_path, 'r', encoding=encoding) as f:
            lines = [f.readline() for _ in range(20)]
        
        if len(lines) < 2:
            result['details'] = "File has less than 2 lines"
            return result

        # Detect header line
        for i, line in enumerate(lines):
            values = line.strip().split(separator)
            numeric_count = sum(1 for v in values if v.replace('.', '', 1).replace('-', '', 1).isdigit())
            if numeric_count >= 2 and i > 0:
                header_line_index = i - 1
                header_line = lines[header_line_index].strip()
                break
        else:
            header_line_index = 0
            header_line = lines[0].strip()

        header_values = header_line.split(separator)
        header_columns = len(header_values)
        result['header_columns'] = header_columns

        # Collect data lines
        data_sample_rows = lines[header_line_index + 1:header_line_index + 1 + sample_rows]
        if not data_sample_rows:
            result['details'] = "No data rows found"
            return result

        # Count data columns
        data_column_counts = [len(row.strip().split(separator)) for row in data_sample_rows]
        most_common_data_columns = max(set(data_column_counts), key=data_column_counts.count)
        result['data_columns'] = most_common_data_columns
        result['sample_data_column_counts'] = data_column_counts

        # Check for mismatch
        if header_columns != most_common_data_columns:
            result['has_mismatch'] = True
            result['difference'] = most_common_data_columns - header_columns

            # Fill missing header columns
            fixed_header_values = header_values.copy()
            for idx in range(len(fixed_header_values), most_common_data_columns):
                col_name = f"Unnamed_{idx}"
                fixed_header_values.append(col_name)
                result['missing_header_columns'][idx] = {'inferred_type': 'unknown', 'sample_values': []}

            # Read the whole file with fixed headers
            df = pd.read_csv(
                file_path,
                sep=separator,
                encoding=encoding,
                skiprows=header_line_index + 1,
                header=None,
                names=fixed_header_values
            )
            result['recommended_action'] = f"Header had {header_columns} cols, but data has {most_common_data_columns}. Missing columns added."
            result['details'] = f"Inserted {most_common_data_columns - header_columns} placeholder columns in header."
        else:
            # Read normally
            df = pd.read_csv(
                file_path,
                sep=separator,
                encoding=encoding,
                skiprows=header_line_index,
                header=0
            )
            result['details'] = "Header and data columns match."

        result['df'] = df
        return result

    except Exception as e:
        result['details'] = f"Error reading file: {str(e)}"
        return result