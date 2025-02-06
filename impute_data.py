import pandas as pd
import statsmodels.imputation.mice as mice
from scipy.stats import mode

def impute_missing_values(filename):
    df = pd.read_csv(filename)

    # Remove 'site_ID' and 'subj_id' columns
    df = df.drop(columns=['site_ID', 'subj_id'])

    # Map 'VENTDOM' values to numbers
    ventdom_mapping = {'LV': 1, 'RV': 2, 'NONE': 3}
    df['VENTDOM'] = df['VENTDOM'].map(ventdom_mapping)

    # Check if 'VENTDOM' has missing values and impute using mode if necessary
    if df['VENTDOM'].isnull().any():
        ventdom_mode = mode(df['VENTDOM'].dropna())[0][0]
        df['VENTDOM'].fillna(ventdom_mode, inplace=True)

    # Exclude non-numeric columns from MICE imputation
    numeric_cols = df.select_dtypes(include=['number']).columns
    numeric_df = df[numeric_cols]

    # Apply MICE imputation to only numeric columns
    imputer = mice.MICEData(numeric_df)
    num_iterations = 10
    for i in range(num_iterations):
        imputer.update_all()
    df_imputed_numeric = imputer.data

    # Merge imputed numeric data back with non-numeric data
    non_numeric_cols = df.columns.difference(numeric_cols)
    df_imputed = df[non_numeric_cols].join(df_imputed_numeric)

    # DataFrame with missing rows removed
    df_dropped = df.dropna()

    print(f"Rows in DataFrame with imputed values: {df_imputed.shape[0]}")
    print(f"Rows in DataFrame with missing values removed: {df_dropped.shape[0]}")

    return df_imputed, df_dropped

def convert_ventdom_back(df):
    """
    Convert numeric values in 'VENTDOM' column back to original categorical values.

    Parameters:
    df (DataFrame): The DataFrame with the 'VENTDOM' column in numeric format.

    Returns:
    DataFrame: The DataFrame with the 'VENTDOM' column in original categorical format.
    """

    # Reverse mapping for 'VENTDOM'
    reverse_ventdom_mapping = {1: 'LV', 2: 'RV', 3: 'NONE'}

    # Apply reverse mapping
    df['VENTDOM'] = df['VENTDOM'].map(reverse_ventdom_mapping)

    return df

# Example usage
# imputed_df, dropped_df = impute_missing_values('path_to_your_file.csv')
