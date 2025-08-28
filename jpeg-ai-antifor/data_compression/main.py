import pandas as pd
def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.set_index('id')
    df = df.drop(columns=['Unnamed: 0', 'original_path'])
    return df