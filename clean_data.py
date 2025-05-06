import pandas as pd
import glob
import os
import csv

# Create folder to save cleaned files if it doesn't exist
os.makedirs('cleaned_data', exist_ok=True)

def clean_scimagojr_data(file_path):
    try:
        # Safely read the file even if some rows are malformed
        df = pd.read_csv(file_path, delimiter=';', quotechar='"', on_bad_lines='warn', low_memory=False)
    except Exception as e:
        print(f"❌ Failed to read {file_path}: {e}")
        return None

    # Convert European decimal commas to dots and cast to float
    for col in ['SJR', 'Cites / Doc. (2years)', 'Ref. / Doc.']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(',', '.', regex=False).astype(float)

    # Strip whitespace from all string columns
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].str.strip()

    # Split ISSN into two separate columns if the column exists
    if 'Issn' in df.columns:
        df[['ISSN Primary', 'ISSN Secondary']] = df['Issn'].str.split(pat=',', n=1, expand=True)

    # Handle missing values
    if 'Publisher' in df.columns:
        df['Publisher'] = df['Publisher'].fillna('Unknown')
    if 'SJR' in df.columns:
        df['SJR'] = df['SJR'].fillna(0)

    return df

# Update this path to where your raw data files are located
all_files = glob.glob("C:/Users/Yasi/desktop/Starma/raw_data/sjr-journal/scimagojr-journal-*.csv")

for file_path in all_files:
    year = os.path.basename(file_path).split('-')[-1].split('.')[0]
    cleaned_df = clean_scimagojr_data(file_path)
    if cleaned_df is None:
        print(f"⚠️ Skipping {file_path} due to a read or parse error.")
        continue
    cleaned_df.to_csv(f'cleaned_data/cleaned-journal-{year}.csv', index=False, quoting=csv.QUOTE_ALL)
    print(f"✅ Cleaned file for {year} saved to 'cleaned_data/cleaned-journal-{year}.csv'")
