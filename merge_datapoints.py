import os
import pandas as pd
import glob

def merge_csv_files(input_folder, output_file):
    """
    Merge all CSV files in a folder into one file.
    
    For country metadata files (identified by containing 'country' column and no duplicate countries),
    copy the country data to every year a country appears in the merged dataset.
    
    Parameters:
    -----------
    input_folder : str
        Path to folder containing CSV files
    output_file : str
        Path to save the merged output file
    """
    # Get list of all CSV files in the folder
    csv_files = glob.glob(os.path.join(input_folder, "*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {input_folder}")
        return
    
    # Initialize an empty list to store DataFrames
    dfs = []
    country_metadata_df = None
    
    for file in csv_files:
        print(f"Reading file: {file}")
        df = pd.read_csv(file)
        
        # Check if this is potentially the country metadata file
        if 'country' in df.columns and not df.duplicated(subset=['country']).any():
            print(f"Found country metadata file: {file}")
            country_metadata_df = df
        elif 'geo' in df.columns and 'time' in df.columns:
            dfs.append(df)
    
    if not dfs:
        print("No regular data files found with 'geo' and 'time' columns")
        return
    
    # Merge all regular data files on 'geo' and 'time'
    print("Merging regular data files...")
    merged_df = dfs[0]
    for df in dfs[1:]:
        merged_df = pd.merge(merged_df, df, on=['geo', 'time'], how='outer')
    
    # Handle the country metadata file if it exists
    if country_metadata_df is not None:
        print("Adding country metadata...")
        # Rename 'country' column to 'geo' for merging
        country_metadata_df = country_metadata_df.rename(columns={'country': 'geo'})
        
        # Merge with the country metadata
        merged_df = pd.merge(merged_df, country_metadata_df, on='geo', how='left')
    
    # Sort by country and time
    merged_df = merged_df.sort_values(['geo', 'time'])
    
    # Save the merged DataFrame to CSV
    print(f"Saving merged file to {output_file}")
    merged_df.to_csv(output_file, index=False)
    print("Merge completed successfully!")
    
    print(f"Total rows in merged file: {len(merged_df)}")
    print(f"Total columns in merged file: {len(merged_df.columns)}")
    print(f"Columns in merged file: {merged_df.columns.tolist()}")

if __name__ == "__main__":
    input_folder = "data/datapoints"
    output_file = "data/merged_data.csv"
    
    merge_csv_files(input_folder, output_file)