import pandas as pd
import os
import numpy as np

def split_csv_by_country(input_file, output_folder="country_data"):
    """
    Split a CSV file into multiple files based on country code (first column).
    Remove columns that are completely empty in each country's data.
    
    Args:
        input_file (str): Path to the input CSV file
        output_folder (str): Folder to save the output files (created if doesn't exist)
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output directory: {output_folder}")
    
    # Read the CSV file
    print(f"Reading CSV file: {input_file}")
    df = pd.read_csv(input_file)
    
    country_col = df.columns[0]
    
    # Group by country and save each group to a separate file
    countries = df[country_col].unique()
    print(f"Found {len(countries)} unique countries")
    
    for country in countries:
        country_data = df[df[country_col] == country].copy()
        
        # Remove columns that are entirely empty
        empty_columns = []
        for column in country_data.columns:
            # Check if all values in the column are empty
            is_empty = country_data[column].isna() | (country_data[column] == '') | (country_data[column].astype(str) == 'nan')
            if is_empty.all():
                empty_columns.append(column)
        
        # Drop empty columns
        if empty_columns:
            country_data.drop(columns=empty_columns, inplace=True)
            print(f"Removed {len(empty_columns)} empty columns for {country}")
        
        # Create a valid filename
        safe_country_name = str(country).replace('/', '_').replace('\\', '_')
        output_file = os.path.join(output_folder, f"{safe_country_name}.csv")
        
        # Save to CSV
        country_data.to_csv(output_file, index=False)
        print(f"Saved {len(country_data)} rows with {len(country_data.columns)} columns for {country} to {output_file}")


if __name__ == "__main__":
    input_csv = "data/merged_data.csv"
    output_dir = "data/by_country"
    
    split_csv_by_country(input_csv, output_dir)
    print("CSV splitting complete!")