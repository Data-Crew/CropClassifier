"""
This script extracts the compressed file generated in the `create_ground_truth` notebook and writes a new Parquet file 
containing the CDL layers.

It ensures the required directories exist, extracts the contents of the ZIP file, and moves the extracted files to 
a structured output directory.

### Usage Example:
Run the script from the command line as follows:

```bash
python extract_CDL_layers.py
```

Ensure that the `CDL_samples.zip` file is placed inside the `data/` directory before running the script.
"""

import os
import zipfile
import shutil

def extract_cdl_layers():
    # Define local paths
    data_dir = "data/"
    zip_filename = "CDL_samples.zip"
    output_dir = "data/CDL_samples_extracted/"
    parquet_output_dir = "data/CDL_samples.parquet/"

    # Ensure the `data/` directory exists
    os.makedirs(data_dir, exist_ok=True)

    # Extract the ZIP file into `data/CDL_samples_extracted/`
    zip_path = os.path.join(data_dir, zip_filename)

    if os.path.exists(zip_path):
        os.makedirs(output_dir, exist_ok=True)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        
        print(f"Extraction complete: {output_dir}")

        # Move extracted files to `data/CDL_samples.parquet/`
        os.makedirs(parquet_output_dir, exist_ok=True)
        
        for file_name in os.listdir(output_dir):
            shutil.move(os.path.join(output_dir, file_name), parquet_output_dir)
        
        print(f"Files moved to: {parquet_output_dir}")

        # Optionally, remove the temporary extraction directory
        shutil.rmtree(output_dir)
    else:
        print(f"Error: {zip_path} not found. Please place the ZIP file in the 'data/' directory.")

if __name__ == "__main__":
    extract_cdl_layers()
