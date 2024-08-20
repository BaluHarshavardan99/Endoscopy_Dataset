# Data-Pipeline

This folder contains scripts essential for processing endoscopy videos and creating the endoscopy text-image pairs dataset. Below is a brief description of each script:

## Scripts Overview

1. **Test.py**
    - Purpose: Extracts keyframes from endoscopy videos.
    - Details: Utilizes FFmpeg to process the videos and output keyframes based on defined hyperparameters.

2. **csv_file_creator.py**
    - Purpose: Generates CSV files that pair extracted keyframes with corresponding metadata.
    - Details: This script processes the keyframes and creates structured CSV files to be used in further data processing.

3. **add_class_column.py**
    - Purpose: Adds classification labels to the keyframes.
    - Details: Enhances the `output_csv_file.csv` by appending a new column that contains the classification labels for each keyframe, which are derived from the endoscopy classifier.

## Usage

1. Process the endoscopy videos and extract the keyframes using `Test.py`.
2. Use `csv_file_creator.py` to generate CSV files containing the relevant metadata and keyframe information.
3. Run `add_class_column.py` to append classification labels to the generated CSV file.

## Requirements

- FFmpeg
- Relevant Python libraries as specified in the main `README.md`.



