
import os
import glob
import pandas as pd

# classes
from classes.Grading_Model import Grading_Model
from experiements.classes.Dataset import Dataset

# constants
from experiements.constants import SEEDS

def regression():

    # loop through all cosine_similarity datasets

        # maybe as in feature_engenearing, create a data_dict?
        # save the df inside Dataset

        # dataset.split_datasets()

    # Define the directory where the data is located
    base_dir = 'data/feature_engenearing/data/'

    # Use glob to get a list of all the parquet files in the directory
    dataset_files = glob.glob(os.path.join(base_dir, '**/*.parquet'), recursive=True)

    # Initialize an empty list to store the dictionaries
    data_list = []

    # Loop through all the dataset files
    for dataset_file in dataset_files:
        # Split the file path into parts
        parts = dataset_file.split(os.sep)

        # Extract the needed parts from the split file path
        word_embed_model = parts[3]
        abriviation_method = parts[5]
        abriviation_method_method = parts[7]
        dataset_name = parts[8].replace('.parquet', '')

        # Read the parquet file as a pandas DataFrame
        df = pd.read_parquet(dataset_file)

        # Create a dictionary with the necessary information and DataFrame
        data_dict = {
            'word_embed_model': word_embed_model,
            'abriviation_method': abriviation_method,
            'abriviation_method_method': abriviation_method_method,
            'dataset_name': dataset_name,
            'dataset': df
        }

        # Append the dictionary to the data_list
        data_list.append(data_dict)

    # loop through data_list to get results
    for seed in SEEDS:
        for data in data_list:

            dataset = Dataset(
                dir="data/feature_engenearing/data/consine_similarity/data/embed_words_conceptnet/data/add/cosine_similarity/data/",
                file_name="texas",
                seed=seed,
            )

            dataset.split_datasets()

        # continue here...