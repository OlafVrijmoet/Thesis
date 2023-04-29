
import os
import pandas as pd

class Dataset:

    def __init__(self, name, model_name) -> None:
        self.name = name
        self.model_name = model_name

        self.basic_processed_df = None

    def get_dataset(self):

        # check if there is already a dataset that has been processed before
        processed_dataset_exists = self.processed_dataset()

        if processed_dataset_exists == False:
            # fetch base dataset from data/splits/self.name
            self.basic_processed_df = pd.read_csv(f"data/splits/{self.name}.csv")

    def processed_dataset(self) -> bool:

        # check if basic processing already done, located at data_saved/basic_processed/name
        if os.path.exists(f"data_saved/basic_processed/{self.name}.csv"):
            # fetch data and save it to self.basic_processed_df
            self.basic_processed_df = pd.read_csv(f"data_saved/basic_processed/{self.name}.csv")
            return True
        else:
            return False

    def process_dataset(self):

        # itterate rows of basic_processed_df
        for index, row in self.basic_processed_df.iterrows():
            # process row
            processed_row = self.process_row(row)
            self.basic_processed_df.loc[index] = processed_row

    def process_row(self, row):
        # Implement row processing logic here
        pass

    def save(self):

        # save basic_processed_df at data_saved/basic_processed/name, create dir if it doesn't exist yet
        if not os.path.exists("data_saved/basic_processed"):
            os.makedirs("data_saved/basic_processed")
        self.basic_processed_df.to_csv(f"data_saved/basic_processed/{self.name}.csv", index=False)
