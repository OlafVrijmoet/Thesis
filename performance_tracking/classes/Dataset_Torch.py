
import torch

# classes
from performance_tracking.classes.Dataset import Dataset

# services
from services.get_df import get_df

class Dataset_Torch(Dataset):

    def __init__(self, dir, file_name, seed, left_out_dataset=None) -> None:
        super().__init__(dir, file_name, seed, left_out_dataset)

    def get_data(self, dir, file_name):

        # get df
        found, df_name, dataset = get_df(dir=dir, file_name=file_name, know_type="csv")
        found_pth, df_name_pth, dataset_pth = get_df(dir=dir, file_name=file_name, know_type="pth")

        if dataset == None or dataset_pth == None:

            return None
    
        dataset["tokenized_for_BERT"] = dataset_pth

        return dataset
