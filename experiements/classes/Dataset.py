
from sklearn.model_selection import train_test_split

# services
from services.get_df import get_df

class Dataset:

    def __init__(self,

        # to get dataset
        dir,
        file_name,

        # seed for splitting data
        seed

    ) -> None:
        
        # split seed
        self.seed = seed
        
        # original dataset
        self.dataset = self.get_data(dir, file_name)

        # places to save the splits
        self.train_df = None
        self.test_df = None
        self.validation_df = None

    # get data using dataset_dir
    def get_data(self, dir, file_name):

        found, dataset = get_df(dir=dir, file_name=file_name)

        return dataset
    
    def split_datasets(self):

        # Split the DataFrame into train (70%) and the remaining data (30%)
        self.train_df, temp_df = train_test_split(self.dataset, test_size=0.3, random_state=self.seed)

        # Split the remaining data further into test (20% of the original dataset) and validation (10% of the original dataset) sets
        self.test_df, self.validation_df = train_test_split(temp_df, test_size=1/3, random_state=self.seed)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)
