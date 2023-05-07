
from sklearn.model_selection import train_test_split

class Dataset_Settings:

    def __init__(self, df, may_run_now, required, done = False, parquet=False, name_required_dataset=None, force_run=False,
                train_df=None, test_df=None, validation_df=None,
            x_train=None, x_test=None, x_validation=None, y_train=None, y_test=None, y_validation=None) -> None:
        
        self.df = df
        self.may_run_now = may_run_now
        self.required = required
        self.done = done # will check if it is done inside the Dataset class
        self.parquet = parquet
        self.force_run = force_run

        self.name_required_dataset = name_required_dataset

        self.train_df = train_df
        self.test_df = test_df
        self.validation_df = validation_df
        
        x_train = x_train
        x_test = x_test
        x_validation = x_validation

        y_train = y_train
        y_test = y_test
        y_validation = y_validation
    
    def split_datasets(self, seed):

        # Split the DataFrame into train (70%) and the remaining data (30%)
        self.train_df, temp_df = train_test_split(self.df, test_size=0.3, random_state=seed)

        # Split the remaining data further into test (20% of the original dataset) and validation (10% of the original dataset) sets
        self.test_df, self.validation_df = train_test_split(temp_df, test_size=1/3, random_state=seed)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)
