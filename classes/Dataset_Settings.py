
from sklearn.model_selection import train_test_split

class Dataset_Settings:

    def __init__(self, df, may_run_now, required, done = False, parquet=False, name_required_dataset=None, 
            x_train=None, x_test=None, x_validation=None, y_train=None, y_test=None, y_validation=None) -> None:
        
        self.df = df
        self.may_run_now = may_run_now
        self.required = required
        self.done = done # will check if it is done inside the Dataset class
        self.parquet = parquet

        self.name_required_dataset = name_required_dataset

        x_train = x_train
        x_test = x_test
        x_validation = x_validation

        y_train = y_train
        y_test = y_test
        y_validation = y_validation
    
    def split_datasets(self, seed, x_column_name, y_column_name):

        # get values
        x = self.df[x_column_name].values.reshape(-1, 1)
        y = self.df[y_column_name]

        # Split the DataFrame into train (70%) and the remaining data (30%)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.3, random_state=seed)

        # Split the remaining data further into test (20% of the original dataset) and validation (10% of the original dataset) sets
        self.x_test, self.x_validation, self.y_test, self.y_validation = train_test_split(self.y_train, self.y_test, test_size=1/3, random_state=seed)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)
