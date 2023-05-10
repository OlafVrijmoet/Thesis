
import pandas as pd

# classes
from services.get_df import get_df
from services.save import save

# constants
from performance_tracking.constants import *

class Performance_Row:

    def __init__(self, embedding_model_name, classfication_model_name, dataset_name,
                 dataset_performance=None,  # test set, validation set
                 rmse=None,
                 accuracy=None, precision_macro=None, recall_macro=None, f1_macro=None,
                 precision_micro=None, recall_micro=None, f1_micro=None,
                 precision_weighted=None, recall_weighted=None, f1_weighted=None) -> None:
        
        self.row_add = False
        
        self.past_performance = None
        
        self.row_id = 0 # default if there is no other past performance
        # self.embedding_id = embedding_id
        self.embedding_model_name = embedding_model_name
        # self.classification_id = classification_id
        self.classfication_model_name = classfication_model_name
        # self.dataset_id = dataset_id, 
        self.dataset_name = dataset_name,

        self.dataset_performance = dataset_performance

        self.rmse = rmse

        self.accuracy = accuracy
        self.precision_macro = precision_macro
        self.recall_macro = recall_macro
        self.f1_macro = f1_macro
        self.precision_micro = precision_micro
        self.recall_micro = recall_micro
        self.f1_micro = f1_micro
        self.precision_weighted = precision_weighted
        self.recall_weighted = recall_weighted
        self.f1_weighted = f1_weighted

    def save(self):

        if self.row_add == False:

            # fetch / create df for performance
            self.fetch_saved_performance()

            # add row with this data
            row_data = {
                'row_id': self.row_id,
                'embedding_id': None,  # or any default value if not available
                'embedding_model_name': self.embedding_model_name,
                'classification_id': None,  # or any default value if not available
                'classification_model_name': self.classfication_model_name,
                'dataset_id': None,  # or any default value if not available
                'dataset_name': self.dataset_name,
                'dataset_performance': self.dataset_performance,
                'accuracy': self.accuracy,
                'precision_macro': self.precision_macro,
                'recall_macro': self.recall_macro,
                'f1_macro': self.f1_macro,
                'precision_micro': self.precision_micro,
                'recall_micro': self.recall_micro,
                'f1_micro': self.f1_micro,
                'precision_weighted': self.precision_weighted,
                'recall_weighted': self.recall_weighted,
                'f1_weighted': self.f1_weighted,
                'rmse': self.rmse
            }

            self.past_performance = self.past_performance.append(row_data, ignore_index=True)

            # save df
            save(
                dir=DF_TRACKING_DIR,
                file_name=DF_TRACKING_FILE_NAME,
                df=self.past_performance
            )

        else:
            print("Data for row has already been added to performance df!")

    # only do when saving, than the performance df will only be loaded into memory to add the row
    def fetch_saved_performance(self):

        found, past_performance = get_df(dir=DF_TRACKING_DIR, file_name=DF_TRACKING_FILE_NAME)

        if found == True:
            
            # save past performance df into past_performance df in class
            self.past_performance = past_performance

            # set row_id to row_id of last row + 1
            last_row_id = self.past_performance['row_id'].iloc[-1]
            self.row_id = last_row_id + 1

        else:

            # create new df with class column names
            column_names = [
                'row_id', 'embedding_id', 'embedding_model_name', 'classification_id', 'classification_model_name',
                'dataset_performance', 'accuracy', 'precision_macro', 'recall_macro', 'f1_macro',
                'precision_micro', 'recall_micro', 'f1_micro',
                'precision_weighted', 'recall_weighted', 'f1_weighted'
            ]
            self.past_performance = pd.DataFrame(columns=column_names)

            # save new df
            save(dir=DF_TRACKING_DIR, file_name=DF_TRACKING_FILE_NAME, df=self.past_performance)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)
