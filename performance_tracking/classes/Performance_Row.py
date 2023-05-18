
import pandas as pd
from datetime import datetime

# classes
from services.get_df import get_df
from services.save import save

# services
from services.prompt_user import prompt_user

# constants
from performance_tracking.constants import *

class Performance_Row:

    def __init__(self,
                 
                 settings_performance_tacking, # allowes experiements with same embedding_model_name, classfication_model_name, dataset_name to be added to performance df without asking
                 embedding_seperated, # indicateds if two models are used, one for embedding and one for classifying (True) or one model is used from embedding and classifying (False). Because, the same model might be used in both embedding and classifying

                 embedding_model_name, classfication_model_name, dataset_name,

                 dataset_performance=None,  # test set, validation set
                 rmse=None,
                 accuracy=None, precision_macro=None, recall_macro=None, f1_macro=None,
                 precision_micro=None, recall_micro=None, f1_micro=None,
                 precision_weighted=None, recall_weighted=None, f1_weighted=None) -> None:
        
        self.row_add = False

        self.settings_performance_tacking = settings_performance_tacking
        self.embedding_seperated = embedding_seperated
        
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

        self.time_stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def save(self):

        if self.row_add == False:

            # fetch / create df for performance
            self.fetch_saved_performance()

            # check if there is already performance mearsure for embedding, classifier and dataset
            
            experiement_done_before = self.check_for_duplicates()

            if experiement_done_before == True and self.settings_performance_tacking == PROMPT_EXPERIMENT_DONE:

                # promt user for performance settings
                user_response = prompt_user(
                    prompt=f"""
                    The experiment with embeddings: {self.embedding_model_name}, classifier: {self.classfication_model_name}, dataset: {self.dataset_name}.
                    Your options: \n
                        Replace results with oldest findings: {REPLACE} \n
                        Add results to dataframe as new result: {ADD} \n
                        Delete new findings: {NO_PROMPT_NO_REPEAT}
                    """, 
                    user_options_values={
                        REPLACE: REPLACE,
                        ADD: ADD,
                        NO_PROMPT_NO_REPEAT: NO_PROMPT_NO_REPEAT
                    }
                )
            
            # user responts no saving or general settings no saving, than skip it!
            if user_response == NO_PROMPT_NO_REPEAT or (experiement_done_before == True and self.settings_performance_tacking == NO_PROMPT_NO_REPEAT):
                
                # stop function
                return None
            
            # add row with this data
            row_data = {
                'row_id': self.row_id,
                'embedding_seperated': self.embedding_seperated,
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
                'rmse': self.rmse,
                'time_stamp': self.time_stamp
            }

            if user_response == REPLACE or (experiement_done_before == True and self.settings_performance_tacking == REPLACE):
                
                # Filter the DataFrame
                filtered_df = self.past_performance.query("embedding_model_name == @self.embedding_model_name and classification_model_name == @self.classfication_model_name and dataset_name == @self.dataset_name")

                # !!! filtered_df should not be used to find index, find index inside orininal, including other columns to unieuqly indentify it. !!!

                # !!! also what hoppens if same date, different time ?!!!!


                # find the row with the oldest date
                # Convert the 'Date' column to datetime
               ik vind jou lief olaf, hoop dat deze code goed werkt!!!!  # Find the index of the row with the oldest date
                oldest_index = filtered_df['time_stamp'].idxmin()

                # update the row
                df.loc[oldest_index, row_dict.keys()] = row_dict.values()
                
            # if self.settings_performance_tacking == ADD -> let go


            

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
                'row_id', 'embedding_id', 'embedding_model_name', 'classification_id', 'classification_model_name', 'dataset_id', 'dataset_name'
                'dataset_performance', 'accuracy', 'precision_macro', 'recall_macro', 'f1_macro',
                'precision_micro', 'recall_micro', 'f1_micro',
                'precision_weighted', 'recall_weighted', 'f1_weighted'
            ]
            self.past_performance = pd.DataFrame(columns=column_names)

            # save new df
            save(dir=DF_TRACKING_DIR, file_name=DF_TRACKING_FILE_NAME, df=self.past_performance)

    def check_for_duplicates(self):

        duplicate_row = self.past_performance[
            (self.past_performance['embedding_model_name'] == self.embedding_model_name) &
            (self.past_performance['classification_model_name'] == self.classfication_model_name) &
            (self.past_performance['dataset_name'] == self.dataset_name)
        ]

        return not duplicate_row.empty

        if row_already_exists and not self.repeat_experiement_allowed:

            # change to ask for the following options:
                # - no redoing experiements and no individual propts
                # - no redoing experiements but asking individual propts
                # - redoing experiements allowed, replace old experiement (replace oldest)
                # - redoing experiements allowed, add experiement

            # redo:
            # print(f"The experiment that is about to be done has already been done: {self.embedding_model_name, self.classfication_model_name, self.dataset_name}")
            # replace_results = get_yes_no_input("Do you want to replace the results? (Yes/No): ")

            # add_results = get_yes_no_input("Do you want to add results anyway as a new row? (Yes/No): ")

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)
