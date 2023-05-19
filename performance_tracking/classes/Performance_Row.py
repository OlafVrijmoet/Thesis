
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
            
            # id what is being tracked
            embedding_seperated: bool, # indicateds if two models are used, one for embedding and one for classifying (True) or one model is used from embedding and classifying (False). Because, the same model might be used in both embedding and classifying
            embedding_model_name, classfication_model_name, dataset_name,
            dataset_split, # is it training, test or validation - constants defined in performance_tracking

            # duplicates handeling
            settings_performance_tacking: int, # allowes experiements with same embedding_model_name, classfication_model_name, dataset_name to be added to performance df without asking
            
            rmse=None,
            pears_correlation=None,
            p_value=None,

            accuracy=None, precision_macro=None, recall_macro=None, f1_macro=None,
            precision_micro=None, recall_micro=None, f1_micro=None,
            precision_weighted=None, recall_weighted=None, f1_weighted=None) -> None:
        
        # dataframe of past performance, current experiement performance is added to this df
        self.past_performance = None

        # id'ing experiment
        self.row_id = 0 # default, if there is no other past performance
        self.embedding_seperated = embedding_seperated
        self.embedding_model_name = embedding_model_name
        self.classfication_model_name = classfication_model_name
        self.dataset_name = dataset_name
        self.dataset_split = dataset_split

        self.time_stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # duplicates handeling
        self.settings_performance_tacking = settings_performance_tacking
        
        # performance measurements, regression
        self.rmse = rmse
        self.pears_correlation = pears_correlation # still implement!!!
        self.p_value = p_value

        # performance measurements, classification
        self.accuracy = accuracy # what is this? Is this not just precision?!
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

        # fetch / create df for performance
        self.fetch_saved_performance()

        # check if experiement is done before
        experiement_done_before = self.check_for_duplicates()

        # to be able to re-use settings in class and run user selected settings with one var
        running_settings = self.settings_performance_tacking
        
        # check to prompt use, if indicated to prompt and experiement done before
        if experiement_done_before == True and running_settings == PROMPT_EXPERIMENT_DONE:

            # promt user for performance settings
            running_settings = prompt_user(
                prompt=f"""
                The experiment with embeddings: {self.embedding_model_name}, classifier: {self.classfication_model_name}, dataset: {self.dataset_name}.
                Your options: \n
                    Replace results with oldest findings: {REPLACE} \n
                    Add results to dataframe as new result: {ADD} \n
                    Delete new findings: {NO_SAVING} \n
                """, 
                user_options_values={
                    REPLACE: REPLACE,
                    ADD: ADD,
                    NO_SAVING: NO_SAVING
                }
            )
        
        # user responds no saving or general settings no saving, than skip it!
        if running_settings == NO_SAVING or (experiement_done_before == True and running_settings == NO_PROMPT_NO_REPEAT):
            
            # stop function, don't save results of this experiment
            return None
        
        # create row with current experiement data
        row_data = {

            # id'ing experiment
            'row_id': self.row_id,
            'embedding_seperated': self.embedding_seperated,
            'embedding_model_name': self.embedding_model_name,
            'classification_model_name': self.classfication_model_name,
            'dataset_name': self.dataset_name,
            'dataset_split': self.dataset_split,

            'time_stamp': self.time_stamp,

            # performance measurements, regression
            'rmse': self.rmse,
            'pears_correlation': self.pears_correlation,
            'p_value': self.p_value,

            # performance measurements, classification
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
            
        }

        if running_settings == ADD or experiement_done_before == False:

            # add row
            self.past_performance = self.past_performance.append(row_data, ignore_index=True)
        
        elif (experiement_done_before == True and running_settings == REPLACE):
            
            # Get the indices of the rows in the original DataFrame which match the conditions
            match_indices = self.past_performance[
                (self.past_performance.embedding_model_name == self.embedding_model_name) & 
                (self.past_performance.classification_model_name == self.classfication_model_name) &
                (self.past_performance.dataset_name == self.dataset_name) &
                (self.past_performance.dataset_split == self.dataset_split)
            ].index

            # Convert the time_stamp column to datetime format
            self.past_performance['time_stamp'] = pd.to_datetime(self.past_performance['time_stamp'])

            # Get the index of the row with the oldest time_stamp among the matched rows
            oldest_index = self.past_performance.loc[match_indices, 'time_stamp'].idxmin()

            # update the row
            self.past_performance.loc[oldest_index] = row_data

        else:
            print("Something weard happened. Results not saved!")
            print(f"experiement_done_before: {experiement_done_before}")
            print(f"running_settings: {running_settings}")

        # save df
        save(
            dir=DF_TRACKING_DIR,
            file_name=DF_TRACKING_FILE_NAME,
            df=self.past_performance
        )
    
    # print info
    def print_experiement_info(self):

        print("\n-----")
        print(f"embedding_model_name: {self.embedding_model_name}")
        print(f"classfication_model_name: {self.classfication_model_name}")
        print(f"dataset_name: {self.dataset_name}")

    def print_regression_preformance(self):

        print("\n*** regression performance ***")
        print(f"rmse: {self.rmse}")
        print(f"pears_correlation: {self.pears_correlation}")
        print(f"p_value: {self.p_value}")

    def print_classification_performance(self):

        # just the weighted measurments
        print("\n*** classification performance ***")
        print(f"accuracy: {self.accuracy}")
        print(f"precision_weighted: {self.precision_weighted}")
        print(f"recall_weighted: {self.recall_weighted}")
        print(f"f1_weighted: {self.f1_weighted}")

    # --- Services ---
        # only used internally

    # only do when saving, than the performance df will only be loaded into memory to add the row
    def fetch_saved_performance(self):
        found, past_performance = get_df(dir=DF_TRACKING_DIR, file_name=DF_TRACKING_FILE_NAME)
        
        if found == True:

            # save past performance df into past_performance df in class
            self.past_performance = past_performance

            # default is 0, so only change is len longer than 0
            if len(past_performance) != 0:
                
                # set row_id to max_value_id + 1
                max_value_id = self.past_performance['row_id'].max()
                self.row_id = max_value_id + 1

        else:

            # create new df with class column names
            column_names = [
                # id'ing experiment
                'row_id', 'embedding_seperated', 'embedding_model_name', 'classification_model_name', 'dataset_name', 'dataset_split',
                'time_stamp',
                
                # performance measurements, regression
                'rmse',
                'pears_correlation',
                'p_value'

                # performance measurements, classification
                'accuracy', 
                'precision_macro', 'recall_macro', 'f1_macro',
                'precision_micro', 'recall_micro', 'f1_micro',
                'precision_weighted', 'recall_weighted', 'f1_weighted'
            ]
            self.past_performance = pd.DataFrame(columns=column_names)

            # save new df - maybe don't do here?! - just when it's saved
            save(dir=DF_TRACKING_DIR, file_name=DF_TRACKING_FILE_NAME, df=self.past_performance)

    # returns True if experiement done before
    def check_for_duplicates(self):
        
        # gives df with all rows with unique experiement ids
        duplicate_row = self.past_performance[
            (self.past_performance['embedding_model_name'] == self.embedding_model_name) &
            (self.past_performance['classification_model_name'] == self.classfication_model_name) &
            (self.past_performance['dataset_name'] == self.dataset_name) & 
            (self.past_performance['embedding_seperated'] == self.embedding_seperated) & 
            (self.past_performance['dataset_split'] == self.dataset_split)
        ]

        # checks if df is empty, returns True if experiement done before
        return not duplicate_row.empty
    
    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)
