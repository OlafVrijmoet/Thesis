
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

        # options:
            # prompt user - done
            # delete performance / don't save latest performance - done
            # add row - done
            # replace oldest row - not done

        user_response = None
        
        # check to prompt use, if indicated to prompt and experiement done before
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
        
        # user responds no saving or general settings no saving, than skip it!
        if (user_response == NO_PROMPT_NO_REPEAT and experiement_done_before == True) or (experiement_done_before == True and self.settings_performance_tacking == NO_PROMPT_NO_REPEAT):
            
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

        if user_response == ADD or self.settings_performance_tacking == ADD:

            # add row
            self.past_performance = self.past_performance.append(row_data, ignore_index=True)
        
        # todo:
            # find index: self.past_performance on lastest time_stamp & same experiement
            # replace row at found index
        elif (experiement_done_before == True and user_response == REPLACE) or (experiement_done_before == True and self.settings_performance_tacking == REPLACE):
            
            # Filter the DataFrame for same experiment
            filtered_df = self.past_performance.query("embedding_model_name == @self.embedding_model_name and classification_model_name == @self.classfication_model_name and dataset_name == @self.dataset_name and dataset_split == @self.dataset_split")

            # !!! filtered_df should not be used to find index, find index inside orininal, including other columns to unieuqly indentify it. !!!

            # !!! also what hoppens if same date, different time ?!!!!

            # find the row with the oldest date
            # Convert the 'Date' column to datetime
            # ik vind jou lief olaf, hoop dat deze code goed werkt!!!!  # Find the index of the row with the oldest date
            
            
            oldest_index = filtered_df['time_stamp'].idxmin()

            # update the row
            df.loc[oldest_index, row_dict.keys()] = row_dict.values()

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
        
        print(found)
        print(past_performance)
        
        if found == True:

            # save past performance df into past_performance df in class
            self.past_performance = past_performance

            # default is 0, so only change is len longer than 0
            if len(past_performance) != 0:
                # set row_id to row_id of last row + 1
                last_row_id = self.past_performance['row_id'].iloc[-1]
                self.row_id = last_row_id + 1

        else:

            # create new df with class column names
            column_names = [
                # id'ing experiment
                'row_id', 'embedding_seperated', 'embedding_model_name', 'classification_model_name', 'dataset_name', 'dataset_split',
                'time_stamp',
                
                # performance measurements, regression
                'rmse',
                'pears_correlation',

                # performance measurements, classification
                'accuracy', 
                'precision_macro', 'recall_macro', 'f1_macro',
                'precision_micro', 'recall_micro', 'f1_micro',
                'precision_weighted', 'recall_weighted', 'f1_weighted'
            ]
            self.past_performance = pd.DataFrame(columns=column_names)

            # save new df - maybe don't do here?! - just when it's saved
            save(dir=DF_TRACKING_DIR, file_name=DF_TRACKING_FILE_NAME, df=self.past_performance)

    # !!! ToDo !!!
    # returns True if experiement done before
    def check_for_duplicates(self):
        
        # gives df with all rows with unique experiement ids
        duplicate_row = self.past_performance[
            (self.past_performance['embedding_model_name'] == self.embedding_model_name) &
            (self.past_performance['classification_model_name'] == self.classfication_model_name) &
            (self.past_performance['dataset_name'] == self.dataset_name) & 
            (self.past_performance['embedding_seperated'] == self.embedding_seperated)
        ]

        # checks if df is empty, returns True if experiement done before
        return not duplicate_row.empty

        # if row_already_exists and not self.repeat_experiement_allowed:

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
