
import pandas as pd

# serives
from services.get_dfs import get_dfs
from services.get_df import get_df

class Logs:

    def __init__(self) -> None:
        pass
        
        self.logs_df = self.get_logbook()

    def get_logbook(self):

        # check if there are logs
        found_logs, _, logs_df = get_df(dir="grading_models/api/progress_tracking", file_name="logs")

        # create new log df if no log file exists
        if found_logs == False:

            dfs = {

            }
            get_dfs(dict=dfs, dir="data/splits/data")

            # Initialize an empty list to hold the records
            records = []

            # Loop over the dictionary of dataframes
            for df_name, df in dfs.items():
                record = {
                    "df_name": df_name,
                    "finished_predictions": False,  # replace with your condition
                    "index": 0,  # get the last index in the dataframe
                    "split": "",  # add your split info here
                    "length": len(df),  # get the length of the dataframe
                }
                records.append(record)

            # Create the new dataframe
            logs_df = pd.DataFrame(records)

        return logs_df

    def update_index_df(self, df_name, index):
        # Find the row with the df_name and update the index value
        self.logs_df.loc[self.logs_df['df_name'] == df_name, 'index'] = index

        # Save the updated logs_df to a CSV file
        self.logs_df.to_csv("grading_models/api/progress_tracking/logs.csv", index=False)

    def get_index_df(self, df_name):

        # Use query to get the row and then access the 'index' column
        index_value = self.logs_df.query("df_name == @df_name")['index'].values[0]

        return index_value

    # get attribute values
    def __getitem__(self, key):
        return getattr(self, key)
    
    # set attribute values
    def __setitem__(self, key, value):
        setattr(self, key, value)    
