
import os
import pandas as pd

def get_df(dir, file_name, parquet=False):
    # identify file type
    file_type = ".parquet" if parquet == True else ".csv"

    # check if file exsits
    if os.path.exists(f"{dir}/{file_name}{file_type}"):
        
        # create df from parquet file
        if parquet == True:
            df = pd.read_parquet(f"{dir}/{file_name}{file_type}")

        # create df from csv file
        else:
            df = pd.read_csv(f"{dir}/{file_name}{file_type}")

        print(f"Found {file_name} with type {file_type} in ({dir}) and converted it to df")

        # return info on existance and df
        return True, df

    else:
        print(f"File {file_name} not found in ({dir})")

        # return info on existance
        return False, None
