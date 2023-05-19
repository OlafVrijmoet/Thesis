
import os
import pandas as pd

def get_df(dir, file_name, parquet=False): # can get rid of parquet

    # check if parquet file exsits
    if os.path.exists(f"{dir}/{file_name}.parquet"):

        df = pd.read_parquet(f"{dir}/{file_name}.parquet")

        print(f"Found {file_name} with type parquet in ({dir}) and converted it to df")

        # return info on existance
        return True, df
    
    elif os.path.exists(f"{dir}/{file_name}.csv"):

        df = pd.read_csv(f"{dir}/{file_name}.csv")

        print(f"Found {file_name} with type csv in ({dir}) and converted it to df")
        
        # return info on existance
        return True, df
    
    else:

        print(f"File {file_name} not found in ({dir})")

        # return info on existance
        return False, None
