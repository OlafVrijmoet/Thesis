
import os

def save(dir, file_name, df, parquet=False):

    if not os.path.exists(dir):
        os.makedirs(dir)

    if parquet:
        df.to_parquet(f"{dir}/{file_name}.parquet", index=False)
    else:
        df.to_csv(f"{dir}/{file_name}.csv", index=False)
