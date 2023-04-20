
import os

def save(dir, file_name, df):

    if not os.path.exists(dir):
        os.makedirs(dir)

    df.to_csv(f"{dir}/{file_name}.csv", index=False)
