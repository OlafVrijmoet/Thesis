
# libaries
import os
import pandas as pd

# classes
from data.processed.classes.Split_Data import Split_Data

# constants
from data.processed.constants import *
from data.split.constants import *

def split_data():

    data_split = Split_Data(
        datasets_base_dir=SPLITS,
        dir_datasets=STANDARDIZED_BASE,
        dir_new_datasets=SPLITS,

        save_existing=True
    )

    data_split.create_data_splits()
