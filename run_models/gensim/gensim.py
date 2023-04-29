
# classes
from classes.Datasets import Datasets
from classes.Process_Stages import Process_Stages

# constants
from data.splits.constants import *

def gensim():

    gensim_dataset = Datasets(
        force_re_run = False,

        base_dir=SPLITS,
        model_name="gensim",

        progress_stages=Process_Stages(
            lower=True, 
            only_text=True, 
            strip_extra_whitespace=True, 
            spelling_check=True,
            strip_punctuation=True,
        )
    )

    gensim_dataset.get_datasets()

    # pre-processing


    # models
