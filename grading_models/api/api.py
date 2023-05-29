
import os

# classes
from performance_tracking.classes.Measurement_Settings import Measurement_Settings
from performance_tracking.classes.Dataset_api import Dataset_api

# classes local
from grading_models.api.classes.Openai_Grading import Openai_Grading
from grading_models.api.progress_tracking.classes.Logs import Logs

# services
from services.get_df import get_df

# constants
from experiements.constants import SEEDS, SHOTS
from performance_tracking.constants import *

def api():

    for SEED in SEEDS:

        for SHOT in SHOTS:

            # iterate over all files in the folder
            for file_name in os.listdir("data/splits/data"):

                # get logs
                logs = Logs()

                # get file name without file type for get_df
                file_name, _ = os.path.splitext(file_name)

                # dataset = Dataset_api(
                #     dir="data/splits/data",
                #     file_name=file_name,
                #     seed=SEED,
                #     shots=SHOT # DO NOT CHANGE UNTILL FINISHED WITH RUN THROUGH ALL DATASETS!
                # )
                
                dataset = Dataset_api(
                    dir="testing/data",
                    file_name="sample",
                    seed=SEED,
                    shots=SHOT # DO NOT CHANGE UNTILL FINISHED WITH RUN THROUGH ALL DATASETS!
                )

                dataset.split_datasets()

                dataset_grading = Openai_Grading(
                    # parent
                    model="gpt-3.5-turbo",
                    dataset=dataset,
                    measurement_settings=Measurement_Settings(
                        dataset_name=dataset["name"],
                        embedding_seperated=False,
                        embedding_model_name="gpt-3.5-turbo",
                        sentence_embedding_method=None,
                        feature_engenearing_method=None,
                        grading_model="gpt-3.5-turbo",
                        
                        seed_data_split=SEED,

                        # inform user settings
                        print_regression=True,
                        print_classification=True,
                        
                        # save settings
                        settings_performance_tracking=NO_SAVING,
                        save_performance=True
                    ),

                    # child
                    y_column="assigned_points",

                    y_normalized=True,

                )

                dataset_grading.validation()
