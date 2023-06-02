
import os

# classes
from performance_tracking.classes.Measurement_Settings import Measurement_Settings
from performance_tracking.classes.Dataset_api import Dataset_api

# classes local
from grading_models.api.classes.Openai_Grading import Openai_Grading
from grading_models.api.classes.Openai_Grading_Norm import Openai_Grading_Norm

# services
from services.get_df import get_df

# constants
from experiements.constants import SEEDS, SHOTS
from performance_tracking.constants import *
from constants_dir.path_constants import DATASETS_TO_SKIP

def api():

    for SEED in SEEDS:

        for SHOT in SHOTS:

            # iterate over all files in the folder
            for file_name in os.listdir("data/splits/data"):

                # get file name without file type for get_df
                file_name, _ = os.path.splitext(file_name)

                # if file_name in DATASETS_TO_SKIP:
                #     continue

                if file_name is not "concatenated_domains":

                    dataset = Dataset_api(
                        dir="data/splits/data",
                        file_name=file_name,
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
                            settings_performance_tracking=ADD,
                            save_performance=True
                        ),

                        # child
                        y_column="assigned_points",

                        y_normalized=False,

                        shots=SHOT

                    )

                    dataset_grading.validation()

                if file_name == "concatenated_domains" or file_name == "concatenated_datasets":

                    datasets = DATASETS
                    if file_name == "concatenated_domains":
                        datasets = DOMAINS

                    for dataset_name_to_split in datasets:

                        dataset = Dataset_api(
                            dir="data/splits/data",
                            file_name=file_name,
                            seed=SEED,
                            shots=SHOT, # DO NOT CHANGE UNTILL FINISHED WITH RUN THROUGH ALL DATASETS!
                            left_out_dataset=dataset_name_to_split
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
                                left_out_dataset=dataset_name_to_split,
                                
                                seed_data_split=SEED,

                                # inform user settings
                                print_regression=True,
                                print_classification=True,
                                
                                # save settings
                                settings_performance_tracking=NO_PROMPT_NO_REPEAT,
                                save_performance=True
                            ),

                            # child
                            y_column="assigned_points",

                            y_normalized=False,

                            shots=SHOT

                        )

                        dataset_grading.validation()
