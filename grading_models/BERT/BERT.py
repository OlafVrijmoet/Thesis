
import os
from grading_models.BERT.classes.Py_Torch import Py_Torch
from performance_tracking.classes.Measurement_Settings import Measurement_Settings
from performance_tracking.constants import *

from transformers import BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup

# classes
from performance_tracking.classes.Measurement_Settings import Measurement_Settings
from performance_tracking.classes.Dataset_Torch import Dataset_Torch

# classes local
from grading_models.BERT.classes.Py_Torch import Py_Torch

# services
from services.get_df import get_df

# constants
from experiements.constants import SEEDS, SHOTS
from performance_tracking.constants import *
from constants_dir.path_constants import DATASETS_TO_SKIP

def bert():

    model_name = "bert-base-cased"
    
    base_dir = f"data/BERT_ASAG_tokenization/data/{model_name}/data/spelling_corrected/BERT_tokens/data"
    
    for SEED in SEEDS:

        # Use os.listdir to get a list of all files in the directory
        for filename in os.listdir(base_dir):
            # Concatenate the directory name with the filename to get the full path
            full_path = os.path.join(f"{base_dir}/", filename)
            
            # Check if the path is a file
            if os.path.isfile(full_path):

                df_name = os.path.splitext(filename)[0]

                if df_name in DATASETS_TO_SKIP:
                        continue
                
                if df_name is not "concatenated_domains":

                    print(f"Running {model_name} on {df_name}")

                    # Now you can do whatever you want with the file
                    dataset = Dataset_Torch(
                        dir = base_dir,
                        file_name = df_name,
                        seed = SEED,
                        batch_size=128
                    )

                    dataset.split_datasets()
                    dataset.init_dataloaders()

                    dataset_grading = Py_Torch(
                        # parent
                        model=BertForSequenceClassification.from_pretrained(model_name, num_labels=1),
                        dataset=dataset,
                        measurement_settings=Measurement_Settings(
                            dataset_name=dataset["name"],
                            embedding_seperated=False,
                            sentence_embedding_method=None,
                            feature_engenearing_method=None,

                            embedding_model_name=model_name,
                            grading_model=model_name,
                            
                            seed_data_split=42,

                            description = "seplling_corrected",

                            # inform user settings
                            print_regression=True,
                            print_classification=True,
                            
                            # save settings
                            settings_performance_tracking=ADD,
                            save_performance=True
                        ),

                        # child
                        y_column="assigned_points",

                        y_normalized=False, # idd not normalized because measurments are doen on non normalized values!

                        lr = 2e-5,
                        saved_model_dir = f"grading_models/BERT/saved_models/{model_name}/{dataset['name']}",
                        epochs_to_run = 20,

                    )

                    dataset_grading.model_init()
                    dataset_grading.train()

                if df_name == "concatenated_domains" or df_name == "concatenated_datasets":

                    datasets = DATASETS
                    if df_name == "concatenated_domains":
                        datasets = DOMAINS

                    for dataset_name_to_split in datasets:

                        # Now you can do whatever you want with the file
                        dataset = Dataset_Torch(
                            dir = base_dir,
                            file_name = df_name,
                            seed = SEED,
                            batch_size=128,
                            left_out_dataset=dataset_name_to_split
                        )

                        dataset.split_datasets()
                        dataset.init_dataloaders()

                        dataset_grading = Py_Torch(
                            # parent
                            model=BertForSequenceClassification.from_pretrained(model_name, num_labels=1),
                            dataset=dataset,
                            measurement_settings=Measurement_Settings(
                                dataset_name=dataset["name"],
                                embedding_seperated=False,
                                sentence_embedding_method=None,
                                feature_engenearing_method=None,

                                embedding_model_name=model_name,
                                grading_model=model_name,
                                
                                seed_data_split=42,

                                description = "seplling_corrected",

                                # inform user settings
                                print_regression=True,
                                print_classification=True,
                                
                                # save settings
                                settings_performance_tracking=ADD,
                                save_performance=True
                            ),

                            # child
                            y_column="assigned_points",

                            y_normalized=False, # idd not normalized because measurments are doen on non normalized values!

                            lr = 2e-5,
                            saved_model_dir = f"grading_models/BERT/saved_models/{model_name}/{dataset['name']}",
                            epochs_to_run = 20,

                        )

                        dataset_grading.model_init()
                        dataset_grading.train()
