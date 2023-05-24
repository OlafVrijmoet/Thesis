
import os

# classes
from run_models.cosine_sililarity.classes.Dataset_Cosine import Dataset_Cosine
from classes.Dataset_Settings import Dataset_Settings

def consine_similarity(dataset_dict):

    dataset = Dataset_Cosine(
        df_name=dataset_dict["dataset_name"],
        model_name=f"cosine_similarity", # used for dir name inside data_saved

        language="english",

        datasets = {
            f"emebeded_words": Dataset_Settings(
                df=None,
                df_name=dataset_dict["abriviation_method"],
                base_dir=f"data/embed_sentences/data/{dataset_dict["embed_model"]}",

                may_run_now=False,
                required=True,
                parquet=True,
                force_run=False
            ),
            f"cosine_similarity": Dataset_Settings(
                df=None,
                df_name="cosine_similarity",
                base_dir=f"data/feature_engenearing/consine_similarity/{dataset_dict["embed_model"]}/{dataset_dict["abriviation_method"]}",

                may_run_now=True,
                required=True,
                parquet=True,
                name_required_dataset="gensim",
                force_run=True
            ),
        },

        columns_to_add = {f"cosine_similarity": {"cosine_similarity": []}},

    )

    # get dataset, process dataset, save dataset
    dataset.get_dataset()
    dataset.process_dataset()
    dataset.add_columns()
    dataset.save()

