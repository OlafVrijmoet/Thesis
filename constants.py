
# data processing
from data.raw.all_to_csv import all_to_csv
from data.standardized.standardize import standardize
from data.processed.process_text import process_text
from data.split.split_data import split_data
from word_embedding.embed_words import embed_words

# classes
from classes.Phase_Settings import Phase_Settings

# what to run
RAW_PHASE = Phase_Settings(
    name="Raw", 
    function=all_to_csv,
    run=False
)

STANDARDIZE_PHASE = Phase_Settings(
    name="Standardize", 
    function=standardize,
    run=False
)

PROCESS_TEXT_PHASE = Phase_Settings(
    name="Process text", 
    function=process_text,
    run=False
)

SPLIT_DATA = Phase_Settings(
    name="Split data", 
    function=split_data,
    run=True
)

EMBED_WORDS = Phase_Settings(
    name="Embed words", 
    function=embed_words,
    run=False
)

# paths
# LEMMITIZED = "data/lemmitized_data"
# RAW = "data/raw_data"
# STEMMED = "data/stemmed_data"

# LEMMITIZED_DATASETS = f"{LEMMITIZED}/datasets"
# LEMMITIZED_DOMAIN = f"{LEMMITIZED}/domain"

# RAW_DATASETS = f"{RAW}/datasets"
# RAW_DOMAIN = f"{RAW}/domain"

# STEMMED_DATASETS = f"{STEMMED}/datasets"
# STEMMED_DOMAIN = f"{STEMMED}/domain"
