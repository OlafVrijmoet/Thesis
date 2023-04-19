
# data processing
from data.raw.all_to_csv import all_to_csv
from data.standardized.standardize import standardize
from data.processed.process_text import process_text

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
    run=True
)
