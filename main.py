
# services
from services.run_phase import run_phase

# constants
from constants import *

def main():

    # raw data
    run_phase(RAW_PHASE)

    # standardize data
    run_phase(STANDARDIZE_PHASE)

    # # process text - !!!OUTDATED!!!
    # run_phase(PROCESS_TEXT_PHASE)

    # create datasets splits, example: based on domains
    run_phase(SPLIT_DATA)

    # # embed words
    # run_phase(EMBED_WORDS)

    # run Gensim models
    run_phase(GENSIM)

    # run Embedding models
    run_phase(EMBEDDING)

    # add cosine similarity
    run_phase(COSINE_SIMILARITY)

    # run experiements where embedding and classification are seperate
    run_phase(CLASSIFICATION_EXPERIMENTS)

if __name__ == "__main__":

    main()
