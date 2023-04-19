
# services
from services.run_phase import run_phase

# constants
from constants import *

def main():

    # raw data
    run_phase(RAW_PHASE)

    # standardize data
    run_phase(STANDARDIZE_PHASE)


if __name__ == "__main__":

    main()

