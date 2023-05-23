
# services
from services.get_dfs import get_dfs

# contants
from data.splits.constants import SPLITS

def embed_words():

    datasets = {}

    # get dfs from former phase
    datasets = get_dfs(dict=datasets, dir=SPLITS)
    
    