
class Dataset:

    def __init__(self,
                 
        dataset_dir,

        seed


    ) -> None:
        

        
        # split seed
        self.seed = seed
        
        # original dataset
        self.dataset = self.get_data(dataset_dir)

        # places to save the splits
        self.train_df = None
        self.test_df = None
        self.validation_df = None

    # get data using dataset_dir
    def get_data(dir):

        
