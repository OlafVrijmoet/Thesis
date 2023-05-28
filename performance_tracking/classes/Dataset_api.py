
from sklearn.model_selection import train_test_split

import pandas as pd

# classes
from performance_tracking.classes.Dataset import Dataset

class Dataset_api(Dataset):

    def __init__(self,
                 dir,
                 file_name,
                 seed,
                 shots  # New parameter specifying number of examples per question
                 ) -> None:

        # Call the parent class's constructor
        super().__init__(dir, file_name, seed)

        self.name = file_name

        # Store the number of examples per question
        self.shots = shots

    # Define the generate_rows function within the class
    def generate_rows(self, group):

        shots = self.shots

        # Make sure there are enough rows to sample
        if len(group) < self.shots:
            print(f"Not enough shots: {self.name}")
            shots = len(group)
            # raise ValueError("The group must contain at least as many rows as the number of shots.")

        indices = group.sample(shots, random_state=self.seed).index

        result = {}
        for i, index in enumerate(indices):
            row = group.loc[index]
            result[f'student_answer_{i+1}'] = row['student_answer']
            result[f'reference_answer_{i+1}'] = row['reference_answer']
            result[f'assigned_points_{i+1}'] = row['assigned_points']

        return pd.Series(result)

    # Modify the split_datasets method to also generate rows
    def split_datasets(self):
        # Call the parent class's split_datasets method
        super().split_datasets()

        # Concatenate train and test
        combined_df = pd.concat([self.train, self.test])

        # Generate rows for the combined DataFrame
        grouped_with_shots = combined_df.groupby('question_id').apply(self.generate_rows).reset_index()

        # Merge the generated rows with the original datasets
        self.train = pd.merge(self.train, grouped_with_shots, on='question_id', how='left')
        self.test = pd.merge(self.test, grouped_with_shots, on='question_id', how='left')
        self.validation = pd.merge(self.validation, grouped_with_shots, on='question_id', how='left')
