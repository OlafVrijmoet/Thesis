
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

    # Function to generate new rows based on randomly selected samples within each group
    def generate_rows(self, group):
        shots = self.shots
        sample_with_replacement = False

        # Adjust sample size if group size is less than `shots`
        if len(group) < self.shots:
            sample_with_replacement = True

        # Randomly sample indices from group
        indices = group.sample(shots, random_state=self.seed, replace=sample_with_replacement).index

        result = {}
        for i, index in enumerate(indices):
            row = group.loc[index]
            # Store student's answer, reference answer, and assigned points in the result dictionary
            result[f'student_answer_{i+1}'] = row['student_answer']
            result[f'reference_answer_{i+1}'] = row['reference_answer']
            result[f'assigned_points_{i+1}'] = row['assigned_points']

        return pd.Series(result)

    # Function to split dataset and generate new rows
    def split_datasets(self):
        # Call parent class's split_datasets method
        super().split_datasets()

        # Concatenate train and test
        combined_df = pd.concat([self.train, self.test])

        # Apply `generate_rows` function to each group and reset the index for each dataset
        train_grouped_with_shots = self.train.groupby('question_id').apply(self.generate_rows).reset_index()
        combined_grouped_with_shots = combined_df.groupby('question_id').apply(self.generate_rows).reset_index()

        # Merge new rows with the original datasets
        self.train = pd.merge(self.train, train_grouped_with_shots, on='question_id', how='left')
        self.test = pd.merge(self.test, combined_grouped_with_shots, on='question_id', how='left')
        self.validation = pd.merge(self.validation, combined_grouped_with_shots, on='question_id', how='left')

        # Drop rows with NaN values in 'student_answer_1' column if it exists in the dataframe. 0 shots will not have any examples to drop
        if 'student_answer_1' in self.train.columns:
            self.train.dropna(subset=['student_answer_1'], inplace=True)
        if 'student_answer_1' in self.test.columns:
            self.test.dropna(subset=['student_answer_1'], inplace=True)
        if 'student_answer_1' in self.validation.columns:
            self.validation.dropna(subset=['student_answer_1'], inplace=True)
