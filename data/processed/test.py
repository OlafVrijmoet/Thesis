
#%%
import pandas as pd
import nltk

# create two identical DataFrames
df1 = pd.DataFrame({'student_answer': ['The quick brown fox', 'Jumped over the lazy dog']})
df2 = df1.copy()

# tokenize the 'student_answer' column in df1
for index, row in df1.iterrows():
    tokens = nltk.word_tokenize(row['student_answer'])
    df2.loc[index, 'student_answer'] = tokens

# print the results
print(df1)
print(df2)

# %%
