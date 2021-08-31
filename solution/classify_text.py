## Imports
import pandas as pd


## Data
train = pd.read_csv('../data/train_set.csv')
test = pd.read_csv('../data/test_set.csv')
print(f"Train {train.shape}")
print(f"Test {test.shape}")
print(train.head())

## preprocessing
