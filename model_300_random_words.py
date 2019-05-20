import pandas as pd
import random
from matplotlib import pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor

orth = pd.read_csv ("3k/orth.csv", header=None)
phon = pd.read_csv ("3k/phon.csv", header=None)

# Select 300 random words
n = orth.shape[0]
ix_train = random.sample (range(n), 300)
ix_test = [i for i in range(n) if not i in ix_train]

orth_train_random = orth.iloc[ix_train,:]
orth_test_random = orth.iloc[ix_test,:]
phon_train_random = phon.iloc[ix_train,:]
phon_test_random = phon.iloc[ix_test,:]

# Verify selections
print([
    orth_train_random.shape[0],
    phon_train_random.shape[0],
    orth_test_random.shape[0],
    phon_test_random.shape[0]
])

# Fit to 300 random words
M.fit(orth_train_random, phon_train_random)
M.score(orth_train, phon_train)
M.score(orth_test, phon_test)
