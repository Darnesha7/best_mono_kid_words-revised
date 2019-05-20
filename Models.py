import pandas as pd
import random
from matplotlib import pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
orth_train = pd.read_csv("3k/orth_aoa300_reps_train.csv", header=None, keep_default_na=False, index_col = 0)
orth_test = pd.read_csv("3k/orth_aoa300_reps_test.csv", header=None, keep_default_na=False, index_col = 0)
phon_train = pd.read_csv("3k/phon_aoa300_reps_train.csv", header=None, keep_default_na=False, index_col = 0)
phon_test = pd.read_csv("3k/phon_aoa300_reps_test.csv", header=None, keep_default_na=False, index_col = 0)
orth = pd.read_csv ("3k/orth.csv", header=None)
phon = pd.read_csv ("3k/phon.csv", header=None)
low_3k_score_words = pd.read_csv ("low_3k_score_words.csv", header=None, keep_default_na=False, index_col = 0)

M = MLPClassifier(
    hidden_layer_sizes = (100,), 
    activation='logistic',
    solver='lbfgs',
    tol = 1e-8,
    max_iter=3000)

n = orth.shape[0]
ix_train = random.sample (range(n), 300)
ix_test = [i for i in range(n) if not i in ix_train]

orth_train_random = orth.iloc[ix_train,:]
orth_test_random = orth.iloc[ix_test,:]
phon_train_random = phon.iloc[ix_train,:] 
phon_test_random = phon.iloc[ix_test,:]

print([
    orth_train_random.shape[0],
    phon_train_random.shape[0],
    orth_test_random.shape[0],
    phon_test_random.shape[0]
])

M.fit(orth_train_random, phon_train_random)
M.score(orth_train, phon_train)
M.score(orth_test, phon_test)
