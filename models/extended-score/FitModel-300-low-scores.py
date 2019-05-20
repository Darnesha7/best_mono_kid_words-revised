import pandas as pd
import random
from matplotlib import pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor

# Load representations for training and testing the model
orth_train = pd.read_csv("examples/train_orth.csv", header=None, keep_default_na=False, index_col = 0)
orth_test = pd.read_csv("examples/test_orth.csv", header=None, keep_default_na=False, index_col = 0)
phon_train = pd.read_csv("examples/train_phon.csv", header=None, keep_default_na=False, index_col = 0)
phon_test = pd.read_csv("examples/test_phon.csv", header=None, keep_default_na=False, index_col = 0)

# Define the model architecture and training parameters
M = MLPClassifier(
    hidden_layer_sizes = (100,), 
    activation='logistic',
    solver='lbfgs',
    tol = 1e-8,
    max_iter=3000)

# Train model on the orth to phon mapping
M.fit(orth_train, phon_train)

# Score the model (compute proportion of accurate trials)
train_acc = M.score(orth_train, phon_train)
test_acc = M.score(orth_test, phon_test)

## Compute models fit to random training and test sets of the same sizes
N = 10
orth = pd.concat(orth_train,orth_test)
phon = pd.concat(phon_train,phon_test)
test_acc_r = [0] * N
train_acc_r = [0] * N
for k in range(N):
    # Print a progress indicator
    print(k)
    
    # Select random sets
    n = orth.shape[0]
    ix_train = random.sample(range(n), 300)
    ix_test = [i for i in range(n) if not i in ix_train]

    orth_train_r = orth.iloc[ix_train,:]
    orth_test_r = orth.iloc[ix_test,:]
    phon_train_r = phon.iloc[ix_train,:] 
    phon_test_r = phon.iloc[ix_test,:]

    # Train model on the orth to phon mapping
    M.fit(orth_train, phon_train)

    # Score the model (compute proportion of accurate trials)
    train_acc_r[k] = M.score(orth_train_r, phon_train_r)
    test_acc_r[k] = M.score(orth_test_r, phon_test_r)