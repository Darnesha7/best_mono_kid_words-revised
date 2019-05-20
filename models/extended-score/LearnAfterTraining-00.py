import pandas as pd
import random
from matplotlib import pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor

# Load representations for training and testing a model based on the 300 "child friendly" words.
orth_train = pd.read_csv("models/extended-score/examples/train_orth.csv", header=None, keep_default_na=False, index_col = 0)
orth_test = pd.read_csv("models/extended-score/examples/test_orth.csv", header=None, keep_default_na=False, index_col = 0)
phon_train = pd.read_csv("models/extended-score/examples/train_phon.csv", header=None, keep_default_na=False, index_col = 0)
phon_test = pd.read_csv("models/extended-score/examples/test_phon.csv", header=None, keep_default_na=False, index_col = 0)


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

# Provide new experiences, and see how fast the model learns to perform
for x in phon_train.columns:
    try:
        tmp = phon_train[x].astype(np.float64)
    except ValueError:
        print(x)