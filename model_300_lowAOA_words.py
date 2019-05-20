import pandas as pd
import numpy as np
import random
from matplotlib import pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor

word_train = pd.read_csv('3k/orthaoa300training.csv', header=0, index_col = 0, keep_default_na = False)
word_test = pd.read_csv('3k/orthaoa300testing.csv', header=0, index_col = 0, keep_default_na = False)
orth_train = pd.read_csv("3k/orth_aoa300_reps_train.csv", header=None, keep_default_na=False, index_col = 0)
orth_test = pd.read_csv("3k/orth_aoa300_reps_test.csv", header=None, keep_default_na=False, index_col = 0)
phon_train = pd.read_csv("3k/phon_aoa300_reps_train.csv", header=None, keep_default_na=False, index_col = 0)
phon_test = pd.read_csv("3k/phon_aoa300_reps_test.csv", header=None, keep_default_na=False, index_col = 0)

# Define the model
M = MLPClassifier(
    hidden_layer_sizes = (100,), 
    activation='logistic',
    solver='lbfgs',
    tol = 1e-8,
    max_iter=3000,
    warm_start = True)

# Fit to the lowest AOA words
M.fit(orth_train, phon_train)
Y = pd.DataFrame(data = np.zeros((31,2)), columns = ('train_acc','test_acc'))
Y.train_acc[0] = M.score(orth_train, phon_train)
Y.test_acc[0] = M.score(orth_test, phon_test)

M.max_iter = 1
for i in range(30):
    M.fit(orth_test, phon_test)
    Y.train_acc[i+1] = M.score(orth_train, phon_train)
    Y.test_acc[i+1] = M.score(orth_test, phon_test)

# Check predictions for each test item
phon_test_pred = M.predict_proba(orth_test)

X = pd.DataFrame(data = np.zeros((word_test.shape[0],2)), index = word_test.word, columns = ('SSE','units_wrong'))
for i,(x,y) in enumerate(zip(phon_test.to_numpy(),phon_test_pred)):
    X.SSE[i] = sum((x-y)**2)
    X.units_wrong[i] = sum((x-y)>0.5)

print(X[X.units_wrong > 0].sort_values(by = 'SSE', ascending = False))
print(Y)
