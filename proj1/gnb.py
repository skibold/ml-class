import numpy as np
from sklearn.naive_bayes import GaussianNB
from sys import argv

train_x_location = argv[1] #"x_train.csv"
train_y_location = argv[2] #"y_train.csv"
log = argv[3]
test_x_location = "x_test.csv"
test_y_location = "y_test.csv"

print("Reading training data")
xtrain = np.loadtxt(train_x_location, dtype=np.float, delimiter=",")
ytrain = np.loadtxt(train_y_location, dtype=np.int, delimiter=",")
xtest = np.loadtxt(test_x_location, dtype=np.float, delimiter=",")
ytest = np.loadtxt(test_y_location, dtype=np.int, delimiter=",")

gnb = GaussianNB()
ypred = gnb.fit(xtrain, ytrain).predict(xtest)
print((ytest == ypred).sum()/len(ytest))
with open(log, 'a') as fout:
	fout.write("{}\n".format((ytest == ypred).sum()/len(ytest)))