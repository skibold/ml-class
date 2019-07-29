import numpy as np
from sys import argv
from svm import train as svm_train
from svm import test as svm_test
from gnb import train as gnb_train
from gnb import test as gnb_test
from Proj1Boosted import train as ada_train
from Proj1Boosted import test as ada_test

def transform_labels(labels):
	return np.array([1 if i==1 else -1 for i in labels])
	
def train(xtrain, ytrain):
	a = ada_train(xtrain,ytrain,10)
	g = gnb_train(xtrain,ytrain) 
	s = svm_train(xtrain,ytrain)
	return a,g,s

def test(a,g,s, xtest):
	ap = ada_test(a,xtest)
	gp = transform_labels(gnb_test(g,xtest))
	sp = svm_test(s,xtest)
	return ap, gp, sp
	
if __name__ == "__main__":
	train_x_location = argv[1] #"x_train.csv"
	train_y_location = argv[2] #"y_train.csv""
	test_x_location = argv[3] #"x_test.csv"
	test_y_location = argv[4] #"y_test.csv"
	#log = argv[5]

	#print("Reading training data")
	xtrain = np.loadtxt(train_x_location, dtype=np.float, delimiter=",")
	ytrain = np.loadtxt(train_y_location, dtype=np.int, delimiter=",")
	xtest = np.loadtxt(test_x_location, dtype=np.float, delimiter=",")
	ytest = transform_labels(np.loadtxt(test_y_location, dtype=np.int, delimiter=","))

	a,g,s = train(xtrain, ytrain)
	apred, gpred, spred = test(a,g,s,xtest)
	#print(apred)
	majority = [np.sign(apred[i]+gpred[i]+spred[i]) for i in range(len(ytest))]
	aacc = (ytest == apred).sum()/len(ytest)
	gacc = (ytest == gpred).sum()/len(ytest)
	sacc = (ytest == spred).sum()/len(ytest)
	macc = (ytest == majority).sum()/len(ytest)
	#print(apred, gpred, spred, majority)
	print(aacc, gacc, sacc, macc)