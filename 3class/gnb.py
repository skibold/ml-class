import numpy as np
from sklearn.naive_bayes import GaussianNB
from sys import argv
	
def train(xtrain, ytrain):
	gnb = GaussianNB()
	gnb.fit(xtrain, ytrain)
	return gnb
	
def test(gnb, xtest):
	return gnb.predict(xtest)
	
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
	ytest = np.loadtxt(test_y_location, dtype=np.int, delimiter=",")

	gnb = train(xtrain, ytrain)
	ypred = test(gnb, xtest)
	print((ytest == ypred).sum()/len(ytest))
	#with open(log, 'a') as fout:
	#	fout.write("{}\n".format((ytest == ypred).sum()/len(ytest)))