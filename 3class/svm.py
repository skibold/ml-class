import numpy as np
from sklearn.svm import SVC
from sys import argv

def transform_labels(labels):
	return np.array([1 if i==1 else -1 for i in labels])

def train(xtrain, ytrain):
	clf = SVC(kernel='linear') #, degree=2)
	clf.fit(xtrain, transform_labels(ytrain))
	return clf
	
def test(clf, xtest):
	return clf.predict(xtest)
	
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

	clf = train(xtrain, ytrain)
	ypred = test(clf, xtest)
	print((transform_labels(ytest) == ypred).sum()/len(ytest))