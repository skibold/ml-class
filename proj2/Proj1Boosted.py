from AdaBoost import *
from sys import argv

x_train = argv[1]
y_train = argv[2]
x_test = argv[3]
y_test = argv[4]
T = int(argv[5])

feats = np.loadtxt(x_train, delimiter=',',  dtype=np.float).transpose()
labels = np.loadtxt(y_train, delimiter=',', dtype=np.int)
ab_labels = np.array([1 if i==1 else -1 for i in labels])
abs = []
for i in range(30):
	abs.append(AdaBoost(T, feats[i], ab_labels))
	abs[-1].train()
 
x_test = np.loadtxt(x_test, delimiter=',', dtype=np.float)
y_test = np.loadtxt(y_test, delimiter=',', dtype=np.float)
ab_y_test = np.array([1 if i==1 else -1 for i in y_test])
predictions = []
for i in range(x_test.shape[0]):
	predictions.append(np.sign(np.sum([abs[j].eval(x_test[i][j]) for j in range(x_test.shape[1])])))
	
print("accuracy = {}".format(len(np.where(predictions == ab_y_test)[0]) / len(ab_y_test)))
