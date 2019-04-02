import sys
import csv
import os
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score 
from sklearn.metrics import mean_squared_error


# 使用csv.reader读取数据
def ReadData(file):
    with open(file, 'r') as f:
        newfile = csv.reader(f, delimiter = ';')
        dataList = np.array(list(newfile))
    return dataList

def getXY(dataList, label):
    features_name = dataList[0]
    features_value = dataList[1:]
    # 归一化所有数据
    cols_name = list(features_name)
    label_index = cols_name.index(label)
    # 提取label列的数据
    Y = [y for y in features_value[:,label_index]]
    # 提取feature的数据
    X = np.delete(features_value, label_index, axis = 1)

    # 换一种归一化全体的思路
    min_max_scaler = MinMaxScaler()
    X = min_max_scaler.fit_transform(X)
    return X, Y



# OneHotEncoder，将第4列、第5列、12列、16、15列化为OneHotEncoder
from sklearn.preprocessing import OneHotEncoder
# 提取第4、5、12、15、16列--3,4,11,14,15--2,3，10，13，14
def OneHotEncoding(DataList, l):
    for k in l:
        X2 = [100 * x for x in DataList[:,k]]
        X2 = np.array(X2).reshape(-1,1)
        enc = OneHotEncoder(sparse=False)
        X2 = enc.fit_transform(X2)
        #print(X2)
        DataList = np.concatenate((DataList, X2), axis=1)
    DataList = np.delete(DataList, l, axis=1)
    return DataList
    #print(X_train_class.shape)

'''
去掉”Reason for absence“之后，对Day_of_the_week Disciplinary_failure 
Social drinker，Social smoker，即第2，10，13，14列 执行 OneHotEncoder操作
'''


# 用SVM进行分类和回归
# 用SVM分类：linear > rbf > sigmoid > poly
# 用SVM回归：linear = rbf = sigmoid = poly
def svmClass(X_train_class, Y_train_class, X_val_class, Y_val_class):
	l = ['linear', 'rbf', 'sigmoid', 'poly']
	micro_f1_max = 0
	micro_f1_model = ""
	for i in l:
	    clf = SVC(kernel=i)
	    clf.fit(X_train_class, Y_train_class)
	    clf_predict = clf.predict(X_val_class)
	    from sklearn.metrics import f1_score 
	    micro_f1 = f1_score(Y_val_class, clf_predict, average='micro')
	    if micro_f1 > micro_f1_max:
	        micro_f1_max = micro_f1
	        micro_f1_model = "SVM_SVC" + i
	return micro_f1_model, micro_f1_max

def svmRegression(X_train_reg, Y_train_reg, X_val_reg, Y_val_reg):
	Y_val_reg = list(map(float, Y_val_reg))
	mse_min = 500
	mse_model = ""
	# 用SVR回归
	l = ['linear', 'rbf', 'sigmoid', 'poly']
	from sklearn.svm import SVR
	for i in l:
	    clr = SVR(i)
	    clr.fit(X_train_reg, Y_train_reg)
	    clr_predict = clr.predict(X_val_reg)
	    from sklearn.metrics import mean_squared_error
	    mse_reg = mean_squared_error(Y_val_reg, clr_predict)
	    if mse_reg < mse_min:
	        mse_min = mse_reg
	        mse_model = "SVM_SVR" + i
	return mse_model, mse_min

# 用MLP进行分类和回归
# 用MLPClassifier参数: adam > lbfgs > sgd
# activation: identity = relu = ranh > logistic
# 用MLPRegressor进行回归分析： lbfgs < adam < sgd
def mlpClass(X_train_class, Y_train_class, X_val_class, Y_val_class):
	clf = MLPClassifier(solver='adam', activation='tanh', alpha=1e-5,hidden_layer_sizes=(100, 30), random_state=1)
	clf.fit(X_train_class,Y_train_class)
	clf_predict = clf.predict(X_val_class)

	micro_f1 = f1_score(Y_val_class, clf_predict, average='micro')
	micro_f1_model = "MLPClassifier"
	return micro_f1_model, micro_f1

def mlpRegression(X_train_reg, Y_train_reg, X_val_reg, Y_val_reg):
	X_train_reg0 = [list(map(float, x)) for x in X_train_reg]
	Y_train_reg0 = list(map(float, Y_train_reg))
	Y_val_reg = list(map(float, Y_val_reg))

	from sklearn.neural_network import MLPRegressor
	clr = MLPRegressor(solver='sgd', alpha=1e-5,hidden_layer_sizes=(100, 50), random_state=1)
	clr.fit(X_train_reg0, Y_train_reg0)
	clr_predict = clr.predict(X_val_reg)

	from sklearn.metrics import mean_squared_error
	mse_reg = mean_squared_error(Y_val_reg, clr_predict)
	mse_model = "MLPRegression"
	return mse_model, mse_reg

# 用Linear Model进行分类和回归
def linearClass(X_train_class, Y_train_class, X_val_class, Y_val_class):
	from sklearn import linear_model
	clf = linear_model.SGDClassifier(max_iter=1000, tol=1e-3)
	clf.fit(X_train_class,Y_train_class)
	clf_predict = clf.predict(X_val_class)

	micro_f1 = f1_score(Y_val_class, clf_predict, average='micro')
	micro_f1_model = "Linear_Model_SGDClassifier"
	return micro_f1_model, micro_f1
'''
Linear Model回归
lbfgs = liblinear = sag = saga
'''
def linearRegression(X_train_reg, Y_train_reg, X_val_reg, Y_val_reg):
	from sklearn.linear_model import LogisticRegression
	clf = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial')

	X_train_reg0 = [list(map(float, x)) for x in X_train_reg]
	Y_train_reg0 = list(map(float, Y_train_reg))

	clf.fit(X_train_reg0, Y_train_reg0)
	clf_predict = clf.predict(X_val_reg)

	
	Y_val_reg = list(map(float, Y_val_reg))

	mse_reg = mean_squared_error(Y_val_reg, clf_predict)
	mse_model = "Linear_Model_LogisticRegression"
	return mse_model, mse_reg

if __name__ == "__main__":
	file1 = sys.argv[1]
	file2 = sys.argv[2]

	cwd = os.getcwd()

	new_train_csv = os.path.join(cwd, file1)
	new_val_csv = os.path.join(cwd, file2)

	trainData = ReadData(new_train_csv)
	valData = ReadData(new_val_csv)

	X_train_class, Y_train_class = getXY(trainData, 'Reason for absence')
	X_val_class, Y_val_class = getXY(valData, 'Reason for absence')
	X_train_reg, Y_train_reg = getXY(trainData, 'Absenteeism time in hours')
	X_val_reg, Y_val_reg = getXY(valData, 'Absenteeism time in hours')

	l1 = [2, 10, 13, 14]
	l2 = [3, 11, 14, 15]
	X_train_class = OneHotEncoding(X_train_class, l1)
	X_val_class = OneHotEncoding(X_val_class, l1)
	X_train_reg = OneHotEncoding(X_train_reg, l2)
	X_val_reg = OneHotEncoding(X_val_reg, l2)

	micro_f1_max = 0
	micro_f1_model = ""
	mse_min = 500
	mse_model = ""

	mode1, f1_svm = svmClass(X_train_class, Y_train_class, X_val_class, Y_val_class)
	model2, f1_mlp = mlpClass(X_train_class, Y_train_class, X_val_class, Y_val_class)
	model3, f1_linear = linearClass(X_train_class, Y_train_class, X_val_class, Y_val_class)

	micro_f1_max = max(f1_svm, f1_mlp, f1_linear)

	model_1, mse_svm = svmRegression(X_train_reg, Y_train_reg, X_val_reg, Y_val_reg)
	model_2, mse_mlp = mlpRegression(X_train_reg, Y_train_reg, X_val_reg, Y_val_reg)
	model_3, mse_linear = linearRegression(X_train_reg, Y_train_reg, X_val_reg, Y_val_reg)

	mse_min = min(mse_mlp, mse_svm, mse_linear)
	
	print("Micro-average F1 of classification: ")
	print(micro_f1_max*100,"%")
	print("Mean squared error of regression:")
	print(mse_min)
