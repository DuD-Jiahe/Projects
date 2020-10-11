from sklearn.svm import SVC
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
import time

#input training set
train = read_csv(r"new_data.csv")
train = train.values
scaler= MinMaxScaler((0,1))
train=scaler.fit_transform(train)
y_train=train[:,1]
X_train=train[:,[5, 6, 7, 8, 10, 12, 13, 15, 16, 17, 18, 19]]

#input testing set
test = read_csv(r"test_set.csv")
test = test.values
scaler= MinMaxScaler((0,1))
test=scaler.fit_transform(test)
y_test=test[:,1]
X_test=test[:,[5, 6, 7, 8, 10, 12, 13, 15, 16, 17, 18, 19]]

X_test=test[:,[5, 6, 7, 8, 10, 12, 13, 15, 16, 17, 18, 19]]#4:19]
#print(data_test.shape)

#start time
start=time.time()

#Support Vector Machine
kernel_range = ['linear']#'rbf',
#gamma_range = [0.1]#0.1,0.01
probability_range=[True]
param_svm = dict(kernel=kernel_range, probability=probability_range) #gamma=gamma_range,
clf_svm = GridSearchCV(SVC(), param_svm)
clf_svm.fit(X_train, y_train)
print(clf_svm.best_params_)
print(clf_svm.best_estimator_)


print('svm----------')
#for testing data
svm_pred_test = clf_svm.predict(X_test)
#accuracy_score
score_svm = accuracy_score(y_test, svm_pred_test)
print('acc_test: ',score_svm)
#precison
pre_svm=average_precision_score(y_test, svm_pred_test)
print('pre_test: ',pre_svm)
#recall
rec_svm=recall_score(y_test, svm_pred_test)
print('recall_test: ',rec_svm)
#f1
f1_svm=f1_score(y_test, svm_pred_test)
print('f1_svm: ',f1_svm)
#roc_auc
auc_svm=roc_auc_score(y_test, svm_pred_test)
print('roc_test: ',auc_svm)

#end time
end=time.time()
print('running time: %s seconds' %(end-start))