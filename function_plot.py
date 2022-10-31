import scipy.io as sio

import matplotlib.pyplot as plt
import numpy as np
import sklearn.discriminant_analysis
from sklearn import svm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import time

def Load_mat_single(data_path):

    mat_contents = sio.loadmat(data_path)

    for i, key in enumerate(mat_contents):
        print(i, key)

    return mat_contents


def mat_to_array(mat_contents):
    mat_name = []
    mat_data = []

    for i, (k, v) in enumerate(mat_contents.items()):
        mat_name.append(k)
        mat_data.append(v)

    vibration_signal_all = np.array(mat_data[3])
    return vibration_signal_all


def plot_confusion_matrix(Y_test, prediction, clf):
    
    cm = confusion_matrix(Y_test, prediction, labels=clf.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=clf.classes_)
                            
    disp.plot()

    plt.show()


def train_test(X_train,Y_train, X_test,Y_test,clf):
    start_time = time.time()



 
    clf.fit(X_train,Y_train)
    print(f"The experiment is %s \n" % (clf))
    print("The shape of X_train is {} \n".format(X_train.shape))
    print("The train time is --- %s seconds ---" % (time.time() - start_time))
    
    start_time_test = time.time()
    prediction = clf.predict(X_test)
    print("The test time is --- %s seconds ---" % (time.time() - start_time_test))


    error = sum(prediction != Y_test)

    return error, prediction