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


def plot_confusion_matrix(Y_test, prediction, clf,X_train):
    
    cm = confusion_matrix(Y_test, prediction, labels=clf.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=clf.classes_)
    
    disp.plot()
    disp.ax_.set_title('{}+{}'.format(clf,X_train.shape))
    plt.show()


def train_test(X_train,Y_train, X_test,Y_test,clf, show_time =False):
    
    if show_time == True:
        print("The experiment is %s \n" % (clf))
        print("The shape of X_train is {} \n".format(X_train.shape))
        
        #start training classifier
        start_time = time.time()
        clf.fit(X_train,Y_train)
        train_time = time.time()-start_time
        print("The train time is --- %.8f seconds ---" % (train_time))
        
        #testing the fitted model
        start_time_test = time.time()
        prediction = clf.predict(X_test)
        test_time = time.time()-start_time_test
        print("The test time is --- %.8f seconds ---" % (test_time))
        
        #compute error
        error = sum(prediction != Y_test)

        return error, prediction, train_time, test_time
        
    else:           
        clf.fit(X_train,Y_train)
        
        start_time_test = time.time()
        prediction = clf.predict(X_test)


        error = sum(prediction != Y_test)
        
        return error, prediction