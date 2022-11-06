"""System packages that comparison functions used"""
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import time

def Load_mat_single(data_path):
    """Load the .mat data"""
    mat_contents = sio.loadmat(data_path)
    for i, key in enumerate(mat_contents):
        print(i, key)
    return mat_contents

def mat_to_array(mat_contents):
    """Convert the .mat data into Numpy array"""
    mat_name = []
    mat_data = []
    for i, (k, v) in enumerate(mat_contents.items()):
        mat_name.append(k)
        mat_data.append(v)
    vibration_signal_all = np.array(mat_data[3])
    return vibration_signal_all


<<<<<<< HEAD
def plot_confusion_matrix(Y_test, prediction, clf,X_train):
    """Plot confusion matrix"""
=======
def plot_confusion_matrix(Y_test, prediction, clf, X_test, FE_type):
    
>>>>>>> 72e107a6bcf06cde99b77796d9d444dac725cf62
    cm = confusion_matrix(Y_test, prediction, labels=clf.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=clf.classes_)
    
    disp.plot()
    disp.ax_.set_title('{}+{}+{}'.format(FE_type,clf,X_test.shape))
    plt.show()


def train_test(X_train,Y_train, X_test,Y_test,clf, show_time =False):
    """Train the data with AI model and display the execution time"""
    # Enable time flag
    if show_time == True:
        print("The experiment is %s \n" % (clf))
        print("The shape of X_train is {} \n".format(X_train.shape))
<<<<<<< HEAD
        # Time start
        start_time = time.time()
        # Train the data
        clf.fit(X_train,Y_train)
        print("The train time is --- %.8f seconds ---" % (time.time() - start_time))
        # Time end
=======
        
        #start training classifier
        start_time = time.time()
        clf.fit(X_train,Y_train)
        train_time = time.time()-start_time
        print("The train time is --- %.8f seconds ---" % (train_time))
        
        #testing the fitted model
>>>>>>> 72e107a6bcf06cde99b77796d9d444dac725cf62
        start_time_test = time.time()
        # Predict the test set
        prediction = clf.predict(X_test)
<<<<<<< HEAD
        print("The test time is --- %.8f seconds ---" % (time.time() - start_time_test))
        # Statistical ERROR
        error = sum(prediction != Y_test)
    # Disable time flag
    else:
        # Train the data
=======
        test_time = time.time()-start_time_test
        print("The test time is --- %.8f seconds ---" % (test_time))
        
        #compute error
        error = sum(prediction != Y_test)

        return error, prediction, train_time, test_time
        
    else:           
>>>>>>> 72e107a6bcf06cde99b77796d9d444dac725cf62
        clf.fit(X_train,Y_train)
        # Time start
        start_time_test = time.time()
        # Predict the test set
        prediction = clf.predict(X_test)
<<<<<<< HEAD
        # Statistical ERROR
=======


>>>>>>> 72e107a6bcf06cde99b77796d9d444dac725cf62
        error = sum(prediction != Y_test)
        
        return error, prediction