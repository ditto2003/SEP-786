import scipy.io as sio

import matplotlib.pyplot as plt
import numpy as np
import sklearn.discriminant_analysis
from sklearn import svm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay



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

# Load data
path_good = 'data//baseline_20220915_sv.mat'
path_bad= 'data//fault7_20220915_sv.mat'

mat_contents_good = Load_mat_single(path_good)
good_data = mat_to_array(mat_contents_good)

mat_contents_bad = Load_mat_single(path_bad)
bad_data = mat_to_array(mat_contents_bad)


# constuct the data

X = np.concatenate((good_data,bad_data))

n_sample = good_data.shape[0]
n_feature  = good_data.shape[1]

Y = np.zeros(n_sample)
Y = np.concatenate((Y, np.ones(n_sample)))

# PCA


X_mean = X - np.mean(X)


C_x = np.dot(X_mean.T,X_mean)

SS_pca,V = np.linalg.eig(C_x)

sortIndex = np.flip(np.argsort(SS_pca))

dimension = good_data.shape[1]
VSorted = np.empty((dimension,0))

for i in range(dimension):
    VSorted = np.append(VSorted, V[:,sortIndex[i]].reshape(dimension,1), axis=1)


classificationError_lda_pca = np.zeros(4,)
classificationError_svm_pca = np.zeros(4,)

Score_Sorted = np.dot(X,VSorted)

train_index  = np.arange(0,n_sample*0.75).astype(int).tolist()+np.arange(n_sample,n_sample+n_sample*0.75).astype(int).tolist()
test_index = np.arange(n_sample*0.75,n_sample).astype(int).tolist()+np.arange(n_sample+n_sample*0.75,n_sample+n_sample).astype(int).tolist()
X_train  = Score_Sorted[train_index,:]
X_test = Score_Sorted[test_index,:]


Y_train = Y[train_index]
Y_test = Y[test_index]






for numDims in range(5,9): 
    
    # reconstruction
    Score_Reduced = X_train[:,0:numDims]
    print('Reduced score shape is ', Score_Reduced.shape)
    # VReduced = VSorted[:,0:numDims]
    # print('Reduced right-singular vectors shape is {}\n'.format(VReduced.shape))
    

    lda_pca = sklearn.discriminant_analysis.LinearDiscriminantAnalysis()

    lda_pca.fit(Score_Reduced,Y_train)
    #testing
    prediction_lda_pca = lda_pca.predict(X_test[:,0:numDims])
    classificationError_lda_pca[numDims-5] = sum(prediction_lda_pca != Y_test)

    print("Confusion matrix for lda with PCA")

    cm_lda_pca = confusion_matrix(Y_test, prediction_lda_pca, labels=lda_pca.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_lda_pca,display_labels=lda_pca.classes_)
                            
    disp.plot()

    plt.show()



# for numDims in range(5,9): 
    
#     # reconstruction
#     Score_Reduced = X_train[:,0:numDims]
#     print('Reduced score shape is ', Score_Reduced.shape)
#     # VReduced = VSorted[:,0:numDims]
#     # print('Reduced right-singular vectors shape is {}\n'.format(VReduced.shape))
#     clf_svm_pca = svm.SVC(kernel = 'linear')
#     clf_svm_pca.fit(Score_Reduced,Y_train)
#     prediction_svm_pca = clf_svm_pca.predict(X_test[:,0:numDims])
#     classificationError_svm_pca[numDims-5] = sum(prediction_svm_pca != Y_test)


#     print("Confusion matrix for SCM with PCA")

#     cm_svm_pca = confusion_matrix(Y_test, prediction_svm_pca , labels=clf_svm_pca.classes_)
#     disp = ConfusionMatrixDisplay(confusion_matrix=cm_svm_pca,display_labels=lda_pca.classes_)
                            
#     disp.plot()

#     plt.show()
