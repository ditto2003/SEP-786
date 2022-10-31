import scipy.io as sio

import matplotlib.pyplot as plt
import numpy as np
import sklearn.discriminant_analysis
from sklearn import svm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from function_plot import Load_mat_single
from function_plot import mat_to_array
from function_plot import plot_confusion_matrix
from function_plot import train_test
import time




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

print("Start PCA process...")


X_mean = X - np.mean(X)


C_x = np.dot(X_mean.T,X_mean)

SS_pca,V = np.linalg.eig(C_x)

sortIndex = np.flip(np.argsort(SS_pca))

dimension = good_data.shape[1]
VSorted = np.empty((dimension,0))

for i in range(dimension):
    VSorted = np.append(VSorted, V[:,sortIndex[i]].reshape(dimension,1), axis=1)


classificationError_lda_pca = np.zeros(5,)
classificationError_svm_pca = np.zeros(5,)

Score_Sorted = np.dot(X,VSorted)

train_index  = np.arange(0,n_sample*0.75).astype(int).tolist()+np.arange(n_sample,n_sample+n_sample*0.75).astype(int).tolist()
test_index = np.arange(n_sample*0.75,n_sample).astype(int).tolist()+np.arange(n_sample+n_sample*0.75,n_sample+n_sample).astype(int).tolist()

#  data for training and testing
X_train  = Score_Sorted[train_index,:]
X_test = Score_Sorted[test_index,:]


Y_train = Y[train_index]
Y_test = Y[test_index]


# Classifier 1 LDA with PCA

start_time = time.time()
main()
print("--- %s seconds ---" % (time.time() - start_time))

for numDims in range(4,9): 
    
   
    Score_Reduced = X_train[:,0:numDims]
    print('Reduced score shape is ', Score_Reduced.shape)
   

    lda_pca = sklearn.discriminant_analysis.LinearDiscriminantAnalysis()
    X_test_temp = X_test[:,0:numDims]

    error,prediction_lda_pca = train_test(Score_Reduced,Y_train,X_test_temp,Y_test,  lda_pca)
    
    
    classificationError_lda_pca[numDims-4] =error
    print("Confusion matrix for LDA with PCA")

    plot_confusion_matrix(Y_test, prediction_lda_pca, lda_pca)




# Classifier 2 SVM with PCA

for numDims in range(4,9): 
    
    
    Score_Reduced = X_train[:,0:numDims]
    print('Reduced score shape is ', Score_Reduced.shape)

    clf_svm_pca = svm.SVC(kernel = 'linear')



    X_test_temp = X_test[:,0:numDims]

    error, prediction_svm_pca = train_test(
        Score_Reduced, Y_train, X_test_temp, Y_test,  clf_svm_pca)
    
    classificationError_svm_pca[numDims-4] = error



    print("Confusion matrix for SVM with PCA")
    plot_confusion_matrix(Y_test, prediction_svm_pca, lda_pca)



# # Feature selection-backward search

print("Start Feature Selection process...")

X_train_fs = X[train_index,:]

X_test_fs = X[test_index,:]

n_train = X_train_fs.shape[0]
n_test = X_test_fs.shape[0]
final_dimension = 5

# Classifier 1 LDA with Feature selection
print('Start Feature Selection with LDA...')




removed = []

index_all = [0,1,2,3,4,5, 6, 7]
remaining = index_all




classificationError_lda_fs= n_test*np.ones(final_dimension)

lda_fs = sklearn.discriminant_analysis.LinearDiscriminantAnalysis()

error_temp, item_ignored = train_test(X_train_fs, Y_train, X_test_fs, Y_test, lda_fs)

classificationError_lda_fs[0] = error_temp


for iteration in range(final_dimension-1):

    error_inside = n_test*np.ones(n_feature)
    for idx, item in enumerate(remaining):
        temp_removed = removed[:]
        temp_removed.append(item)
        # print(temp_removed)

        Xtrain_temp = np.delete(X_train_fs, temp_removed, axis=1)
        Xtest_temp = np.delete(X_test_fs, temp_removed, axis=1)

        lda_temp = sklearn.discriminant_analysis.LinearDiscriminantAnalysis()

        error_temp,item_ignored = train_test(Xtrain_temp, Y_train,
                                Xtest_temp, Y_test, lda_temp)

        error_inside[idx] = error_temp

    worst_i = np.argmin(error_inside)

    worst_item = remaining[worst_i]

    removed.append(worst_item)
    print("The removed colomns", removed)

    remaining.remove(worst_item)
    print("The remained colomns {}\n".format(remaining))

    X_train_selection = np.delete(X_train_fs, removed, 1)
    X_test_selection = np.delete(X_test_fs, removed, 1)


    lda_fs = sklearn.discriminant_analysis.LinearDiscriminantAnalysis()


    error, prediction_lda_fs = train_test(X_train_selection, Y_train,
                        X_test_selection, Y_test, lda_fs)

    classificationError_lda_fs[iteration+1] = error

    print("Confusion matrix for LDA with Feature selection")

    plot_confusion_matrix(Y_test, prediction_lda_fs, lda_fs)
    
       


Classifier 2 SVM with Feature selection

print('Start Feature Selection with SVM...')

classificationError_svm_fs = n_test*np.ones(final_dimension)

removed = []

index_all = [0, 1, 2, 3, 4, 5, 6, 7]
remaining = index_all


clf_svm_fs = svm.SVC(kernel = 'linear')
error_temp, item_ignored = train_test(
    X_train_fs, Y_train, X_test_fs, Y_test, clf_svm_fs)

classificationError_svm_fs[0] = error_temp


for iteration in range(final_dimension-1):

    error_inside = n_test*np.ones(n_feature)
    for idx, item in enumerate(remaining):
        temp_removed = removed[:]
        temp_removed.append(item)
        # print(temp_removed)

        Xtrain_temp = np.delete(X_train_fs, temp_removed, axis=1)
        Xtest_temp = np.delete(X_test_fs, temp_removed, axis=1)

        svm_temp = svm.SVC(kernel = 'linear')
        error_temp, item_ignored = train_test(Xtrain_temp, Y_train,
                                              Xtest_temp, Y_test, svm_temp)

        error_inside[idx] = error_temp

    worst_i = np.argmin(error_inside)

    worst_item = remaining[worst_i]

    removed.append(worst_item)
    print("The removed colomns", removed)

    remaining.remove(worst_item)
    print("The remained colomns {}\n".format(remaining))

    X_train_selection = np.delete(X_train_fs, removed, 1)
    X_test_selection = np.delete(X_test_fs, removed, 1)

    svm_fs = svm.SVC(kernel='linear')

    error, prediction_svm_fs = train_test(X_train_selection, Y_train,
                                          X_test_selection, Y_test, svm_fs)


    classificationError_svm_fs[iteration+1] = error

    print("Confusion matrix for SVM with Feature selection")

    plot_confusion_matrix(Y_test, prediction_svm_fs, svm_fs)



plt.figure()
plt.scatter([8,7,6,5,4], np.flip(classificationError_lda_pca), c = 'b', marker = '*', label = "PCA+LDA")
plt.scatter([8, 7, 6, 5, 4], np.flip(classificationError_svm_pca), c='r', marker='o', label = "PCA+SVM")
plt.xlabel('Dimension')
plt.ylabel('Error')
plt.title('PCA Error')
plt.legend()


plt.figure()
plt.scatter([8,7,6,5,4], classificationError_lda_fs, c = 'b', marker = '*', label = "Feature Selection+LDA")
plt.scatter([8, 7, 6, 5, 4], classificationError_svm_fs, c='r', marker='o', label = "Feature Selection+SVM")
plt.xlabel('Dimension')
plt.ylabel('Error')
plt.title('Feature Selection Error')
plt.legend()