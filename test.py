import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import sklearn.discriminant_analysis
# from sklearn import svm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns

from sklearn import model_selection
import timeit
# import sys

PATH_GOOD = '../SEP-786/data/baseline_20220915_sv.mat'
PATH_BAD = '../SEP-786/data/fault7_20220915_sv.mat'


def load_mat_single(data_path):
    """Load the .mat files"""
    mat_contents = sio.loadmat(data_path)
    # for i, key in enumerate(mat_contents):
        # print(i, key)
    return mat_contents

def mat_to_array(mat_contents):
    """Conver the .mat data into numpy array"""
    mat_name = []
    mat_data = []
    for i, (k, v) in enumerate(mat_contents.items()):
        mat_name.append(k) 
        mat_data.append(v)

    vibration_signal_all = np.array(mat_data[3])
    return vibration_signal_all

def pca_data(data, debug=False):
    """PCA algorithm to reduct dimention"""
    # 1. Normalize the data by minus each mean of the column
    data_normalized = data - np.mean(data, axis=0)
    # 2. Calc covariance matrix by normalized data
    cov_normalized = np.dot(data_normalized.T, data_normalized)
    # 3. Calc the eigen value and eigen vector by covariance matrix
    eig_value, eig_vector = np.linalg.eig(cov_normalized)
    # 4. Order the eigen value from the large to small
    sorted_eigen_value_index = np.argsort(-eig_value)   # (8, )
    # 5. Sort the eigen vector by the sorted index of the eigen value
    sorted_eigen_vector = eig_vector[:, sorted_eigen_value_index]   # (8,8)
    ####################
    # Reduct one dimension
    sorted_eigen_vector = eig_vector[:, :-2]
    ####################
    # 6. Reduct the dimention by convert to score space
    data_pca = np.dot(data_normalized, sorted_eigen_vector)     # (96000, 8)
    # 7. Re-construct the data
    reconstruct_data = np.dot(data_pca, sorted_eigen_vector.T) + np.mean(data, axis=0)

    # Plot the MSE
    mse_comparison = []
    mse = []
    for col in range(data.shape[1]):
        mse.append(np.sum(np.square(reconstruct_data[col]-data[col]), axis=0) / data.shape[0])  # 96000
    # print(f"Original MSE: {mse}")
    plt.ioff()
    plt.figure(figsize=(10, 5))
    # plt.xlabel('# of Features')
    # plt.ylabel('MSE')
    plt.plot(mse, label="Original data")
    mse_comparison.append(np.sum(mse, axis=0))

    if debug is True:
        while(True):
            reduct_dim = int(input("Type the number of the dimensional reduction:"))
            if reduct_dim >= sorted_eigen_value_index.shape[0]:
                print(f"{reduct_dim} exceeds the maximum dimension of {sorted_eigen_value_index.shape[0]}")
                plt.legend()
                plt.show()
                return reconstruct_data
                # return sys.exit(1)
            # Modifiy the vector by selected reduction of dimension
            modified_vector = sorted_eigen_vector[:, :(sorted_eigen_vector.shape[0]-reduct_dim)]
            # Reduct the data
            reduct_data = np.dot(data_normalized, modified_vector)
            # reconstruct the data
            modified_reconstruct_data = np.dot(reduct_data, modified_vector.T) + np.mean(data, axis=0)
            # Calc MSE between reconstructed data and original data
            mse = []
            for col in range(data.shape[1]):
                mse.append(np.sum(np.square(modified_reconstruct_data[col]-data[col]), axis=0) / data.shape[0])  # 96000
            print(f"MSE: {mse}")
            
            # Plot the MSE
            plt.plot(mse, label=str(reduct_dim))

    # plt.ion()
    return reconstruct_data
    """Optimizing process for automation test"""
    # for num in range(1, sorted_eigen_value_index.shape[0]):
    #     modified_vector = sorted_eigen_vector[:, :(sorted_eigen_vector.shape[0]-num)]
    #     reduct_data = np.dot(data_normalized, modified_vector)
    #     modified_reconstruct_data = np.dot(reduct_data, modified_vector.T) + np.mean(data, axis=0)
    #     for col in range(data.shape[1]):
    #         mse.append(np.sum(np.square(modified_reconstruct_data[col]-data[col]), axis=0) / data.shape[0])  # 96000
    #     mse_comparison.append(np.sum(mse, axis=0))
    # print(mse_comparison)
    # print(min(mse_comparison))
    # # Order the MSE comparison from small to large
    # mse_index = np.argsort(mse_comparison)
    # # Choose the minimum effect reduction of dimension
    # reduct_dim = mse_index
    # print(reduct_dim)
    # # Reconstruct the data again
    # modified_vector = sorted_eigen_vector[:, :sorted_eigen_vector.shape[0]-num]
    # reduct_data = np.dot(data_normalized, modified_vector)
    # return np.dot(reduct_data, modified_vector.T) + np.mean(data, axis=0)

    # Compare the MSE to select the minimum effect reduction number
    # for num in range(0, sorted_eigen_value_index.shape[0]):
    #     mse_comp = []
    #     mse = []
    #     for col in range(data.shape[1]):

def lda_test(data, label):
    """LDA algorithm"""
    X_train, X_test, y_train, y_test = model_selection.train_test_split(data, label, test_size=0.25, random_state=1)
    lda = sklearn.discriminant_analysis.LinearDiscriminantAnalysis()
    lda.fit(X_train, y_train)
    prediction = lda.predict(X_test)
    error = sum(prediction != y_test)
    print(f"Total feature selection error with three features: {error}")
    plot_confusion_matrix("LDA", y_test, prediction)

def plot_confusion_matrix(model, y_test, y_pred):
    plt.figure(figsize=(10, 5))
    plt.ioff()
    cf_matrix = confusion_matrix(y_test, y_pred)
    ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues')
    ax.set_title('%s' % model)
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ')

    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['False', 'True'])
    ax.yaxis.set_ticklabels(['False', 'True'])
    ## Display the visualization of the Confusion Matrix.
    # 
    #    
if __name__ == "__main__":
    # Load the .mat files
    mat_contents_good = load_mat_single(PATH_GOOD)
    mat_contents_bad = load_mat_single(PATH_BAD)
    # Convert to numpy array
    good_data = mat_to_array(mat_contents_good) # numpy.ndarray (48000,8)
    bad_data = mat_to_array(mat_contents_bad)   # numpy.ndarray (48000,8)
    # Concatenate the data set and label set
    X = np.concatenate((good_data,bad_data), axis=0)    # (96000,8)
    Y = np.concatenate((np.zeros(good_data.shape[0]), np.ones(bad_data.shape[0])), axis=0)  # (96000, )

    # LDA for original data
    lda_test(data=X, label=Y)
    lda_time = timeit.Timer(stmt="lda_test", setup="from __main__ import lda_test, X, Y")
    print("LDA processing time: ", lda_time.timeit(number=1000), "milliseconds")


    # PCA reduct one dimension
    pca_data = pca_data(X, debug=True)
    pca_time = timeit.Timer(stmt="pca_data", setup="from __main__ import pca_data, X")
    print("PCA processing time: ", pca_time.timeit(number=1000), "milliseconds")

    # LDA for selected features data
    lda_test(data=pca_data, label=Y)
    lda_time = timeit.Timer(stmt="lda_test", setup="from __main__ import lda_test, pca_data, Y")
    print("LDA processing time: ", lda_time.timeit(number=1000), "milliseconds")

    plt.show()