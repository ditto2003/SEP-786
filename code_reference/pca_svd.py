import numpy as np

#PCA functions
#PCA(w/ dim reduce) - SVD on data X; returns in score space, X_mean, Vt from SVD on X
def pca(X,k=None):
       #determine k size
       if k == None:
              k = np.size(X,1)
       #remove mean: feature normalization
       mu = np.mean(X, axis=0)
       Xc = X - mu
       #SVD on Xc; full_matrices=False for n >> p matrices (thin SVD)
       U,S,Vt = np.linalg.svd(Xc,full_matrices=False)
       #transpose Vt -> columns are principle axes
       V = Vt.T
       #compute scores projected on k-dimensions = XV
       score1 = np.dot(Xc,V[:,:k])
       #compute scores projected on k-dimensions = US - proved identical to XV
       # S_diag = np.diag(S)
       # S_k = S_diag[:k,:k]
       # U_k = U[:,:k]
       # score2 = np.dot(U_k,S_k)

       return score1,mu,Vt

#PCA reconstruction; input: scores, Vt from original SVD, mean; returns X_estimate
def pca_recon(X_reduce, X_mean, Vt):
       #read current dimension size
       k = np.size(X_reduce,1)
       #reconstruct X
       X_rebuild = np.dot(X_reduce,Vt[:k,:])
       X_rebuild = X_rebuild + X_mean
       
       return X_rebuild

