import numpy as np

def EfromF(F,K):
    
    E = np.dot(K.T, np.dot(F,K)) #initial E matrix
    U, _, V = np.linalg.svd(E)

    D = np.diag([1,1,0])
    final_E =  np.dot(U, np.dot(D,V))

    return final_E
