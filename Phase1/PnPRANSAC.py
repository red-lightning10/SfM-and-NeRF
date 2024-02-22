from LinearPnP import LinearPnP
from LinearTriangulation import create_projection_matrix
import numpy as np

def project_from_world_to_image(X, K, R, C):

    P = create_projection_matrix(K, C, R)
    X_h = np.hstack((X, np.ones((X.shape[0], 1))))
    x = P @ X_h.T
    x = x / x[2, :]
    x = x[0:2, :].T
    return x

def PnPRANSAC(X_3d, x_2d, K, threshold, nIterations=1000):
    """
    X_3d: 3D points in world frame
    x_2d: 2D points in image frame
    K: Camera intrinsics
    threshold: RANSAC threshold
    max_iterations: Maximum RANSAC iterations
    """
 
    inliers = []
    C_best = []
    R_best = []
    num_max_inliers = 0

    for i in range(nIterations):
        # Randomly sample 6 points
        idx = np.random.choice(X_3d.shape[0], 6, replace=False)
        X_sample = X_3d[idx]
        x_sample = x_2d[idx]
        # Solve PnP
        R, C = LinearPnP(x_sample, X_sample, K)
        # Project 3D points to 2D
        x_2d_proj = project_from_world_to_image(X_3d, K, R, C)
        # Calculate reprojection error
        error = np.linalg.norm(x_2d - x_2d_proj, axis=1)
        # Count inliers
        inliers_check = error < threshold
        # Update best inliers
        if sum(inliers_check) > num_max_inliers:
            num_max_inliers = sum(inliers_check)
            inliers_indices = np.where(inliers_check)
            C_best = C
            R_best = R
    
    inliers = X_3d[inliers_indices]
    return R_best, C_best, inliers