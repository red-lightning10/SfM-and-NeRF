import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rotation
from scipy.sparse import lil_matrix

def getVisibleCameraPointPairs(visibilityMatrix):
    cameraIndices = []
    pointIndices = []
    for pointIndex, visibilityRow in enumerate(visibilityMatrix):
        for cameraIndex, isVisible in enumerate(visibilityRow):
            if isVisible:
                cameraIndices.append(cameraIndex)
                pointIndices.append(pointIndex)
    return np.array(cameraIndices), np.array(pointIndices)

def createSparsityPattern(numCameras, numPoints, visibilityMatrix):
    cameraIndices, pointIndices = getVisibleCameraPointPairs(visibilityMatrix)
    numObservations = cameraIndices.size * 2
    numParameters = numCameras * 6 + numPoints * 3
    
    sparsityPattern = lil_matrix((numObservations, numParameters), dtype=int)
    observationIndices = np.arange(cameraIndices.size)
    
    # Camera parameters
    for paramIndex in range(6):
        col = cameraIndices * 6 + paramIndex
        sparsityPattern[2 * observationIndices, col] = 1
        sparsityPattern[2 * observationIndices + 1, col] = 1
    
    # Point parameters
    for paramIndex in range(3):
        col = numCameras * 6 + pointIndices * 3 + paramIndex
        sparsityPattern[2 * observationIndices, col] = 1
        sparsityPattern[2 * observationIndices + 1, col] = 1
        
    return sparsityPattern.tocsc()

def computeP(K, rotationMatrix, cameraCenter):
    cameraCenter = cameraCenter.reshape(3, 1)
    identityMatrix = np.identity(3)
    return K @ (rotationMatrix @ np.hstack((identityMatrix, -cameraCenter)))

def calculateReprojectionErrors(parameters, allPoints, visibilityMatrix, numCameras, numPoints, K):
    eulerAngles = parameters[:numCameras*3].reshape(numCameras, 3)
    translations = parameters[numCameras*3:numCameras*6].reshape(numCameras, 3)
    worldPoints = parameters[numCameras*6:].reshape(numPoints, 3)
    
    rotationMatrices = [Rotation.from_euler('zyx', angles).as_matrix() for angles in eulerAngles]
    errors = []

    for pointIndex, visibilityRow in enumerate(visibilityMatrix):
        worldPointHomogeneous = np.append(worldPoints[pointIndex], 1)
        
        for cameraIndex, isVisible in enumerate(visibilityRow):
            if not isVisible:
                continue
                
            imageKey = f'image{cameraIndex+1}_points'
            if imageKey not in allPoints[pointIndex]:
                continue
                
            observedU, observedV = allPoints[pointIndex][imageKey]
            C = translations[cameraIndex].reshape(3, 1)
            identityMatrix = np.identity(3)
            P = K @ (rotationMatrices[cameraIndex] @ np.hstack((identityMatrix, -C)))
            
            # Calculate projected coordinates
            rowU, rowV, rowW = P
            projectedU = (rowU @ worldPointHomogeneous) / (rowW @ worldPointHomogeneous)
            projectedV = (rowV @ worldPointHomogeneous) / (rowW @ worldPointHomogeneous)
            
            # Calculate squared errors
            errors.append((observedU - projectedU)**2)
            errors.append((observedV - projectedV)**2)
    
    return np.array(errors).ravel()

def BundleAdjustment(Cset, Rset, PC, K, max_iterations=50):
    """
    Bundle adjustment to refine camera poses and 3D points
    Based on reference implementation
    """
    print(f"Bundle adjustment: {len(Cset)} cameras, {len(PC.point_cloud)} points")
    
    numCameras = len(Cset)
    numPoints = len(PC.point_cloud)
    
    if numCameras == 0 or numPoints == 0:
        print("No cameras or points to optimize")
        return Cset, Rset, PC
    
    # Convert data to reference format
    worldPoints = np.array(PC.point_cloud)
    initialRotations = Rset
    initialTranslations = [C.flatten() for C in Cset]
    
    # Create visibility matrix and allPoints structure
    visibilityMatrix = np.zeros((numPoints, numCameras), dtype=bool)
    allPoints = []
    
    # Build visibility matrix and observation data
    for pointIdx in range(numPoints):
        pointObservations = {}
        for camIdx in range(numCameras):
            if camIdx + 1 in PC.point_correspondence:
                observed_points = PC.point_correspondence[camIdx + 1]
                if pointIdx < len(observed_points):
                    observedU, observedV = observed_points[pointIdx]
                    pointObservations[f'image{camIdx+1}_points'] = (observedU, observedV)
                    visibilityMatrix[pointIdx, camIdx] = True
        allPoints.append(pointObservations)
    
    try:
        # Convert rotations to Euler angles for parameterization
        eulerAngles = [Rotation.from_matrix(R).as_euler('zyx') for R in initialRotations]
        
        # Initialize parameter vector
        initialParameters = np.hstack([
            np.array(eulerAngles).ravel(),
            np.array(initialTranslations).ravel(),
            worldPoints.ravel()
        ])
        
        # Create sparsity pattern and optimize
        sparsityMatrix = createSparsityPattern(numCameras, numPoints, visibilityMatrix)
        optimizationResult = least_squares(
            fun=calculateReprojectionErrors,
            jac_sparsity=sparsityMatrix,
            method="trf",
            x0=initialParameters,
            ftol=1e-9,
            verbose=0,
            x_scale='jac',
            args=(allPoints, visibilityMatrix, numCameras, numPoints, K)
        )
        
        if optimizationResult.success:
            print(f"Bundle adjustment converged after {optimizationResult.nfev} function evaluations")
            
            # Extract optimized parameters
            optimizedParameters = optimizationResult.x
            optimizedRotations = [
                Rotation.from_euler('zyx', angles).as_matrix()
                for angles in optimizedParameters[:numCameras*3].reshape(numCameras, 3)
            ]
            optimizedTranslations = optimizedParameters[numCameras*3:numCameras*6].reshape(numCameras, 3)
            optimizedWorldPoints = optimizedParameters[numCameras*6:].reshape(numPoints, 3)
            
            # Convert back to original format
            optimized_Cset = [t.reshape(3,1) for t in optimizedTranslations]
            PC.point_cloud = optimizedWorldPoints.tolist()
            
            return optimized_Cset, optimizedRotations, PC
            
        else:
            print(f"Bundle adjustment failed to converge: {optimizationResult.message}")
            return Cset, Rset, PC
            
    except Exception as e:
        print(f"Bundle adjustment error: {e}")
        return Cset, Rset, PC