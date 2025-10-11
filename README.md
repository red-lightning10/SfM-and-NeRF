# SfM and NeRF Project

This repository contains implementations of Structure from Motion (SfM) and Neural Radiance Fields (NeRF) for 3D reconstruction and novel view synthesis given sparse set of images. 

## Phase 1: Structure from Motion (SfM)

### How to Run

1. Navigate to the Phase1 directory:
```bash
cd Phase1
```

2. Run the main pipeline:
```bash
python3 Wrapper.py
```

### Requirements
- Python 3
- OpenCV
- NumPy
- SciPy
- Matplotlib

### Output
The pipeline generates:
- Feature correspondences between images
- Epipolar lines visualization
- Reprojection visualizations
- 3D point cloud reconstruction
- Camera pose visualization

## Phase 2: Neural Radiance Fields (NeRF)

Neural Radiance Fields synthesize novel views of 3D scenes using neural networks to represent continuous volumetric scenes.

### How to Run Phase 2

1. Navigate to the Phase2 directory:
```bash
cd Phase2
```

2. Run the NeRF pipeline:
```bash
python3 Wrapper.py --data_path ./data/lego --mode train
```

### Requirements
- Python 3
- PyTorch
- NumPy
- OpenCV
- Matplotlib

### Output
The pipeline generates:
- Trained NeRF models
- Novel view synthesis
- Loss plots during training
- Rendered images

## Data Structure

```
Data/
├── 1.png, 2.png, 3.png, 4.png, 5.png  # Input images for SfM
├── calibration.txt                     # Camera calibration parameters
├── matching*.txt                       # Feature matching files
└── Calibration/                        # Additional calibration images
    ├── cameraParams.mat
    └── *.png
```

## Results

Results are saved in:
- `Phase1/Results/` - SfM outputs (correspondences, epipolar lines, 3D reconstruction)
- `Phase2/images/` - NeRF rendered images
- `Phase2/checkpoints/` - Trained model checkpoints