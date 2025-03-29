# PointCloudManipulation

The goal of this repository is to understand and implement all the pointcloud manipulation techniques using popular libraries like open3d , pcl 

### Filtering Techniques

* Downsampling: Reducing point density while preserving features (voxel grid, random sampling)
* Outlier Removal: Statistical outlier removal, radius outlier removal
* Noise Filtering: Bilateral filtering, moving least squares (MLS)

### Registration

* Iterative Closest Point (ICP): Aligns multiple point clouds by minimizing distance between points
* Normal Distributions Transform (NDT): Represents point distributions as combinations of normal distributions
* Feature-Based Registration: Using keypoints and descriptors (FPFH, SHOT)
Global Registration: RANSAC-based approaches for initial alignment

### Segmentation

* Region Growing: Grouping points based on proximity and similar attributes
* Model Fitting: RANSAC for identifying geometric primitives (planes, cylinders)
* Clustering: K-means, DBSCAN, Euclidean clustering
* Graph-Based: Normalized cuts, minimum spanning trees
* Deep Learning: PointNet/PointNet++, RandLA-Net for semantic segmentation

### Feature Extraction

* Normal Estimation: Computing surface normals and curvatures
* Keypoint Detection: ISS, Harris 3D
* Descriptors: FPFH, SHOT, RoPS, PFH
* Geometric Feature Calculation: Shape distributions, spin images

### Surface Reconstruction

* Poisson Surface Reconstruction: Creates watertight meshes
* Greedy Triangulation: Fast triangulation between nearest neighbors
* Alpha Shapes: Connecting points within alpha radius
* Ball Pivoting: Rolling a ball on point set to form triangles

### Deep Learning Approaches

* Point Cloud Completion: Filling missing regions (PCN, PoinTr)
* Upsampling: Increasing point density (PU-Net, PU-GAN)
* Style Transfer: Transferring geometric details between point clouds
* Point Cloud Generation: GAN/VAE-based approaches

### Visualization and Rendering

* Point-Based Rendering: Splatting, surfel rendering
* Level-of-Detail: Adaptive rendering based on view distance
* Color Manipulation: Colorization by height, intensity, classification

### Data Structures

* Octree/KD-Tree: Hierarchical spatial partitioning for efficient operations
* Voxel Grid: Regular 3D grid for structured representation
* Graph Representation: Points as vertices, connections as edges

### Transformation Operations

* Rigid Transformations: Translation, rotation, scaling
* Non-Rigid Deformation: Free-form deformation, as-rigid-as-possible
* Point Set Resampling: Blue noise sampling, Poisson disk sampling


### Clustering Techniques

* K-means Clustering: Partitions points into K clusters by iteratively assigning points to the nearest centroid and recalculating centroids
* DBSCAN (Density-Based Spatial Clustering of Applications with Noise): Groups points based on density, identifying core points, border points, and noise points without requiring a predetermined number of clusters
* Euclidean Clustering: Groups points based on Euclidean distance thresholds, commonly implemented with k-d trees for efficiency
* Hierarchical Clustering: Creates nested clusters by either merging (agglomerative) or splitting (divisive) based on distance metrics
* Gaussian Mixture Models (GMM): Models clusters as a mixture of Gaussian distributions
* Mean-shift Clustering: Non-parametric technique that finds modes in a density function
* Spectral Clustering: Uses eigenvalues of similarity matrices to reduce dimensionality before clustering

### RANSAC Plane Fitting

* Basic RANSAC Plane Fitting: Iteratively samples three points, forms a plane hypothesis, counts inliers, and selects the best-fitting plane
* Progressive RANSAC: Extracts multiple planes sequentially, removing inliers after each detection
* MLESAC/MSAC: Modified RANSAC variants with improved scoring functions
* Parallel RANSAC: Runs multiple RANSAC processes simultaneously for efficiency
* Constrained RANSAC: Incorporates prior knowledge about plane orientations or positions
* Multi-plane Fitting: Simultaneously detecting multiple planes (e.g., J-Linkage, T-Linkage)
* Regularized RANSAC: Adds regularization terms to prevent overfitting to noise

### Point Cloud Classification

* Traditional ML Classification: SVM, Random Forests using hand-crafted features
* Deep Learning Classification: PointNet/PointNet++, DGCNN, PointCNN for direct point cloud classification
* Multi-view Based: Rendering multiple 2D views and using CNNs
Voxel-based: Converting to voxel grids for 3D CNNs

### Compression and Transmission

* Octree-based Compression: Hierarchical compression schemes
* Graph-based Compression: Using graph signal processing
* Deep Learning Compression: Autoencoder approaches
* Progressive Transmission: Level-of-detail based transmission methods

### Time-Series/Dynamic Point Cloud Processing

* Scene Flow Estimation: Calculating point movements between consecutive frames
* 4D Point Cloud Processing: Handling temporal dimensions for moving objects
* Dynamic Registration: Aligning point clouds with moving objects

### Quality Assessment

* Point Cloud Quality Metrics: Measuring distortion, point-to-point/point-to-plane errors
* Perceptual Quality Assessment: Human visual system-based metrics

### Domain Adaptation and Transfer Learning

* Cross-domain Point Cloud Processing: Adapting models between different sensors
* Sim-to-Real Transfer: Transferring models trained on synthetic data to real data

These techniques are particularly important for indoor scene understanding, building information modeling (BIM), and robotics applications where identifying planar surfaces and distinct object clusters is critical.

I attempt to implement all these using an interesting usecase that helps me understand all that is required in current industry for robotics and autonomous vehicles.