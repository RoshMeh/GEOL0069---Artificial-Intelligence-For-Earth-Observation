# GEOL0069_Week4 Assessment: Satellite Echo Classification

This repository focuses on classifying echoes from sea ice and leads using machine learning techniques. It aims to generate the average echo shape and standard deviation for both classes and assess the classification performance against the official ESA classification using a confusion matrix.



<!-- ABOUT THE PROJECT -->
## About The Project

This project investigates the integration of Sentinel-3 and Sentinel-2 optical data, leveraging unsupervised learning techniques to classify sea ice and leads. By fusing multiple satellite datasets and applying machine learning models, it aims to enhance environmental feature classification and provide a practical approach to unsupervised learning in real-world Earth Observation (EO) applications.

Why This Matters:
 * Combining different satellite datasets offers a more comprehensive and detailed view of Earth's surface.
 * Machine learning enables the detection of patterns and the automated classification of EO data, improving efficiency and accuracy.
   

As AI continues to evolve, its applications in satellite data analysis are expanding. This project serves as a hands-on guide to integrating remote sensing data with machine learning, providing insights into innovative classification techniques.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


### Built With

This project utilises advanced Python libraries and geospatial tools to efficiently process, analyse, and classify Earth Observation (EO) data, providing accurate environmental insights and data-driven decision-making:

- **NumPy** – Numerical computations  
- **Pandas** – Data processing  
- **Matplotlib & Seaborn** – Visualization  
- **Scikit-Learn** – Clustering (K-Means, GMM)  
- **NetCDF4** – Handling satellite datasets  
- **SciPy** – Statistical analysis  

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- Step 0: Read in Functions Needed -->
### Step 0: Read in Functions Needed

Start by making sure that Google Drive is mounted in Google Colab for access to files.
```python
from google.colab import drive
drive.mount('/content/drive')
```
Then install rasterio and netCDF4

```python
pip install rasterio
pip install netCDF4
```

<!-- Step 1: Get the Metadata for satellites (Sentinel-2 and Sentinel-3 OLCI in this case) -->
### Step 1:Unsupervised Learning

Unsupervised learning is a powerful branch of machine learning that discovers hidden patterns in data without predefined labels. This technique is particularly useful for Earth Observation (EO) applications, where labeled datasets can be scarce.


In this section, we apply unsupervised learning to classify sea ice and lead based on satellite data. By leveraging these techniques, we aim to:

* Identify structures and relationships in environmental datasets.
* Automatically classify different surface features.
* Gain insights into how clustering can assist in remote sensing analysis.

<!-- Introduction to Unsupervised Learning Methods [Bishop and Nasrabadi, 2006] -->
### Introduction to Unsupervised Learning Methods

<!-- Introduction to K-means Clustering -->
#### K-Means Clustering for Sea Ice and Lead

What is K-Means?

K-Means is an unsupervised learning algorithm that groups data into k clusters based on similarity. The core idea is to:

* Define k cluster centers.
* Assign each data point to the nearest center.
* Recalculate cluster centers iteratively until convergence.

Why Use K-Means for Clustering?


K-Means is widely used in remote sensing and environmental classification because:

It is efficient and can scale to large datasets.
It does not require labeled data, making it ideal for exploratory analysis.
It effectively segments complex datasets like satellite imagery.


<!-- Why K-means for Clustering? -->

#### Key Components of K-Means
The K-Means algorithm consists of four fundamental steps:

* Choosing K: Define the number of clusters (k) based on the dataset.
* Centroid Initialization: Select initial cluster centers to begin grouping.
* Assignment Step: Assign each data point to the closest centroid.
* Update Step: Recalculate centroids based on assigned points.


These steps are repeated until cluster centers stabilize, ensuring an optimal grouping of data.

<!-- Key Components of K-means -->
#### The Iterative Process of K-Means


K-Means follows an iterative refinement process:

* Data points are re-assigned to clusters at each step.
* The centroids shift dynamically until minimal intra-cluster changes occur.
* The algorithm converges when data points no longer switch clusters.


This ensures a well-defined clustering of features, such as differentiating between sea ice and open water leads.

<!-- The Iterative Process of K-means -->
#### Advantages of K-Means in Remote Sensing
* Computationally efficient: Works well with large satellite datasets.
* Scalable and interpretable: Easy to understand and visualize.
* Effective for environmental data: Can distinguish between natural patterns in Earth Observation imagery.


By applying K-Means clustering, we can extract meaningful insights from satellite-derived features, aiding in automatic classification of environmental elements like sea ice and water leads.

<!-- Basic Code Implementation -->
#### Code

Using scikit-learn, we generate 100 random data points and apply the K-Means clustering algorithm. The model is initialized and assigns each data point to one of the clusters. To visualize the results, a scatter plot is created, where:

* Each point is color-coded based on its assigned cluster.
* The cluster centroids are highlighted with a black dot.


This visualization effectively demonstrates how K-Means identifies patterns and segments data into distinct groups. 

```python
# Python code for K-means clustering
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

# Sample data
X = np.random.rand(100, 2)

# K-means model
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

# Plotting
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
plt.show()
```
<img src="https://github.com/Guoxuan-Li/GEOL0069_Week4/blob/main/Images/K_means.jpg?raw=true" width="500" height="400" />


<!-- Gaussian Mixture Models (GMM) [Bishop and Nasrabadi, 2006] -->
### Gaussian Mixture Models (GMM) [Bishop and Nasrabadi, 2006]

<!-- Introduction to Gaussian Mixture Models -->
#### Introduction to Gaussian Mixture Models
Gaussian mixture model (GMM) is a probabilistic model used to represent normally distributed subpopulations in a population. The model assumes that the data is generated from a mixture of several Gaussian distributions, each with its own mean and variance[Reynolds et al., 2009]. GMM is widely used in clustering and density estimation because they provide a way to represent complex distributions by combinations of simple distributions.

<!-- Why Gaussian Mixture Models for Clustering? -->
#### Why Gaussian Mixture Models for Clustering?
Gaussian mixture model is useful where soft clusering is needed and flexibility in cluster covariance. GMM provides the probability that each data point belongs to each cluster, providing a soft classification and understanding of data uncertainty. And it allows clusters to have different sizes and shapes, allowing for more flexibility in capturing the true variance in the data.

<!-- Key Components of GMM -->
#### Key Components of GMM
There are three key components of GMM: Number of Gaussians Components, Expectation-Macimisatin (EM) Algorithm and Covariance Type. First set a Gaussian number (Components), using the EM algorithm to fit, iteratively improve the data possibilities of the given model. The covariance type of Gaussian determines the shape, size and orientation of the clusters.

<!-- The EM Algorithm in GMM -->
#### The EM Algorithm in GMM
The Expectation-Maximisation (EM) algorithm is a two-step process until convergence: Expectation Step (E-step) and Maximisation Step (M-step). The E-step is used to calculate the probability that each data point belongs to each cluster. And the M-step updates the parameters of the Gaussian function (mean, covariance, and mixing coefficients) to maximise the probability of the data given these distributions.

<!-- Advantages of GMM -->
#### Advantages of GMM
The GMM provides a probabilistic framework for soft clustering, which providing more information about uncertainty in data allocation. In addition, it can adapt to ellipsoidal cluster shapes, due to the flexible covariance structure.

<!-- Basic Code Implementation -->
#### Basic Code Implementation
Using sklearn, randomly create 100 data points, initialise a GMM model with three groups, and assign each data point. The data is visualised as a scatter plot, the clusters are represented by color-coded points, and the calculated cluster centers are marked with black dots. This demonstrates how GMM can group data using probability distributions.

```python
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import numpy as np

# Sample data
X = np.random.rand(100, 2)

# GMM model
gmm = GaussianMixture(n_components=3)
gmm.fit(X)
y_gmm = gmm.predict(X)

# Plotting
plt.scatter(X[:, 0], X[:, 1], c=y_gmm, cmap='viridis')
centers = gmm.means_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
plt.title('Gaussian Mixture Model')
plt.show()
```

<img src="https://github.com/Guoxuan-Li/GEOL0069_Week4/blob/main/Images/Gaussian.jpg?raw=true" width="500" height="400" />

<!-- Image Classification -->
### Image Classification
This section will explore the application of these unsupervised methods to image classification tasks, with a particular focus on distinguishing between sea ice and lead in Sentinel-2 images.

<!-- K-Means Implementation -->
#### K-Means Implementation
This script applies K-means clustering for Sentinel-2 satellite images to classify surface features. The clustering results are visualised and different colors are used to represent different clusters. This method can realise the unsupervised classification of Sentinel-2 images for environmental monitoring and research.

```python
import rasterio
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

base_path = "/content/drive/MyDrive/GEOL0069/week4/Unsupervised Learning/S2A_MSIL1C_20190301T235611_N0207_R116_T01WCU_20190302T014622.SAFE/GRANULE/L1C_T01WCU_A019275_20190301T235610/IMG_DATA/" # You need to specify the path
bands_paths = {
    'B4': base_path + 'T01WCU_20190301T235611_B04.jp2',
    'B3': base_path + 'T01WCU_20190301T235611_B03.jp2',
    'B2': base_path + 'T01WCU_20190301T235611_B02.jp2'
}

# Read and stack the band images
band_data = []
for band in ['B4']:
    with rasterio.open(bands_paths[band]) as src:
        band_data.append(src.read(1))

# Stack bands and create a mask for valid data (non-zero values in all bands)
band_stack = np.dstack(band_data)
valid_data_mask = np.all(band_stack > 0, axis=2)

# Reshape for K-means, only including valid data
X = band_stack[valid_data_mask].reshape((-1, 1))

# K-means clustering
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
labels = kmeans.labels_

# Create an empty array for the result, filled with a no-data value (e.g., -1)
labels_image = np.full(band_stack.shape[:2], -1, dtype=int)

# Place cluster labels in the locations corresponding to valid data
labels_image[valid_data_mask] = labels

# Plotting the result
plt.imshow(labels_image, cmap='viridis')
plt.title('K-means clustering on Sentinel-2 Bands')
plt.colorbar(label='Cluster Label')
plt.show()

del kmeans, labels, band_data, band_stack, valid_data_mask, X, labels_image
```

<img src="https://github.com/Guoxuan-Li/GEOL0069_Week4/blob/main/Images/K_mean%20clustering%20on%20s2%20bands.jpg?raw=true" width="500" height="400" />

This graph shows the K-means clustering results for Sentinel-2. Yellow areas may represent sea ice, while green areas indicate other surface types. Purple areas are the no data area.

<!-- GMM Implementation -->
#### GMM Implementation
The script applies GMM clustering to Sentinel-2 satellite images to classify surface features. The clustering results are visualised and different colors are used to represent different clusters.

```python
import rasterio
import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

# Paths to the band images
base_path = "/content/drive/MyDrive/GEOL0069/week4/Unsupervised Learning/S2A_MSIL1C_20190301T235611_N0207_R116_T01WCU_20190302T014622.SAFE/GRANULE/L1C_T01WCU_A019275_20190301T235610/IMG_DATA/" # You need to specify the path
bands_paths = {
    'B4': base_path + 'T01WCU_20190301T235611_B04.jp2',
    'B3': base_path + 'T01WCU_20190301T235611_B03.jp2',
    'B2': base_path + 'T01WCU_20190301T235611_B02.jp2'
}

# Read and stack the band images
band_data = []
for band in ['B4']:
    with rasterio.open(bands_paths[band]) as src:
        band_data.append(src.read(1))

# Stack bands and create a mask for valid data (non-zero values in all bands)
band_stack = np.dstack(band_data)
valid_data_mask = np.all(band_stack > 0, axis=2)

# Reshape for GMM, only including valid data
X = band_stack[valid_data_mask].reshape((-1, 1))

# GMM clustering
gmm = GaussianMixture(n_components=2, random_state=0).fit(X)
labels = gmm.predict(X)

# Create an empty array for the result, filled with a no-data value (e.g., -1)
labels_image = np.full(band_stack.shape[:2], -1, dtype=int)

# Place GMM labels in the locations corresponding to valid data
labels_image[valid_data_mask] = labels

# Plotting the result
plt.imshow(labels_image, cmap='viridis')
plt.title('GMM clustering on Sentinel-2 Bands')
plt.colorbar(label='Cluster Label')
plt.show()
```
<img src="https://github.com/Guoxuan-Li/GEOL0069_Week4/blob/main/Images/GMM%20clustering%20on%20s2%20bands.jpg?raw=true" width="500" height="400" />

This graph shows the GMM clustering results for Sentinel-2. Yellow areas may represent sea ice, while green areas indicate other surface types. Purple areas are the no data area.

<!-- Altimetry Classification-->
### Altimetry Classification
This section applies unsupervised learning to classify Sentinel-3 altimetry data, focusing on distinguishing sea ice from leads using satellite-derived elevation measurements. This approach enhances the analysis of surface features, improving insights into ice dynamics and oceanographic processes.

<!-- Read in Functions Needed -->
#### Read in Functions Needed
Before delving into the modeling process, the data is pre-processed to ensure compatibility with the analytical model. Transform raw data into meaningful variables such as peaks and stack standard deviations (SSDs) and delete some NaN values from the dataset. Using netCDF4, numpy, matplotlib, scipy, glob and sklearn to extract and prrocess the Sentinel-3 SAR altimetry data to classify sea ice and leads. Visualize the data and plot the mean waveform and standard deviation of each class, the specific scripts are in the code folder -- Unsupervised Learning.

```python
# mean and standard deviation for all echoes
mean_ice = np.mean(waves_cleaned[clusters_gmm==0],axis=0)
std_ice = np.std(waves_cleaned[clusters_gmm==0], axis=0)

plt.plot(mean_ice, label='ice')
plt.fill_between(range(len(mean_ice)), mean_ice - std_ice, mean_ice + std_ice, alpha=0.3)


mean_lead = np.mean(waves_cleaned[clusters_gmm==1],axis=0)
std_lead = np.std(waves_cleaned[clusters_gmm==1], axis=0)

plt.plot(mean_lead, label='lead')
plt.fill_between(range(len(mean_lead)), mean_lead - std_lead, mean_lead + std_lead, alpha=0.3)

plt.title('Plot of mean and standard deviation for each class')
plt.legend()
```
<img src="https://github.com/Guoxuan-Li/GEOL0069_Week4/blob/main/Images/plot%20of%20mean%20and%20sd%20for%20each%20class.jpg?raw=true" width="500" height="400" />


```python
x = np.stack([np.arange(1,waves_cleaned.shape[1]+1)]*waves_cleaned.shape[0])
plt.plot(x,waves_cleaned)  # plot of all the echos
plt.show()
```

<img src="https://github.com/Guoxuan-Li/GEOL0069_Week4/blob/main/Images/cleaned.jpg?raw=true" width="500" height="400" />


```python
# plot echos for the lead cluster
x = np.stack([np.arange(1,waves_cleaned[clusters_gmm==1].shape[1]+1)]*waves_cleaned[clusters_gmm==1].shape[0])
plt.plot(x,waves_cleaned[clusters_gmm==1])  # plot of all the echos
plt.show()
```
<img src="https://github.com/Guoxuan-Li/GEOL0069_Week4/blob/main/Images/echos%20for%20the%20lead%20cluster.jpg?raw=true" width="500" height="400" />


```python
# plot echos for the sea ice cluster
x = np.stack([np.arange(1,waves_cleaned[clusters_gmm==0].shape[1]+1)]*waves_cleaned[clusters_gmm==0].shape[0])
plt.plot(x,waves_cleaned[clusters_gmm==0])  # plot of all the echos
plt.show()
```
<img src="https://github.com/Guoxuan-Li/GEOL0069_Week4/blob/main/Images/echos%20for%20the%20sea%20ice%20cluster.jpg?raw=true" width="500" height="400" />


<!-- Scatter Plots of Clustered Data -->
### Scatter Plots of Clustered Data
Visualizes the clustering results using scatter plots, where different colors represent different clusters. The three graphs illustrate the relationship between sig_0, PP and SSD, where the first graph shows the relationship between sig_0 and PP, the second graph shows sig_0 and SSD, and the third graph shows PP and SSD. 

```python
plt.scatter(data_cleaned[:,0],data_cleaned[:,1],c=clusters_gmm)
plt.xlabel("sig_0")
plt.ylabel("PP")
plt.show()
plt.scatter(data_cleaned[:,0],data_cleaned[:,2],c=clusters_gmm)
plt.xlabel("sig_0")
plt.ylabel("SSD")
plt.show()
plt.scatter(data_cleaned[:,1],data_cleaned[:,2],c=clusters_gmm)
plt.xlabel("PP")
plt.ylabel("SSD")
```
<img src="https://github.com/Guoxuan-Li/GEOL0069_Week4/blob/main/Images/Scatter.png?raw=true" />


<!-- Waveform Alignment Using Cross-Correlation -->
### Waveform Alignment Using Cross-Correlation
This code aligns waveforms in the cluster where clusters_gmm == 0 by using cross-correlation. And visualise a subset of 10 equally spaced waveforms.

```python
from scipy.signal import correlate
 
# Find the reference point (e.g., the peak)
reference_point_index = np.argmax(np.mean(waves_cleaned[clusters_gmm==0], axis=0))
 
# Calculate cross-correlation with the reference point
aligned_waves = []
for wave in waves_cleaned[clusters_gmm==0][::len(waves_cleaned[clusters_gmm == 0]) // 10]:
    correlation = correlate(wave, waves_cleaned[clusters_gmm==0][0])
    shift = len(wave) - np.argmax(correlation)
    aligned_wave = np.roll(wave, shift)
    aligned_waves.append(aligned_wave)
 
# Plot aligned waves
for aligned_wave in aligned_waves:
    plt.plot(aligned_wave)
 
plt.title('Plot of 10 equally spaced functions where clusters_gmm = 0 (aligned)')
```
<img src="https://github.com/Guoxuan-Li/GEOL0069_Week4/blob/main/Images/gmm=0.jpg?raw=true" width="500" height="400" />

<!-- Compare with ESA data -->
### Compare with ESA data
In the ESA dataset, sea ice = 1 and lead = 2. Therefore, we need to subtract 1 from it so our predicted labels are comparable with the official product labels. Finally, creat a confusion matrix and classification report to summarise the accuracy of classifications.

```python
flag_cleaned_modified = flag_cleaned - 1
```

This line of code, flag_cleaned_modified = flag_cleaned - 1, modifies the flag_cleaned array by subtracting 1 from each of its elements. The purpose of this operation is likely to adjust the labeling of data categories, ensuring that the values start from zero instead of one. This can be useful for compatibility with machine learning models that expect zero-based indexing or for standardizing the dataset before further processing. The modified array, flag_cleaned_modified, retains the same structure as flag_cleaned but with all values shifted down by one.


```python
from sklearn.metrics import confusion_matrix, classification_report

true_labels = flag_cleaned_modified   # true labels from the ESA dataset
predicted_gmm = clusters_gmm          # predicted labels from GMM method

# Compute confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted_gmm)

# Print confusion matrix
print("Confusion Matrix:")
print(conf_matrix)

# Compute classification report
class_report = classification_report(true_labels, predicted_gmm)

# Print classification report
print("\nClassification Report:")
print(class_report)
```

<img src="https://github.com/Guoxuan-Li/GEOL0069_Week4/blob/main/Images/matrix.jpg?raw=true" width="500" height="400" />

The results showed high accuracy, which illustrate GMM can effectively distinguish sea ice and leads.






