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
<img src="https://github.com/RoshMeh/GEOL0069---Artificial-Intelligence-For-Earth-Observation/blob/main/IMAGE1.png?raw=true" width="500" height="400" />


<!-- Gaussian Mixture Models (GMM) [Bishop and Nasrabadi, 2006] -->
### Gaussian Mixture Models (GMM)

<!-- Introduction to Gaussian Mixture Models -->
#### Introduction to Gaussian Mixture Models
Gaussian Mixture Models (GMM) is a probabilistic clustering technique that assumes data is generated from a mixture of several Gaussian distributions. Unlike K-Means, which assigns each data point to a single cluster, GMM provides a soft clustering approach, where each point has a probability of belonging to multiple clusters.


In remote sensing and EO applications, GMM is particularly effective in classifying environmental features that exhibit natural variability, such as distinguishing between sea ice and open water leads.

<!-- Why Gaussian Mixture Models for Clustering? -->
#### Why Use Gaussian Mixture Models for Clustering?
GMM is preferred over K-Means in scenarios where:

* Soft clustering is required: Instead of forcing data into rigid clusters, GMM provides a probabilistic classification, capturing uncertainty in the data.
* Flexibility in cluster covariance: Unlike K-Means, which assumes spherical clusters, GMM allows clusters to have different sizes, orientations, and shapes, making it more adaptable to real-world datasets.

<!-- Key Components of GMM -->
#### Key Components of GMM

The three fundamental components of GMM include:

* Number of Gaussian Components: The number of distributions (n_components) used to model the dataset.
* Expectation-Maximization (EM) Algorithm: The iterative algorithm that updates cluster assignments based on probability distributions.
* Covariance Type: Defines the shape, size, and orientation of clusters, providing flexibility in non-spherical data distributions.
<!-- The EM Algorithm in GMM -->
#### The Expectation-Maximization (EM) Algorithm
GMM relies on the EM algorithm, which iteratively refines cluster assignments:

* Expectation Step (E-step): Computes the probability that each data point belongs to each Gaussian component.
* Maximization Step (M-step): Updates the parameters of each Gaussian component (mean, variance, and mixing coefficients) to maximize the likelihood of the data fitting the model.

This iterative process continues until convergence, ensuring optimal clustering.

<!-- Advantages of GMM -->
#### Advantages of Gaussian Mixture Models in Remote Sensing
* Provides a probabilistic framework for clustering, allowing soft classification rather than hard assignments.
* Adapts to non-spherical clusters, making it more flexible for real-world applications.
* Captures data uncertainty, improving the interpretability of Earth Observation data classification.

<!-- Basic Code Implementation -->
#### Code
Using scikit-learn, we generate 100 random data points, apply GMM clustering, and visualize the results:

* Each point is color-coded based on its assigned probability distribution.
* The Gaussian means (centers) are marked with black dots.
* The output illustrates how GMM clusters data based on probability distributions rather than fixed boundaries.
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

<img src="https://github.com/RoshMeh/GEOL0069---Artificial-Intelligence-For-Earth-Observation/blob/main/IMAGE2.png?raw=true" width="500" height="400" />

<!-- Image Classification -->
### Image Classification
In this section, we apply unsupervised learning techniques—specifically K-Means Clustering and Gaussian Mixture Models (GMM)—to classify Sentinel-2 satellite images, focusing on distinguishing sea ice from open water leads.

<!-- K-Means Implementation -->
#### Image Classification Using K-Means Clustering
K-Means groups image pixels into k clusters based on spectral characteristics. In this implementation:

* Sentinel-2 bands (e.g., Red, Green, Blue) are extracted and stacked.
* Valid data points are filtered, removing noise and missing values.
* K-Means clustering segments the image into distinct regions (e.g., ice vs. water).
* The results are visualized, with different colors representing different clusters.

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

<img src="https://github.com/RoshMeh/GEOL0069---Artificial-Intelligence-For-Earth-Observation/blob/main/IMAGE3.png?raw=true" width="500" height="400" />

This visualization represents the K-Means clustering results applied to Sentinel-2 imagery. In the output:

* Yellow regions likely correspond to sea ice.
* Green regions represent other surface types.
* Purple areas indicate no-data zones, where valid satellite observations are unavailable.

<!-- GMM Implementation -->
#### GMM Implementation
Gaussian Mixture Models (GMM) classify image pixels by modeling the data as a mixture of Gaussian distributions, allowing for soft clustering where each pixel has a probability of belonging to multiple clusters.

In this:

* Sentinel-2 spectral bands (e.g., Red, Green, Blue) are extracted and stacked.
* Valid data points are filtered, removing noise and missing values.
* GMM clustering is applied to segment the image into distinct regions (e.g., sea ice vs. open water).
* The results are visualized, where different colors represent different cluster probabilities, providing a more flexible segmentation approach compared to K-Means.

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
<img src="https://github.com/RoshMeh/GEOL0069---Artificial-Intelligence-For-Earth-Observation/blob/main/IMAGE4.png?raw=true" width="500" height="400" />

This visualization depicts the GMM clustering results for Sentinel-2 imagery.

* Yellow regions likely correspond to sea ice.
* Green regions represent other surface types.
* Purple areas indicate no-data zones, where valid satellite observations are unavailable.

<!-- Altimetry Classification-->
### Altimetry Classification
In this section, we leverage unsupervised learning techniques to classify Sentinel-3 altimetry data, specifically distinguishing between sea ice and open leads based on satellite-derived elevation measurements.


By analyzing variations in surface elevation, this approach provides valuable insights into ice dynamics, oceanographic processes, and climate interactions. Accurate classification of sea ice and leads is essential for monitoring polar regions, assessing ice coverage, and understanding the impact of climate change on Arctic and Antarctic environments.

<!-- Read in Functions Needed -->
#### Data Preprocessing for Sentinel-3 Altimetry Classification
Before applying unsupervised learning models, the Sentinel-3 SAR altimetry data undergoes preprocessing to ensure accuracy and compatibility with analytical methods. The raw dataset is transformed into meaningful variables, including:

* Peak detection for identifying elevation changes.
* Stack Standard Deviations (SSDs) to assess variability.
* NaN value removal to clean and optimize the dataset.


Key libraries such as netCDF4, NumPy, Matplotlib, SciPy, Glob, and Scikit-learn are utilized to extract, process, and classify sea ice and leads. The processed data is then visualized, showcasing mean waveforms and standard deviations for each class, aiding in pattern recognition and classification.
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
<img src="https://github.com/RoshMeh/GEOL0069---Artificial-Intelligence-For-Earth-Observation/blob/main/IMAGE5.png?raw=true" width="500" height="400" />


```python
x = np.stack([np.arange(1,waves_cleaned.shape[1]+1)]*waves_cleaned.shape[0])
plt.plot(x,waves_cleaned)  # plot of all the echos
plt.show()
```

<img src="https://github.com/RoshMeh/GEOL0069---Artificial-Intelligence-For-Earth-Observation/blob/main/IMAGE6.png?raw=true" width="500" height="400" />


```python
# plot echos for the lead cluster
x = np.stack([np.arange(1,waves_cleaned[clusters_gmm==1].shape[1]+1)]*waves_cleaned[clusters_gmm==1].shape[0])
plt.plot(x,waves_cleaned[clusters_gmm==1])  # plot of all the echos
plt.show()
```
<img src="https://github.com/RoshMeh/GEOL0069---Artificial-Intelligence-For-Earth-Observation/blob/main/IMAGE7.png?raw=true" width="500" height="400" />


```python
# plot echos for the sea ice cluster
x = np.stack([np.arange(1,waves_cleaned[clusters_gmm==0].shape[1]+1)]*waves_cleaned[clusters_gmm==0].shape[0])
plt.plot(x,waves_cleaned[clusters_gmm==0])  # plot of all the echos
plt.show()
```
<img src="https://github.com/RoshMeh/GEOL0069---Artificial-Intelligence-For-Earth-Observation/blob/main/IMAGE8.png?raw=true" width="500" height="400" />


<!-- Scatter Plots of Clustered Data -->
### Scatter Plots of Clustered Data
The clustering results are visualized using scatter plots, where each color represents a distinct cluster. These plots provide a clear representation of relationships between key variables, helping to distinguish patterns in the data.

The three graphs illustrate:

* The relationship between sig_0 and PP – highlighting variations in surface reflectivity and peak power.
* The relationship between sig_0 and SSD – capturing changes in signal strength and surface roughness.
* The relationship between PP and SSD – revealing patterns in peak power distribution and standard deviation.


These visualizations offer valuable insights into how clusters are formed, aiding in the classification of sea ice and open leads from Sentinel-3 altimetry data.. 

```python
cluster_labels = {0: "Sea Ice", 1: "Leads"}
colors = {0: "cyan", 1: "magenta"}
features = [(0, 1, r"$\sigma_0$ vs PP", r"$\sigma_0$", "PP"),
            (0, 2, r"$\sigma_0$ vs SSD", r"$\sigma_0$", "SSD"),
            (1, 2, "PP vs SSD", "PP", "SSD")]
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=False, sharey=False)
fig.suptitle("Scatter Plots of Clustered Data", fontsize=16, fontweight='bold')
for idx, (x_idx, y_idx, title, xlabel, ylabel) in enumerate(features):
    ax = axes[idx]
    for cluster, color in colors.items():
        mask = clusters_gmm == cluster
        ax.scatter(data_cleaned[mask, x_idx], data_cleaned[mask, y_idx], 
                   c=color, alpha=0.5, label=cluster_labels[cluster], edgecolors='k', s=30)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.text(0.05, 0.9, f"({chr(97 + idx)})", transform=ax.transAxes, fontsize=14, fontweight='bold')
    ax.legend()
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

```
<img src="https://github.com/RoshMeh/GEOL0069---Artificial-Intelligence-For-Earth-Observation/blob/main/IMAGE9.png?raw=true" />


<!-- Waveform Alignment Using Cross-Correlation -->
### Waveform Alignment Using Cross-Correlation
This process aligns waveforms within the cluster where clusters_gmm == 0 using cross-correlation, ensuring consistency in signal patterns. By aligning the waveforms, we enhance the clarity of structural features, making classification more effective.

A subset of 10 equally spaced waveforms is selected and visualized, providing a clearer representation of the underlying signal characteristics within the identified cluster.


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
<img src="https://github.com/RoshMeh/GEOL0069---Artificial-Intelligence-For-Earth-Observation/blob/main/IMAGE10.png?raw=true" width="500" height="400" />

<!-- Compare with ESA data -->
### Compare with ESA data
In the European Space Agency (ESA) dataset, the labeling convention assigns:

* Sea ice = 1
* Leads = 2
To ensure consistency and comparability with our predicted classifications, we adjust the labels by subtracting 1, aligning them with the official ESA product labels.

To evaluate the accuracy and performance of our classification model, we generate:

* A Confusion Matrix to compare predicted vs. actual classifications.
* A Classification Report summarizing key metrics such as precision, recall, and F1-score, providing insights into the effectiveness of the clustering approach.

```python
flag_cleaned_modified = flag_cleaned - 1
```

This line of code, flag_cleaned_modified = flag_cleaned - 1, adjusts the flag_cleaned array by subtracting 1 from each element. The purpose of this operation is likely to standardise the labelling of data categories, ensuring that the values start from zero instead of one. This can be useful for compatibility with machine learning models that require zero-based indexing or for maintaining consistency in the dataset before further processing. The modified array, flag_cleaned_modified, retains the same structure as flag_cleaned, but with all values shifted down by one.


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

<img src="https://github.com/RoshMeh/GEOL0069---Artificial-Intelligence-For-Earth-Observation/blob/main/IMAGE11.png?raw=true" width="500" height="400" />

The confusion matrix results indicate high classification accuracy, demonstrating that Gaussian Mixture Models (GMM) can effectively distinguish between sea ice and open leads. This confirms the model’s ability to accurately cluster environmental features based on Sentinel-3 altimetry data.






