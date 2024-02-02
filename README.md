# Image Segmentation using K-Means Clustering

This Python script performs image segmentation using K-Means Clustering algorithm. Given an original image and a seed image indicating background and foreground regions, the script separates the image into foreground and background segments. It employs OpenCV, NumPy, Matplotlib, and SciPy libraries for image processing and clustering.

## Requirements

- Python 3.x
- OpenCV
- NumPy
- Matplotlib
- SciPy

## Usage

1. Ensure you have all the required libraries installed.
2. Place the original image (`image.png`) and the seed image (`image-stroke.png`) in the same directory as the script.
3. Adjust the value of `k` (desired number of clusters) in the script according to your preference.
4. Run the script.

## How It Works

1. **K-Means Clustering**: The script employs K-Means Clustering to group pixels into k clusters based on their RGB values. The centroids obtained from the clustering represent the dominant colors in the image.

2. **Seed Pixel Extraction**: Seed pixels are extracted from the seed image to identify background and foreground regions. This information is used to initialize the clusters.

3. **Pixel Classification**: Each pixel in the original image is classified as either foreground or background based on its likelihood of belonging to each cluster.

4. **Segmentation**: The segmented image is generated based on the classification results. Foreground pixels are displayed as white, while background pixels are displayed as black.

5. **Saving Results**: The segmented image is saved in a folder named after the chosen number of clusters (`k`) with the seed image's name and '.png' extension.

## Example

For example, if `k = 70`, the segmented image will be saved in a folder named `70` as `image-stroke.png`.

## Note

- Ensure the seed image is properly prepared to distinguish foreground and background regions,
the seed-image should be black, using red pixels as foreground, and blue as background.
- Adjusting the value of `k` may affect the segmentation results.

Feel free to experiment with different values of `k` to achieve the desired segmentation results!
