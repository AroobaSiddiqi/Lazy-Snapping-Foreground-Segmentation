import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import time
from scipy.spatial.distance import cdist
import os

'''-------------------------------------------------------------------------------------------------------------------'''

def k_means_clustering(pixels: np.array, k: int):
#, *seed_pixels):
    
  """
  Args:
    pixels: RGB pixels in an array of lists.
    k: The desired number of clusters.

  Returns:
    A cluster index for each input pixels and the k cluster centroids.
  """

  # Initialize the cluster centroids randomly.
  if not isinstance(pixels, np.ndarray):
    pixels = np.array(pixels)

  # if not seed_pixels:
  #   centroids = np.random.randint(0, 255, (k, 3))
  # else:
  #   centroids = np.array(seed_pixels)[0]
  
  indices = np.random.choice(len(pixels), k, replace=False)
  centroids = pixels[indices]
  
  #centroids = pixels[:k, :]

  centroids_old = np.empty((k,3))

  cluster_assignments = np.empty((pixels.shape[0]), dtype=int)

  # Iterate until the centroids no longer change.
  while True:

    for index, pixel in enumerate(pixels):
                  
        # Calculate eucledian distances of pixel from each centroid.
        distances = [np.linalg.norm(np.array(pixel) - np.array(centroid)) for centroid in centroids]

        # Assign pixel to cluster(a value from 0 - k) with the closest centroid.
        cluster_assignments[index] = np.argmin(distances)
  
    # Re-calculating centroids.
    for k in range(k):
      if k in cluster_assignments:
        mask = (cluster_assignments == k)
        cluster = pixels[mask]
        centroids[k] = np.mean(cluster)

    # Check if the centroids have changed.
    if np.all(centroids == centroids_old):
      break

    centroids_old = centroids

  return cluster_assignments, centroids



def get_seed_pixels(seed_image: str, original_image: str):

  '''
  Args: 
    seed_image: path of seed image
    original_image: path of original image
  
  Returns:
    Colours of background and foreground seed pixels in RGB format.
  '''
  
  seed = cv2.imread(seed_image)
  hsv_seed = cv2.cvtColor(seed, cv2.COLOR_BGR2HSV)
  seed_pixels = np.array(hsv_seed).reshape(-1, 3)
  
  # Create blue mask, 100 and 150 are bounds for blue hue.
  blue_mask = np.logical_and.reduce((seed_pixels[:, 0] >= 100, seed_pixels[:, 0] <= 150))
  red_mask = np.logical_and.reduce((seed_pixels[:, 0] >= 0, seed_pixels[:, 0] <= 20))

  # Get indices of seed pixels
  background_indices = np.where(blue_mask)[0]
  foreground_indices = np.where(red_mask)[0]

  original = cv2.imread(original_image)

  # Reshape the original_image into a 2D array
  original_pixels = np.array(original).reshape(-1, 3)

  background_seed = []
  foreground_seed = []

  for index_b in background_indices:
    background_seed.append(original_pixels[index_b].tolist())

  for index_f in foreground_indices:
    foreground_seed.append(original_pixels[index_f].tolist())

  return background_seed, foreground_seed




def get_likelihood(pixel, cluster_assignments, centroids):

  ''' 
  Args:
    pixel: Input pixel in RGB format.
    cluster_assignments: A cluster index for each image pixel.
    centroids : List of RGB centroids for each cluster.

  Returns:
  The likelihood of a given pixel belonging to one of the clusters represented by centroids.
  
  '''
  
  counts = np.bincount(cluster_assignments, minlength=len(centroids))
  weights = counts / len(cluster_assignments)
  temp = np.expand_dims(pixel, axis=0)
  distances = cdist(centroids, temp, metric='euclidean')
  values = weights*np.exp(-distances)
  likelihood = np.sum(values)
  
  return likelihood


def classify_pixel(pixel, fg_cluster_assignments, fg_centroids, bg_cluster_assignments, bg_centroids):

  '''  
  Args:
    pixel: Input pixel in RGB format.
    fg_cluster_assignments : A cluster index for each foreground pixel.
    fg_centroids : List of RGB centroids for each foreground cluster.
    bg_cluster_assignments : A cluster index for each background pixel.
    bg_centroids : List of RGB centroids for each background cluster.

  Returns:
  Classification of a single pixel into foreground(1) or background(0).

  '''

  fg_sum = get_likelihood(pixel, fg_cluster_assignments, fg_centroids)
  bg_sum = get_likelihood(pixel, bg_cluster_assignments, bg_centroids)

  if bg_sum > fg_sum :
    group = 0 # background
  elif bg_sum < fg_sum :
    group = 1 # foreground
  
  return group


def get_segmented_image(original:str, seed:str, k:int=10):

  '''  
  Args:
  original : Original image path as string.
  seed : Seed image path as string.
  k : Desired number of clusters for foreground and background.
  
  Returns:
  Segmented image with background as black and foreground as white.
  Saves segmented image into a 'k' folder as seed image's name '.png'.

  '''

  # start1 = time.time()   
  bg_pixels, fg_pixels = get_seed_pixels(seed, original)
  # end1 = time.time()
  # t1 = end1 - start1
  # print(f'get_seed_pixels: {t1}')

  # start2 = time.time() 
  bg_cluster_assignments, bg_centroids = k_means_clustering(bg_pixels,k)
  fg_cluster_assignments, fg_centroids = k_means_clustering(fg_pixels,k)
  # end2 = time.time()
  # t2 = end2 - start2
  # print(f'k_means_clustering x 2 : {t2}')
  
  original = cv2.imread(original)
  height, width, _ = original.shape
  original = np.array(original).reshape(-1, 3)
  

  # start3 = time.time() 
  segmented_image = [classify_pixel(pixel, fg_cluster_assignments, fg_centroids, bg_cluster_assignments, bg_centroids) for pixel in original]
  # end3 = time.time()
  # t3 = end3 - start3
  # print(f'classify_pixel : {t3}')

  
  # Define the color mapping
  colors = ['black', 'white']

  # Create an image from the list of labels
  image = np.reshape(segmented_image, (height, width))

  # Create a custom colormap using the defined colors
  cmap = mcolors.ListedColormap(colors)

  # Plot the image
  plt.imshow(image, cmap=cmap)
  plt.axis('off')

  k_folder = str(k)  # Convert k to string if it's an integer

  # Create the folder if it doesn't exist
  if not os.path.exists(k_folder):
    os.makedirs(k_folder)

  # Save the file
  plt.savefig(f'{k_folder}/{seed}.png')

  # Show the segmented image
  plt.show()
    




if __name__ == "__main__":

  k = 70

  original = 'image.png'
  seed = 'image-stroke.png'
  get_segmented_image(original,seed,k)     