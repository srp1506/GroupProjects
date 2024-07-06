from PIL import Image
import os
import numpy as np

def calculate_mean_std(dataset_path):
    # Initialize variables to accumulate sums
    mean_sum = np.zeros(3)
    std_sum = np.zeros(3)
    total_images = 0

    # Iterate through the dataset
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            # Get the image path
            image_path = os.path.join(root, file)
            
            try:
                # Open the image using Pillow
                image = Image.open(image_path).convert('RGB')
                
                # Convert the image to a NumPy array
                image_array = np.array(image)
                
                # Compute the mean and std for each channel
                mean_sum += np.mean(image_array, axis=(0, 1))
                std_sum += np.std(image_array, axis=(0, 1))
                
                # Increment the total number of images
                total_images += 1
            except Exception as e:
                print(f"Skipping {image_path} due to error: {e}")

    # Calculate the overall mean and std
    overall_mean = mean_sum / total_images
    overall_std = std_sum / total_images

    return overall_mean, overall_std

# Specify the path to your dataset
dataset_path = 'dataset'

# Calculate mean and std
mean, std = calculate_mean_std(dataset_path)

# Print the results
print(f'Mean: {mean}')
print(f'Std: {std}')

# Convert to a format suitable for transforms.Normalize
mean_for_transform = tuple(mean / 255.0)
std_for_transform = tuple(std / 255.0)

print(f'Mean for transforms.Normalize: {mean_for_transform}')
print(f'Std for transforms.Normalize: {std_for_transform}')

