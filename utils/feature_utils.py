import numpy as np
import cv2
import mahotas
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import regionprops
import pandas as pd

def extract_texture_features(image):
    """
    Extract texture features using GLCM (Gray Level Co-occurrence Matrix)
    """
    # Convert to grayscale if needed
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Calculate GLCM
    glcm = graycomatrix(image, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], 256, symmetric=True, normed=True)
    
    # Calculate GLCM properties
    contrast = graycoprops(glcm, 'contrast').mean()
    dissimilarity = graycoprops(glcm, 'dissimilarity').mean()
    homogeneity = graycoprops(glcm, 'homogeneity').mean()
    energy = graycoprops(glcm, 'energy').mean()
    correlation = graycoprops(glcm, 'correlation').mean()
    
    # Calculate Haralick features
    haralick = mahotas.features.haralick(image).mean(axis=0)
    
    return {
        'contrast': contrast,
        'dissimilarity': dissimilarity,
        'homogeneity': homogeneity,
        'energy': energy,
        'correlation': correlation,
        'haralick_mean': haralick.mean(),
        'haralick_std': haralick.std()
    }

def extract_shape_features(image):
    """
    Extract shape features from the image
    """
    # Convert to binary image
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return {
            'area': 0,
            'perimeter': 0,
            'circularity': 0,
            'aspect_ratio': 0
        }
    
    # Get the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Calculate shape features
    area = cv2.contourArea(largest_contour)
    perimeter = cv2.arcLength(largest_contour, True)
    circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
    
    # Calculate aspect ratio
    x, y, w, h = cv2.boundingRect(largest_contour)
    aspect_ratio = float(w)/h if h > 0 else 0
    
    return {
        'area': area,
        'perimeter': perimeter,
        'circularity': circularity,
        'aspect_ratio': aspect_ratio
    }

def extract_statistical_features(image):
    """
    Extract statistical features from the image
    """
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    return {
        'mean': np.mean(image),
        'std': np.std(image),
        'skewness': np.mean((image - np.mean(image))**3) / (np.std(image)**3),
        'kurtosis': np.mean((image - np.mean(image))**4) / (np.std(image)**4),
        'entropy': -np.sum(np.histogram(image, bins=256)[0] * np.log2(np.histogram(image, bins=256)[0] + 1e-7))
    }

def extract_all_features(image):
    """
    Extract all features from the image
    """
    texture_features = extract_texture_features(image)
    shape_features = extract_shape_features(image)
    statistical_features = extract_statistical_features(image)
    
    # Combine all features
    all_features = {**texture_features, **shape_features, **statistical_features}
    return pd.Series(all_features) 