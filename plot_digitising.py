""""
Insert plot screenshot, returns raw data

Code written using Gemini Pro 3.1 ono 16.09.2026
"""

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def digitize_spectrum(image_path, plot_area_pixels, axis_limits, output_csv="spectrum_data.csv"):
    """
    Extracts data points from an image of a spectrum plot.
    
    :param image_path: Path to the screenshot.
    :param plot_area_pixels: Tuple of (x_min, x_max, y_min, y_max) defining the plot box in pixels.
    :param axis_limits: Tuple of (x_val_min, x_val_max, y_val_min, y_val_max) defining real-world axis values.
    :param output_csv: Path to save the extracted data.
    """
    px_xmin, px_xmax, px_ymin, px_ymax = plot_area_pixels
    val_xmin, val_xmax, val_ymin, val_ymax = axis_limits
    
    # 1. Load the image and crop it to the plot area
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image not found. Check the file path.")
    
    cropped_img = img[px_ymin:px_ymax, px_xmin:px_xmax]
    
    # 2. Convert to grayscale and apply a threshold to isolate the line
    # Assuming a dark line on a light background. 
    # If your line is colored, HSV color filtering would be better here.
    gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    
    # 3. Extract the line coordinates
    # For every x (column), find the y (row) where the line exists
    data_points = []
    
    height, width = thresh.shape
    for x_px in range(width):
        # Find all y pixels in this column that are part of the line
        y_pixels = np.where(thresh[:, x_px] > 0)[0]
        
        if len(y_pixels) > 0:
            # If the line is thick, average the y-coordinates for a single point
            y_px = np.mean(y_pixels)
            
            # 4. Map pixel coordinates to real-world values
            # X mapping (linear interpolation)
            x_val = val_xmin + (x_px / width) * (val_xmax - val_xmin)
            
            # Y mapping (Note: image Y goes top-to-bottom, so we invert it)
            y_fraction = (height - y_px) / height
            y_val = val_ymin + y_fraction * (val_ymax - val_ymin)
            
            data_points.append([x_val, y_val])
            
    # 5. Save and visualize
    df = pd.DataFrame(data_points, columns=["Wavelength", "Absorbance"])
    df.to_csv(output_csv, index=False)
    print(f"Data successfully saved to {output_csv}")
    
    # Plot the extracted data to verify accuracy
    plt.figure(figsize=(8, 4))
    plt.plot(df["Wavelength"], df["Absorbance"], color='red', label="Extracted Data")
    plt.xlim(val_xmin, val_xmax)
    plt.ylim(val_ymin, val_ymax)
    plt.xlabel("Wavelength")
    plt.ylabel("Absorbance")
    plt.title("Digitized Spectrum")
    plt.grid(True)
    plt.legend()
    plt.show()

# ==========================================
# USER CONFIGURATION
# ==========================================
if __name__ == "__main__":
    # 1. Path to your screenshot
    IMAGE_FILE = "spectrum.png"
    
    # 2. Find these using MS Paint, Preview, or a basic image viewer.
    # Look at the coordinates of the bottom-left and top-right corners of the graph box.
    # Format: (x_start, x_end, y_top, y_bottom) in pixels
    PIXEL_BOUNDS = (100, 800, 50, 450) 
    
    # 3. What do the edges of that pixel box represent in real units?
    # Format: (Wavelength_min, Wavelength_max, Absorbance_min, Absorbance_max)
    AXIS_VALUES = (300, 800, 0.0, 1.5) 
    
    # Run the extractor
    digitize_spectrum(IMAGE_FILE, PIXEL_BOUNDS, AXIS_VALUES)