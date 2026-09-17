""""
Insert plot screenshot, returns raw data

Code written using Gemini Thinking Mode 3.6 on 16.09.2026
"""

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.widgets import RectangleSelector



def digitise_spectrum(image_path, plot_area_pixels, axis_limits, output_csv="digitised.csv"):
    """
    Extracts data points from an image of a spectrum plot.
    
    :param image_path: Path to the screenshot.
    :param plot_area_pixels: Tuple of (x_min, x_max, y_min, y_max) defining the plot box in pixels.
    :param axis_limits: Tuple of (x_val_min, x_val_max, y_val_min, y_val_max) defining real-world axis values.
    :param output_csv: Path to save the extracted data.
    """
    output_csv = image_path[0:-4] + "_digitised.csv"

    px_xmin, px_xmax, px_ymin, px_ymax = plot_area_pixels
    val_xmin, val_xmax, val_ymin, val_ymax = axis_limits
    
    # 1. Load the image and crop it to the plot area
    img = cv2.imread(image_path)   # combines the path and name strings to get the image path name
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
    df = pd.DataFrame(data_points, columns=["Energy in eV", "Intensity"])
    df.to_csv(output_csv, index=False)
    print(f"Data successfully saved to {output_csv}")
    
    # Plot the extracted data to verify accuracy
    plt.figure(figsize=(8, 4))
    plt.plot(df["Energy in eV"], df["Intensity"], color='red', label="Extracted Data")
    plt.xlim(val_xmin, val_xmax)
    plt.ylim(val_ymin, val_ymax)
    plt.xlabel("Energy in eV")
    plt.ylabel("Intensity")
    plt.title("Digitised Spectrum")
    plt.grid(True)
    plt.legend()
    plt.show()



def onselect(eclick, erelease):
    """Callback triggered whenever a rectangle selection is updated."""
    x1, y1 = eclick.xdata, eclick.ydata
    x2, y2 = erelease.xdata, erelease.ydata

    left = min(x1, x2)
    right = max(x1, x2)
    top = min(y1, y2)     # In image coordinates, Y=0 is at the top
    bottom = max(y1, y2)  # Larger Y pixel index is lower on screen

    print(f"\nSelection Bounds:")
    print(f"  Left (X min):   {left:.2f} px")
    print(f"  Right (X max):  {right:.2f} px")
    print(f"  Top (Y top):    {top:.2f} px")
    print(f"  Bottom (Y bot): {bottom:.2f} px")



def on_key_press(event):
    """Closes the figure window when ENTER (or Return) is pressed."""
    if event.key in ['enter', 'return']:
        plt.close(event.canvas.figure)



def extract_rect(image_path):
    # 1. Load image and display plot
    img = mpimg.imread(image_path)
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(img)
    ax.grid(True, color='k', linestyle='--', linewidth=.2)
    ax.set_title("Click & drag to draw plot boundaries. \nAdjust corner handles as needed, then close the window to confirm.")

    # 2. Attach RectangleSelector
    rect_selector = RectangleSelector(
        ax, 
        onselect,
        useblit=True,
        button=[1],              # Left mouse button only
        minspanx=5, minspany=5,  # Ignore accidental tiny clicks
        props=dict(edgecolor='red', facecolor='red', alpha=0.2, fill=True),
        interactive=True         # Keeps box active to drag edges/corners
    )

    plt.show()

    # 3. Connect key press listener for ENTER key
    fig.canvas.mpl_connect('key_press_event', on_key_press)

    # Show plot (script execution pauses here until window is closed or ENTER is pressed)
    plt.show()

    # 4. Extract final selection coordinates after window closes
    xmin, xmax, ymin, ymax = rect_selector.extents
    print("\n" + "="*40)
    print(f"FINAL BOUNDS CONFIRMED:")
    print(f"  (X min, X max, Y top, Y bottom) = ({xmin:.1f}, {xmax:.1f}, {ymin:.1f}, {ymax:.1f})")
    print("="*40)

    return (int(xmin), int(xmax), int(ymin), int(ymax))



def extract_points(image_path):
    """
    Proper axis labelling: Extracts known data points (intensity, energy) from spectrum screenshot in matplotlib environment

    returns calibrated_data as [px, py, energy, intensity]
    """
    # 1. Load and display the image
    img = mpimg.imread(image_path)  # Replace with your image file

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(img)
    ax.grid(True, color='k', linestyle='--', linewidth=.2)
    plt.title("Left-click known points (e.g., axes corners or known peaks).\nRight-click to remove last point. Press ENTER when finished.")

    # 2. Capture pixel points interactively (show_clicks=True draws red markers)
    pixel_points = plt.ginput(n=-1, timeout=0, show_clicks=True)
    plt.close(fig)

    # 3. Associate clicked pixel points with real-world Energy & Intensity values
    calibrated_data = []

    print("\n--- Calibration Input ---")
    for i, (px, py) in enumerate(pixel_points):
        print(f"\nPoint {i+1} at Pixel (X={px:.2f}, Y={py:.2f})")
        energy = float(input("  Enter known Energy (X-value): "))
        intensity = float(input("  Enter known Intensity (Y-value): "))
        calibrated_data.append([px, py, energy, intensity])

    print("\nCollected Calibration Points:", calibrated_data)

    return calibrated_data



def linear_interpolation(pixel_bounds, calibrated_data):
    """
    Adapts the axis values to the true values evaluated from 'extract_points()'
    
    :param pixel_bounds: Tuple of (x_min_px, x_max_px, y_top_px, y_bottom_px)
    :param calibrated_data: List of 2 reference points [[px0, py0, e0, i0], [px1, py1, e1, i1]]
    :return: Tuple of (energy_min, energy_max, intensity_min, intensity_max)
    """
    x_min_px, x_max_px, y_top_px, y_bottom_px = pixel_bounds
    (px0, py0, e0, i0), (px1, py1, e1, i1) = calibrated_data[:2]

    # 1. Calculate signed slopes (Delta Physical / Delta Pixel)
    slope_x = (e1 - e0) / (px1 - px0)
    slope_y = (i1 - i0) / (py1 - py0)  # Naturally negative due to image Y-inversion

    # 2. Linear projection for X (Energy)
    e_at_xmin = e0 + slope_x * (x_min_px - px0)
    e_at_xmax = e0 + slope_x * (x_max_px - px0)

    # 3. Linear projection for Y (Intensity)
    i_at_ytop = i0 + slope_y * (y_top_px - py0)
    i_at_ybottom = i0 + slope_y * (y_bottom_px - py0)

    # 4. Standardize output into (E_min, E_max, I_min, I_max)
    emin, emax = min(e_at_xmin, e_at_xmax), max(e_at_xmin, e_at_xmax)
    imin, imax = min(i_at_ytop, i_at_ybottom), max(i_at_ytop, i_at_ybottom)

    return (emin, emax, imin, imax)



# ==========================================
# USER CONFIGURATION
# ==========================================
if __name__ == "__main__":
    # 1. Path to your screenshot
    IMAGE_PATH = "Images\Plot Digitising\Dummy_Spectrum.png"
    
    # 2. Extract the rectangular bounds of the spectrum
    # Format: (x_start, x_end, y_top, y_bottom) in pixels
    
    #PIXEL_BOUNDS = (183, 3028, 17, 410)

    PIXEL_BOUNDS = extract_rect(IMAGE_PATH)
    print('\n pixel bounds: ', PIXEL_BOUNDS, '\n')
    
    # 3. What do the edges of that pixel box represent in real units?
    # Format: (Wavelength_min, Wavelength_max, Absorbance_min, Absorbance_max)
    
    #AXIS_VALUES = (712, 782, 0.0, 0.2)

    calibrated_data = extract_points(IMAGE_PATH)
    AXIS_VALUES = linear_interpolation(PIXEL_BOUNDS, calibrated_data)
    
    # Run the extractor
    digitise_spectrum(IMAGE_PATH, PIXEL_BOUNDS, AXIS_VALUES)