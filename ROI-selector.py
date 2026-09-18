import csv
import cv2
import numpy as np
import pandas as pd
import tifffile as tiff
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.widgets import RectangleSelector

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



def extract_rect(img, name):
    # 1. Load image and display plot
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(img)
    ax.grid(True, color='k', linestyle='--', linewidth=.2)
    ax.set_title("Click & drag to draw plot boundaries. \nAdjust corner handles as needed, then close the window to confirm. Dataset name: " + name)

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





# ==========================================
# USER CONFIGURATION
# ==========================================
if __name__ == "__main__":

    ### IMPORT DATA
    datapath = "/Users/moritz/Library/Mobile Documents/com~apple~CloudDocs/Studium/Semester 5/Schlussprojekt/monochromator v1/"

    datafolders = [
        name for name in os.listdir(datapath)
        if os.path.isdir(os.path.join(datapath, name))
    ]

    # CSV file
    csv_path = os.path.join("/Users/moritz/Library/Mobile Documents/com~apple~CloudDocs/Studium/Semester 5/Schlussprojekt/roi_bounds.csv")

    file_exists = os.path.exists(csv_path)

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "folder",
                "bg_xmin", "bg_xmax", "bg_ymin", "bg_ymax",
                "particle_xmin", "particle_xmax", "particle_ymin", "particle_ymax"
            ])

        # Create figure with one plot
        fig, ax = plt.subplots(figsize=(6,6))

        for i in range(len(datafolders)):
            path = os.path.join(datapath, datafolders[i])
            datafiles = sorted(os.listdir(path))
            print(datafolders[i][9:13])
            if len(datafiles) < 5:
                print("scan aborted, continuing")
                continue

            data = []

            for j in range(len(datafiles)):
                if datafiles[j].endswith(".tiff"):
                    data.append(
                        tiff.imread(
                            os.path.join(path, datafiles[j])
                        ).astype(float)
                    )

            data = np.array(data)
            img = np.sum(data, axis = 0)
            img[:, :10] = 0
            img /= np.max(img)


            name = datafolders[i][9:13]
            BG_BOUNDS = extract_rect(img, name + " (background)")
            PARTICLE_BOUNDS = extract_rect(img, name + " (particle)")


            #print('\n pixel bounds: ', PIXEL_BOUNDS, '\n')
            
            writer.writerow([
                        name,
                        *BG_BOUNDS,
                        *PARTICLE_BOUNDS
                    ])

