import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import tifffile as tiff


# ==========================================
# USER CONFIGURATION
# ==========================================

BASE_PATH = (
    "/Users/moritz/Library/Mobile Documents/"
    "com~apple~CloudDocs/Studium/Semester 5/"
    "Schlussprojekt/"
)

CSV_PATH = os.path.join(BASE_PATH, "roi_bounds.csv")

MONOCHROMATOR_DIRS = [
    os.path.join(BASE_PATH, "monochromator v1"),
    os.path.join(BASE_PATH, "monochromator v2"),
    os.path.join(BASE_PATH, "monochromator v3"),
]


# ==========================================
# FIND SPECTRUM FOLDER
# ==========================================

def find_spectrum_folder(name):
    """
    Find the spectrum folder corresponding to the four-digit
    identifier stored in the CSV.

    The original ROI script used:
        name = datafolders[i][9:13]

    Therefore this function uses exactly the same convention.
    """

    matches = []

    for mono_dir in MONOCHROMATOR_DIRS:

        if not os.path.isdir(mono_dir):
            continue

        # Each direct subfolder is one spectrum
        for folder in os.listdir(mono_dir):

            folder_path = os.path.join(mono_dir, folder)

            if not os.path.isdir(folder_path):
                continue

            # Same identification scheme as original script
            if len(folder) >= 13 and folder[9:13] == name:
                matches.append(folder_path)

    if len(matches) == 0:
        return None

    if len(matches) > 1:
        print(f"\nWARNING: '{name}' was found multiple times:")
        for match in matches:
            print("   ", match)

    return matches[0]


# ==========================================
# LOAD SPECTRUM
# ==========================================

def load_spectrum(folder):

    datafiles = sorted(os.listdir(folder))

    tiff_files = [
        filename
        for filename in datafiles
        if filename.lower().endswith((".tiff", ".tif"))
    ]

    if not tiff_files:
        return None

    data = []

    for filename in tiff_files:

        filepath = os.path.join(folder, filename)

        data.append(
            tiff.imread(filepath).astype(float)
        )

    data = np.array(data)

    # Same processing as your original script
    img = np.sum(data, axis=0)

    img[:, :10] = 0

    max_value = np.max(img)

    if max_value != 0:
        img /= max_value

    return img


# ==========================================
# READ CSV
# ==========================================

def load_roi_csv(csv_path):

    rows = []

    with open(csv_path, "r", newline="") as f:

        reader = csv.DictReader(f)

        for row in reader:

            rows.append({
                "name": row["folder"],

                "bg": (
                    int(float(row["bg_xmin"])),
                    int(float(row["bg_xmax"])),
                    int(float(row["bg_ymin"])),
                    int(float(row["bg_ymax"]))
                ),

                "particle": (
                    int(float(row["particle_xmin"])),
                    int(float(row["particle_xmax"])),
                    int(float(row["particle_ymin"])),
                    int(float(row["particle_ymax"]))
                )
            })

    return rows


# ==========================================
# DRAW ROI
# ==========================================

def draw_roi(ax, bounds, color, label):

    xmin, xmax, ymin, ymax = bounds

    rect = Rectangle(
        (xmin, ymin),
        xmax - xmin,
        ymax - ymin,
        linewidth=2,
        edgecolor=color,
        facecolor=color,
        alpha=0.25,
        label=label
    )

    ax.add_patch(rect)


# ==========================================
# INTERACTIVE VIEWER
# ==========================================

class ROIViewer:

    def __init__(self, rows):

        self.rows = rows
        self.index = 0

        self.fig, self.ax = plt.subplots(
            figsize=(12, 9)
        )

        self.fig.canvas.mpl_connect(
            "key_press_event",
            self.on_key
        )

        self.show_current()


    def show_current(self):

        self.ax.clear()

        row = self.rows[self.index]

        name = row["name"]

        print("\n" + "=" * 70)
        print(
            f"Spectrum {self.index + 1} / {len(self.rows)}"
        )
        print(f"Identifier: {name}")

        # ------------------------------------------
        # Find folder
        # ------------------------------------------

        folder = find_spectrum_folder(name)

        if folder is None:

            print("ERROR: Spectrum folder not found!")

            self.ax.text(
                0.5,
                0.5,
                f"Spectrum folder not found:\n\n{name}",
                transform=self.ax.transAxes,
                ha="center",
                va="center",
                fontsize=16
            )

            self.ax.set_title(
                f"{name} — NOT FOUND"
            )

            self.fig.canvas.draw_idle()

            return

        print(f"Folder: {folder}")

        # ------------------------------------------
        # Load spectrum
        # ------------------------------------------

        img = load_spectrum(folder)

        if img is None:

            print("ERROR: No TIFF files found!")

            self.ax.text(
                0.5,
                0.5,
                f"No TIFF files found:\n\n{folder}",
                transform=self.ax.transAxes,
                ha="center",
                va="center",
                fontsize=14
            )

            self.fig.canvas.draw_idle()

            return

        print(f"Image shape: {img.shape}")

        # ------------------------------------------
        # Display image
        # ------------------------------------------

        self.ax.imshow(
            img,
            cmap="gray",
            origin="upper"
        )

        # ------------------------------------------
        # Background ROI
        # ------------------------------------------

        draw_roi(
            self.ax,
            row["bg"],
            "blue",
            "Background"
        )

        # ------------------------------------------
        # Particle ROI
        # ------------------------------------------

        draw_roi(
            self.ax,
            row["particle"],
            "red",
            "Particle"
        )

        # ------------------------------------------
        # Plot formatting
        # ------------------------------------------

        self.ax.set_title(
            f"{name}   ({self.index + 1}/{len(self.rows)})\n"
            "← previous   |   → / Enter = next   |   Esc = quit"
        )

        self.ax.set_xlabel("X [px]")
        self.ax.set_ylabel("Y [px]")

        self.ax.legend(
            loc="upper right"
        )

        self.fig.tight_layout()

        self.fig.canvas.draw_idle()


    # ======================================
    # KEYBOARD CONTROLS
    # ======================================

    def on_key(self, event):

        if event.key == "escape":

            plt.close(self.fig)

        elif event.key in [
            "right",
            "enter",
            "return",
            "space"
        ]:

            if self.index < len(self.rows) - 1:

                self.index += 1
                self.show_current()

        elif event.key in [
            "left",
            "backspace"
        ]:

            if self.index > 0:

                self.index -= 1
                self.show_current()


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":

    rows = load_roi_csv(CSV_PATH)

    print(
        f"Loaded {len(rows)} ROI entries from:"
    )
    print(CSV_PATH)

    viewer = ROIViewer(rows)

    plt.show()