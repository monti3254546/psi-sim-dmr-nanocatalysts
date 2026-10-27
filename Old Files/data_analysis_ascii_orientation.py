#CALCULATES THERMAL CONDUCTIVITY FROM A TEMPERATURE MAP. parameter für den fit müssen für die Kamera angepasst werden
import numpy as np
import matplotlib.pyplot as plt
import cv2

filename = '/Users/moritz/Library/Mobile Documents/com~apple~CloudDocs/Studium/Semester 4/Projekte und Praktika/thin film thermal conductivity/19_3_2026_2 good.txt'
temp = np.loadtxt(filename)

# Optional crop FIRST
# temp = temp[10:290, 115:475]

(h, w) = temp.shape[:2]
center = (w // 2, h // 2)


fig, ax = plt.subplots()
im = ax.imshow(temp, aspect='equal', cmap='inferno')

# Correct colorbar usage
cbar = fig.colorbar(im, ax=ax, label='Temperature (K)')

ax.set_xlabel('z (through-thickness)')
ax.set_ylabel('x (lateral)')
ax.set_title('Measured Temperature Map')

points = []
corrected_array = None  # hier wird das Ergebnis gespeichert

def onclick(event):
    global corrected_array
    if event.xdata is None or event.ydata is None:
        return

    points.append((event.xdata, event.ydata))
    ax.plot(event.xdata, event.ydata, 'ro')
    fig.canvas.draw()

    if len(points) == 4:
        points_np = np.array(points, dtype=np.float32)

        # Breite/Höhe bestimmen
        width_top = np.linalg.norm(points_np[1] - points_np[0])
        width_bottom = np.linalg.norm(points_np[2] - points_np[3])
        width = int(max(width_top, width_bottom))

        height_left = np.linalg.norm(points_np[3] - points_np[0])
        height_right = np.linalg.norm(points_np[2] - points_np[1])
        height = int(max(height_left, height_right))

        # Zielrechteck
        dst = np.array([
            [0, 0],
            [width-1, 0],
            [width-1, height-1],
            [0, height-1]
        ], dtype=np.float32)

        # Perspektivische Transformation
        M = cv2.getPerspectiveTransform(points_np, dst)
        straightened = cv2.warpPerspective(temp, M, (width, height), flags=cv2.INTER_NEAREST)

        # Winkel aus Oberkante berechnen (kleine Korrektur)
        top_left, top_right = dst[0], dst[1]
        dx = top_right[0] - top_left[0]
        dy = top_right[1] - top_left[1]
        angle = np.degrees(np.arctan2(dy, dx))

        if abs(angle) > 0.1:
            center = (width // 2, height // 2)
            R = cv2.getRotationMatrix2D(center, -angle, 1)
            straightened = cv2.warpAffine(straightened, R, (width, height),
                                          flags=cv2.INTER_NEAREST,
                                          borderMode=cv2.BORDER_REPLICATE)

        # Punkte entfernen
        for line in list(ax.lines):
            line.remove()

        # Bild anzeigen
        im.set_data(straightened)
        im.set_clim(vmin=straightened.min(), vmax=straightened.max())
        fig.canvas.draw_idle()

        # Array speichern, um es zurückzugeben
        corrected_array = straightened.copy()

        # Punkte löschen für nächste Operation
        points.clear()

fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()
plt.close()
plt.imshow(corrected_array, aspect='equal', cmap='inferno')
plt.show()

temp = corrected_array

lambda_Cu = 235

t_avg = np.average(temp, axis = 0)
x = np.arange(len(t_avg))

fig, ax = plt.subplots()
ax.plot(x, t_avg, label='Avg Temp')
ax.set_title("Click 4 points: [Cu-start, film-start, film-end, Cu-end]")

points_bbox = []
start_film = 0
end_film = 0
start_Cu = 0
end_Cu = 0
def onclick_bbox(event):
    global start_film, end_film, start_Cu, end_Cu
    if event.xdata is None:
        return

    points_bbox.append(int(event.xdata))
    ax.axvline(int(event.xdata), color='k', linestyle='--')
    fig.canvas.draw()
    print(f"Clicked: {int(event.xdata)}")

    if len(points_bbox) == 4:
        # Sort to ensure left-to-right order
        sorted_pts = sorted(points_bbox)
        start_Cu, start_film, end_film, end_Cu = sorted_pts
        print(f"Bounding box indices set:")
        print(f"start_film={start_film}, end_film={end_film}, start_Cu={start_Cu}, end_Cu={end_Cu}")
        plt.close(fig)  # Close after selecting 4 points

fig.canvas.mpl_connect('button_press_event', onclick_bbox)
plt.show()
print()

A_film = np.vstack([x[start_film:end_film], np.ones(len(x[start_film:end_film]))]).T
m_film, c_film = np.linalg.lstsq(A_film, t_avg[start_film:end_film])[0]
plt.plot(x, m_film*x + c_film, 'r', label='Fitted line')


A_Cu = np.vstack([x[start_Cu:end_Cu + 1], np.ones(len(x[start_Cu:end_Cu + 1]))]).T
m_Cu, c_Cu = np.linalg.lstsq(A_Cu, t_avg[start_Cu:end_Cu+1])[0]
plt.plot(x, m_Cu*x + c_Cu, 'g', label='Fitted line')

print(m_Cu, m_film)

lambda_film = lambda_Cu * m_Cu / m_film
print("Thermal conductivity of the film:", lambda_film)
plt.show()
