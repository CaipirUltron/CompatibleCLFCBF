import numpy as np
import matplotlib.pyplot as plt

from common import rot2D

import matplotlib.pyplot as plt
import numpy as np

# Function to plot tilted ellipses
def plot_tilted_ellipse(ax, color, a, b, angle):
    t = np.linspace(0, 2 * np.pi, 100)
    x = a * np.cos(t)
    y = b * np.sin(t)
    x_rot = x * np.cos(angle) - y * np.sin(angle)
    y_rot = x * np.sin(angle) + y * np.cos(angle)
    ax.plot(x_rot, y_rot, color)

# Create a figure and axis
fig, ax = plt.subplots()

# Plotting the first tilted ellipse (blue)
plot_tilted_ellipse(ax, 'blue', 2, 1, np.pi/4)

# Plotting the second tilted ellipse (red)
plot_tilted_ellipse(ax, 'red', 1.5, 0.8, -np.pi/3)

# Plotting the black dashed line passing through the origin at an angle
angle_of_line = np.pi/6  # Specify the angle in radians
x_line = np.linspace(-3, 3, 100)
y_line = np.tan(angle_of_line) * x_line
ax.plot(x_line, y_line, 'black', linestyle='--')

# Set axis limits for better visualization
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)

# Set labels and title
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_title('Tilted Ellipses with Dashed Line')

# Show the plot
plt.show()