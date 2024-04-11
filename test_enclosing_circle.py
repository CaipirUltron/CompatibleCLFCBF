import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from common import enclosing_circle

limits = (-2,-3,1,8)
xmin, ymin, xmax, ymax = limits
xrange = xmax - xmin
yrange = ymax - ymin

radius, center = enclosing_circle(limits)

# Plotting
fig, ax = plt.subplots()

# Plot rectangle
rectangle = Rectangle( (xmin,ymin), xrange, yrange, angle=0.0, linewidth=1, edgecolor='k')
ax.add_patch(rectangle)

# Plot circle
circle = Circle(center, radius, linewidth=1, edgecolor='r', facecolor='none')
ax.add_patch(circle)

# Set axis limits
ax.set_xlim(xmin-radius-1, xmax+radius+1)
ax.set_ylim(ymin-radius-1, ymax+radius+1)

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Minimum Enclosing Circle for Rectangle')

# Show plot
plt.gca().set_aspect('equal', adjustable='box')
plt.grid(True)
plt.show()