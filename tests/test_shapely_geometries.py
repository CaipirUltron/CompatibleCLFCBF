import matplotlib.pyplot as plt
from shapely.geometry import LineString
from shapely.plotting import plot_polygon, plot_line
from shapely import is_geometry
from common import discretize

line = LineString([(-4, 2), (-4, 0), (0, 0), (4, 0), (4, 2)])

fig = plt.figure()

# 1
ax = fig.add_subplot(121)

plot_line(line, ax=ax, add_points=False, linewidth=3)

dilated = line.buffer(1.0, cap_style='round')

for pt in discretize(dilated, spacing=0.3):
    ax.plot(pt[0], pt[1], 'k*', alpha=0.6)

for pt in discretize(line, spacing=0.3):
    ax.plot(pt[0], pt[1], 'k*', alpha=0.6)

plot_polygon(dilated, ax=ax, add_points=False, alpha=0.5)

ax.set_title('a) dilation, cap_style=3')
ax.axis('equal')

#2
ax = fig.add_subplot(122)

plot_polygon(dilated, ax=ax, add_points=False, alpha=0.5)

eroded = dilated.buffer(-0.5)
plot_polygon(eroded, ax=ax, add_points=False, alpha=0.5)

ax.set_title('b) erosion, join_style=1')
ax.axis('equal')

fig.tight_layout()
plt.show()