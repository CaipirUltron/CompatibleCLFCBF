import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from controllers import SplinePath
from graphics import SplineEditor

loaded = False

# Create initial spline from given points...
num_points = 6
pts_x = [-8, -8, -6, 6, 8, 8]
pts_y = [-5, 0, 0, 0, 0, 5]
pts = np.array([pts_x, pts_y]).T
spline_params = { "degree": 3, "points": pts, "orientation": 'left' }

# num_points = 6
# pts = np.random.rand(num_points,2)
# spline_params = { "degree": 3, "points": pts, "orientation": 'left' }

# Or edit from a previously saved configuration
if len(sys.argv) > 1:
    loaded = True
    spline_name = sys.argv[1].replace(".json","")
    location = "graphics/splines/"+spline_name+".json"
    try:
        with open(location,'r') as file:
            print(file)
            print("Loading spline: " + spline_name + ".json")
            spline_params = json.load(file)
            spline_params["points"] = np.array( spline_params["points"] )
    except: print("No such file on " + location)

# Generate spline pathy
spline_path = SplinePath( params=spline_params, init_path_state=[0.0] )

# Initialize spline plot
offset = 1

coords = np.hstack([ spline_params["points"][:,0], spline_params["points"][:,1] ])
min = np.min( coords ) - offset
max = np.max( coords ) + offset
plot_params = {
    "axeslim": (min,max,min,max),
    "path_length": num_points + 1, 
    "numpoints": 200
}

fig, axes = plt.subplots(figsize=(6, 6))
axes.set_title('Spline Editor')
axes.set_xlim( plot_params["axeslim"][0:2] )
axes.set_ylim( plot_params["axeslim"][2:4] )
axes.set_aspect('equal', adjustable='box')

spline_graph, = axes.plot([],[], linestyle='dashed', lw=0.8, alpha=0.8, color='b')
editor = SplineEditor(spline_path, spline_graph, plot_params)
plt.show()

# Save new spline configuration
spline_params["points"] = editor.path.points.tolist()

save = False
if loaded:
    save = True
else:
    print("Save file? Y/N")
    if str(input()).lower() == "y":
        save = True
        print("File name: ")
        spline_name = str(input())
        location = "graphics/splines/"+spline_name+".json"

if save:
    with open(location, "w") as file:
        print("Saving spline at "+location)
        json.dump(spline_params, file, indent=4)