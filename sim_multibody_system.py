import platform, os
import numpy as np

from dataclasses import dataclass
from scipy.integrate import ode

import matplotlib.pyplot as plt
import matplotlib.animation as anim
import matplotlib.colors as mcolors
from matplotlib.patches import Circle

G = 3       # gravitational constant

pos_min, pos_max = -100, 100
vel_min, vel_max = -20, 20

@dataclass
class Body():
    name: str
    mass: float = 1
    radius: float = 1
    color: str = "black"
    position: np.ndarray = None
    velocity: np.ndarray = None
    fixed: bool = False
    def __post_init__(self):
        if not self.position: self.position = np.random.uniform(low=pos_min, high=pos_max, size=2)
        if not self.velocity: self.velocity = np.random.uniform(low=vel_min, high=vel_max, size=2)

        if isinstance(self.position, (list, tuple)): self.position = np.array(self.position)
        if isinstance(self.velocity, (list, tuple)): self.velocity = np.array(self.velocity)
        
sun = Body("Sun", mass=1e5, radius=15, color="orange", position=(0, 0), fixed = True)

mercury = Body("Mercury", mass=1e2, radius=2, color="grey", position=(+50, 0), velocity=(0, +60))
venus = Body("Venus", mass=1e2, radius=3, color="yellow", position=(+80, 0), velocity=(0, +60))
earth = Body("Earth", mass=1e2, radius=4, color="cyan", position=(+110, 0), velocity=(0, +60))
mars = Body("Mars", mass=1e2, radius=3, color="red", position=(+140, 0), velocity=(0, +60))

bodies = [ sun, mercury, venus, earth, mars ]

def bodies2state(bodies):
    ''' Computes multibody system '''
    return np.hstack([ np.hstack([body.position, body.velocity]) for body in bodies ])

def update_bodies(bodies, multibody_state):
    ''' Updates the state of each body according to the multibody state'''
    for k, body in enumerate(bodies):
        body.position = multibody_state[4*k:4*k+2]
        body.velocity = multibody_state[4*k+2:4*k+4]

def get_flow(t):
    '''
    Computes flow for all bodies
    '''
    num_bodies = len(bodies)
    dstate = np.zeros(4*num_bodies)
    for k, body in enumerate(bodies):

        # if body is fixed, ignore motion
        if body.fixed: continue

        # remove self interaction
        neighbors = bodies[:]
        neighbors.remove(body)

        # if some of the neighbors is fixed, ignore all other interactions
        if np.any([ neighbor.fixed for neighbor in neighbors]):
            for neighbor in neighbors: 
                if not neighbor.fixed: neighbors.remove(neighbor)

        gravitational_force = np.zeros(2)
        for neighbor in neighbors:
            delta_pos = neighbor.position - body.position
            distance = np.linalg.norm(delta_pos)
            unit_delta_pos = delta_pos/distance
            gravitational_force += G * (body.mass * neighbor.mass) * unit_delta_pos/distance**2

        dstate[4*k:4*k+2] = body.velocity
        dstate[4*k+2:4*k+4] = (1/body.mass) * gravitational_force
    
    return dstate

#------------------------------ Multibody simulation ----------------------------
T = 20
dt = 0.01

diff_equation = ode(get_flow).set_integrator('dopri5')
initial_multibody_state = bodies2state(bodies)
diff_equation.set_initial_value(initial_multibody_state)

num_steps = int(T/dt)
state_log = np.zeros([num_steps, 4*len(bodies)])
for step in range(0, num_steps):

    t = step*dt
    if platform.system().lower() != 'windows':
        os.system('var=$(tput lines) && line=$((var-2)) && tput cup $line 0 && tput ed')           # clears just the last line of the terminal
    print("Simulating instant t = " + str(float(f'{t:.3f}')) + " s")

    multibody_state = diff_equation.integrate(diff_equation.t+dt)
    update_bodies(bodies, multibody_state)
    state_log[step, :] = bodies2state(bodies)

#------------------------------ Multibody animation ----------------------------
xmin, xmax = -200, 200
ymin, ymax = -200, 200

update_bodies(bodies, initial_multibody_state)

fig, ax = plt.subplots()
plt.title("Simulation of Multibody System")
ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
fig.tight_layout()

time_text = ax.text(xmin+5, ymax-20, str("Time = "), fontsize=10)
time_text.text = str("Time = ")

circles = []
names = []
for body in bodies:
    circle = Circle(tuple(body.position), body.radius, color=body.color)
    names.append( ax.text(0, 0, body.name, fontsize=8) )

    circles.append(circle)
    ax.add_patch(circle)

def update(step):
    ''' Updates animation '''
    curr_multibody_state = state_log[step, :]
    current_time = np.around(step*dt, decimals = 2)
    time_text.set_text("Time = " + str(current_time) + "s")
    
    update_bodies(bodies, curr_multibody_state)
    for circle, body, name in zip(circles, bodies, names):
        circle.center = tuple(body.position)
        name.set_position(tuple(body.position))
    return circles + [time_text] + names

fps = 30
animation = anim.FuncAnimation(fig, func=update, frames = num_steps, interval=1000/fps, repeat=False, blit=True, cache_frame_data=False)

plt.show()