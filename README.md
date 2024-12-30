# Compatible CLF-CBFs

This project contains controller implementations for safety-critical systems using compatible CBF-CBF pairs.
The main objetive is to design controllers capable of achieving global asymptotic stabilization of the origin 
while maintaining positive invariance of the safe set, by using Control Lyapunov and Control Barrier functions.

Such systems are known to generate undesirable equilibria that can be stable. 
Compatibility in a CLF-CBF pair is a property that ensures that the given combination of CLF and CBF does 
not generate stable equilibria other than the origin on the closed-loop system formed by a min-norm QP-based controller.

**Instalation instructions:**

pip3 install -e .

**Running instructions:** 

$ *python3 run_simulation.py <config_file.py in ./examples> <control_mode: "no_control", "nominal", "compatible"> (opt) <starting time>*: **runs specified simulation**.

$ *python3 plot_animation.py <config_file.py in ./examples> <control_mode: "no_control", "nominal", "compatible"> (opt) <starting time>*: **animates specified simulation**.

$ *python3 plot_frame.py <config_file.py in ./examples> <control_mode: "no_control", "nominal", "compatible"> (opt) <time frame>*: **plots simulation at specified time frame**.

This project is structured in the following way:

    ./common

    ./functions
    
    ./dynamic_systems

    ./controllers

    ./graphics
    
    ./tests

    ./examples

    ./quadratic_program

    ./logs

**./common:** Contains basic functionality for all project.

**./functions:** Contains many classes of functions to be used as Control Lyapunov/Barrier functions, and plant models.

**./dynamic_systems:** Contains base class for simulation of control systems and child classes such as Affine Nonlinear and LTI systems.

**./controllers:** Contains different controllers for achieving compatibility.

**./graphics:** All code functionality for graphical simulations.

**./tests:** Testing scripts (DEVEL, not to be used)

**./examples:** Contains simulation examples for some of the implemented controllers.

**./quadratic_program:** Contains class for easily implementing QPs.

**./logs:** Contains log files.


**Example of use:** 

    $ python run_simulation.py LTI_multiple nominal

    Runs simulation of the CLF-CBF-QP nominal controller for an LTI system with specified CLF and CBF list in examples/LTI_multiple.py and stores results in logs/LTI_multiple.json

    $ python plot_animation.py LTI_multiple nominal

    Plots corresponding animation