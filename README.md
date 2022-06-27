# Compatible CLF-CBFs

This project contains controller implementations for safety-critical systems using compatible CBF-CBF pairs.
The main objetive is to design controllers capable of achieving global asymptotic stabilization of the origin 
while maintaining positive invariance of a given connected set, by using Control Lyapunov and Barrier functions.

Such systems are known to generate undesirable equilibria that can be stable. 
Compatibility in a CLF-CBF pair is a property that ensures that the given combination of CLF and CBF does not generate 
stable equilibria other than the origin on the closed-loop system formed by a min-norm QP-based controller.

**Running instructions:** 

$ *python3 examples/<example_file_name>.py*

This project is structured in the following way:

    ./functions
    
    ./dynamic_systems

    ./controllers

    ./graphical_simulation
    
    ./tests

    ./examples

**./functions:** Contains many classes of functions to be used as Control Lyapunov/Barrier functions, and plant models.

**./dynamic_systems:** Contains different classes of dynamic systems for simulation, such as linear, affine nonlinear systems, and polynomial systems.

**./controlers:** Contains different controllers for achieving compatibility.

**./graphical_simulation:** All code functionality for graphical simulations.

**./tests:** Current test scripts (DEVEL, not to be used)

**./examples:** Contains simulation examples for some of the implemented controllers.