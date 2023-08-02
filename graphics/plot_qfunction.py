import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from examples.integrator_nominalQP import clf_params, cbf_params1, cbf_params2, cbf_params3
from controllers.compatibility import LinearMatrixPencil2

cbf_params = cbf_params2

Hv, Hh = clf_params["Hv"], cbf_params["Hh"]
x0, p0 = np.array(clf_params["x0"]), np.array(cbf_params["p0"])

# dim = 3
# Hv = 10*np.random.rand(dim,dim)
# Hv = Hv.T @ Hv
# Hh = 10*np.random.rand(dim,dim)
# Hh = Hh.T @ Hh
# x0, p0 = np.random.rand(dim), np.random.rand(dim)

loaded = 1
if len(sys.argv) > 1:
    loaded = 1
    test_config = sys.argv[1].replace(".json","")
    with open(test_config + ".json") as file:
        print("Loading test: " + test_config + ".json")
        test_vars = json.load(file)
    Hv = np.array(test_vars["Hv"])
    Hh = np.array(test_vars["Hh"])
    x0 = np.array(test_vars["x0"])
    p0 = np.array(test_vars["p0"])
    dim = np.shape(Hv)[0]
    n = dim - 1

p_min, p_max = -10, 40
q_min, q_max = 0, 3
lambda_p = np.linspace(p_min, p_max, 1000)

def q_function(l):
    H = l*Hh - Hv
    v = np.linalg.inv(H) @ Hv @ (p0 - x0)
    return v.T @ Hh @ v

pencil = LinearMatrixPencil2( Hh, Hv )
q = np.zeros(len(lambda_p))
for k in range(len(lambda_p)):
    q[k] = q_function(lambda_p[k])

max_eig = np.argmax(pencil.eigenvalues)
sol = fsolve(q_function, max_eig + 1000*np.random.rand())

fig, ax = plt.subplots(tight_layout=False)
ax.plot(lambda_p, q, color='blue', linewidth=0.8)
ax.plot([p_min, p_max], [1, 1], '--', color='green', linewidth=2.0)
ax.plot(sol, '*')
for eig in pencil.eigenvalues:
    ax.plot([eig, eig], [q_min, q_max], '--', color='red')

ax.set_xlim([p_min, p_max])
ax.set_ylim([q_min, q_max])
ax.set_xlabel("$\lambda$")
# ax.set_ylabel("$q(\lambda)$")
ax.legend(["$q_1(\lambda)$"], fontsize=16)
# ax.legend(["$q_2(\lambda)$"], fontsize=16)

# plt.savefig(test_config + ".eps", format='eps', transparent=True)
plt.savefig("q_function1.eps", format='eps', transparent=True)
plt.show()

test_vars = {"Hv": Hv.tolist(),
             "Hh": Hh.tolist(),
             "x0": x0.tolist(),
             "p0": p0.tolist()}
if (loaded == 0):
    print("Save file? Y/N")
    if str(input()).lower() == "y":
        print("File name: ")
        file_name = str(input())
        with open(file_name+".json", "w") as file:
            print("Saving test data...")
            json.dump(test_vars, file, indent=4)