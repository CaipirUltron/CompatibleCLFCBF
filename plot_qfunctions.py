import sys, importlib
import matplotlib.pyplot as plt

simulation_file = sys.argv[1].replace(".json","")
sim = importlib.import_module("examples."+simulation_file, package=None)

''' ----------------------------------- Initialize plot ----------------------------------- '''
font = {'weight': 'bold', 'size': 8}
plt.rc('font', **font)

for cbf_index, cbf in enumerate(sim.cbfs):

    fig = plt.figure()

    cbf_segments = sim.kerneltriplet.invariant_segs[cbf_index]
    fig.suptitle(f"Q-function for CBF{cbf_index+1}", fontsize=12)

    ax1 = plt.subplot(2, 1, 1)
    ax1.set_title("q(λ)", fontsize=10)

    ax2 = plt.subplot(2, 1, 2)
    ax2.set_title("q(λ/γ(V))", fontsize=10)

    lambdas, norm_lambdas = [], []
    cbf_values = []
    for k, seg in enumerate(cbf_segments):

        ax1.plot( seg["lambdas"], seg["cbf_values"], '-', lw=1, label=f"seg{k+1}" )
        ax2.plot( seg["normalized_lambdas"], seg["cbf_values"], '-', lw=1, label=f"seg{k+1}" )

        lambdas += seg["lambdas"]
        norm_lambdas += seg["normalized_lambdas"]
        cbf_values += seg["cbf_values"]

    ax1.set_xlim( 0.0, max(lambdas) )
    ax1.set_ylim( min(cbf_values), max(cbf_values) )

    ax2.set_xlim( 0.0, max(norm_lambdas) )
    ax2.set_ylim( min(cbf_values), max(cbf_values) )

    ax1.legend(loc="upper right")
    ax2.legend(loc="upper right")

plt.show()