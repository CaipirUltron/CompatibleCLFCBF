import sys, importlib
import matplotlib.pyplot as plt

simulation_file = sys.argv[1].replace(".json","")
sim = importlib.import_module("examples."+simulation_file, package=None)

''' ----------------------------------- Initialize plot ----------------------------------- '''
font = {'weight': 'bold', 'size': 8}
plt.rc('font', **font)

for cbf_index, cbf in enumerate(sim.cbfs):

    fig = plt.figure(figsize=(10,10))

    cbf_segments = sim.kerneltriplet.invariant_segs[cbf_index]
    fig.suptitle(f"Q-function for CBF{cbf_index+1} and x(λ), y(λ)", fontsize=12)

    ax1 = plt.subplot(2, 3, 1)
    ax1.set_title("q(λ)", fontsize=10)

    ax2 = plt.subplot(2, 3, 2)
    ax2.set_title("q(λ/γ(V))", fontsize=10)

    ax3 = plt.subplot(2, 3, 3)
    ax3.set_title("x(λ)", fontsize=10)

    ax4 = plt.subplot(2, 3, 4)
    ax4.set_title("λ", fontsize=10)

    ax5 = plt.subplot(2, 3, 5)
    ax5.set_title("λ/γ(V))", fontsize=10)

    ax6 = plt.subplot(2, 3, 6)
    ax6.set_title("y(λ)", fontsize=10)

    lambdas, norm_lambdas = [], []
    cbf_values, x_values, y_values = [], [], []
    for k, seg in enumerate(cbf_segments):

        ax1.plot( seg["lambdas"], seg["cbf_values"], '-', lw=1, label=f"s{k+1}" )
        ax2.plot( seg["normalized_lambdas"], seg["cbf_values"], '-', lw=1, label=f"s{k+1}" )
        
        ax1.plot( seg["lambdas"], [ 0.0 for _ in seg["cbf_values"] ], 'g--', lw=1.5 )
        ax2.plot( seg["normalized_lambdas"], [ 0.0 for _ in seg["cbf_values"] ], 'g--', lw=1.5 )

        x = [ pt[0] for pt in seg["points"] ]
        y = [ pt[1] for pt in seg["points"] ]

        ax3.plot( seg["lambdas"], x, '-', lw=1, label=f"x(λ), s{k+1}" )

        indices = [ k for k in range(len(seg["lambdas"])) ] 
        ax4.plot( indices, seg["lambdas"], '--', lw=1, label=f"λ, s{k+1}" )
        ax5.plot( indices, seg["normalized_lambdas"], '--', lw=1, label=f"λ, s{k+1}" )
        ax6.plot( seg["lambdas"], y, '-', lw=1, label=f"y(λ), s{k+1}" )

        lambdas += seg["lambdas"]
        norm_lambdas += seg["normalized_lambdas"]
        cbf_values += seg["cbf_values"]
        x_values += x
        y_values += y

    ax1.set_xlim( 0.0, max(lambdas) )
    ax1.set_ylim( min(cbf_values), max(cbf_values) )

    ax2.set_xlim( 0.0, max(norm_lambdas) )
    ax2.set_ylim( min(cbf_values), max(cbf_values) )

    ax3.set_xlim( 0.0, max(lambdas) )
    ax3.set_ylim( min(x_values), max(x_values) )

    ax4.set_xlim( 0, len(lambdas) )
    ax4.set_ylim( 0.0, max(lambdas) )

    ax5.set_xlim( 0, len(lambdas) )
    ax5.set_ylim( 0.0, max(norm_lambdas) )

    ax6.set_xlim( 0.0, max(lambdas) )
    ax6.set_ylim( min(y_values), max(y_values) )

    ax1.legend(loc="upper right")
    ax2.legend(loc="upper right")
    ax3.legend(loc="upper right")
    ax4.legend(loc="upper right")
    ax5.legend(loc="upper right")
    ax6.legend(loc="upper right")

plt.show()