import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
 
# ICML plot settings
def setup_icml_plot(two_column=False):
    """Set up ICML-compatible plot settings."""
    if two_column:
        figure_width = 7  # Full-page width for two-column layout (in inches)
    else:
        figure_width = 3.5  # Half-page width for two-column layout (in inches)
 
    rcParams.update({
        # Font and text
        "text.usetex": True,  # Use LaTeX for text rendering
        "font.family": "serif",  # Use serif fonts
        "font.serif": ["Times New Roman"],  # Set font to Times New Roman
        "axes.labelsize": 9,  # Font size for axis labels
        "axes.titlesize": 10,  # Font size for titles
        "legend.fontsize": 5,  # Font size for legends
        "xtick.labelsize": 8,  # Font size for x-axis ticks
        "ytick.labelsize": 8,  # Font size for y-axis ticks
 
        # Line and marker styles
        "lines.linewidth": 1.2,  # Line width
        "lines.markersize": 3,  # Marker size
 
        # Figure dimensions
        "figure.figsize": (figure_width, figure_width * 0.15),  # TODO change to better ratio
        "figure.dpi": 300,  # High resolution for publication
 
        # Grid
        "axes.grid": True,  # Enable grid
        "grid.alpha": 0.3,  # Grid transparency
        "grid.linestyle": "--",  # Dashed grid lines
 
        # Legend
        "legend.frameon": False,  # No border around legends
    })
 
# LOADING DATA =================================================================
# ==============================================================================
x = np.linspace(0, 8, 8)  # Generate 4 equally spaced x values
 
# MNIST Known cluster
zzz_m_k_p = [0.95868, 0.958179973, 0.955477037, 0.956833643, 0.956691098, 0.957344397, 0.959105845, 0.959330055]
zzz_m_k = [0.95716, 0.953416221, 0.948753359, 0.935387762, 0.934364246, 0.905346668, 0.877656173, 0.883966582]
fedavg_m_k = [0.930745, 0.893799407, 0.844867348, 0.815645909, 0.809354854, 0.761633144, 0.720145573, 0.693643525]
ifca_m_k = [0.91445, 0.925200496, 0.897421436, 0.836328798, 0.836385792, 0.795800616, 0.759490609, 0.759953996]
fedrc_m_k = [0.892665, 0.738156295, 0.532791703, 0.40640604, 0.356091856, 0.305948583, 0.302531365, 0.261487611]
fedem_m_k = [0.8923875, 0.73805646, 0.532685188, 0.406105652, 0.35849779, 0.311627509, 0.311832755, 0.31103636]
fesem_m_k = [0.93581, 0.911881773, 0.886721305, 0.876211873, 0.873723201, 0.853330456, 0.856006458, 0.832014969]
feddrift_m_k = [0.9577275, 0.947690915, 0.919901047, 0.914774518, 0.889791811, 0.875931678, 0.885267582, 0.895640489]
cfl_m_k = [0.932225, 0.896832599, 0.84948284, 0.820947835, 0.813945671, 0.769356017, 0.724478905, 0.695080567]
pfedme_m_k = [0.948408333, 0.949676026, 0.952441528, 0.954061107, 0.955559757, 0.95748784, 0.95891443, 0.960927334]
apfl_m_k = [0.957225, 0.955585629, 0.954231602, 0.954330744, 0.954520379, 0.954083856, 0.952982874, 0.953741034]

std_zzz_m_k_p = [0.002022011, 0.000919172, 0.005601136, 0.002188831, 0.002412299, 0.002807545, 0.00178645, 0.003192284]
std_zzz_m_k = [0.003807335, 0.006243023, 0.011600455, 0.02675103, 0.02280969, 0.031414473, 0.052289396, 0.042243348]
std_fedavg_m_k = [0.005003099, 0.014232698, 0.024337487, 0.022197352, 0.015300718, 0.023776705, 0.019627886, 0.027145578]
std_ifca_m_k = [0.058098341, 0.022231338, 0.032695312, 0.060868411, 0.098769756, 0.082183842, 0.100726446, 0.096845371]
std_fedrc_m_k = [0.040251969, 0.042900299, 0.103055879, 0.049857122, 0.040791609, 0.03086611, 0.026076894, 0.02407389]
std_fedem_m_k = [0.03982443, 0.042785806, 0.103230803, 0.050053862, 0.041495629, 0.029433974, 0.023880183, 0.014881785]
std_fesem_m_k = [0.013092847, 0.018288029, 0.020021729, 0.021196828, 0.032754906, 0.037210677, 0.035034998, 0.051690049]
std_feddrift_m_k = [0.004396957, 0.012688754, 0.029745917, 0.03794027, 0.023397341, 0.039956328, 0.036154557, 0.024248329]
std_cfl_m_k = [0.004277217, 0.014043118, 0.021296264, 0.019638573, 0.017324256, 0.01869977, 0.020566085, 0.023694821]
std_pfedme_m_k = [0.002164942, 0.002821769, 0.003102993, 0.003036425, 0.002771358, 0.002860208, 0.001556028, 0.001898364]
std_apfl_m_k = [0.001475079, 0.002386421, 0.002881358, 0.003677622, 0.002635576, 0.002978893, 0.002605131, 0.003917592]
 
# MNIST Testing phase inference

zzz_m_t_p = [0.956866667, 0.948302362, 0.953881284, 0.959338191, 0.955750796, 0.957342987, 0.957760342, 0.962840074]
zzz_m_t = [0.95608, 0.944467982, 0.944882555, 0.948053267, 0.938781989, 0.934598459, 0.927048231, 0.923245645]
fedavg_m_t = [0.9215, 0.89801921, 0.863136464, 0.853207879, 0.866233138, 0.839177525, 0.811194097, 0.793598033]
ifca_m_t = [0.89932, 0.856526603, 0.829386863, 0.812178236, 0.792301744, 0.740440019, 0.6832205, 0.642242196]
fedrc_m_t = [0.858286667, 0.68835514, 0.501435753, 0.479509412, 0.429648791, 0.370978924, 0.361547355, 0.30318725]
fedem_m_t = [0.85798, 0.68828116, 0.501280263, 0.479184977, 0.427472082, 0.367462494, 0.355969225, 0.321511041]
fesem_m_t = [0.91689, 0.889877898, 0.856267848, 0.844558075, 0.823714333, 0.803219255, 0.759856471, 0.728307223]
feddrift_m_t = [0.854986667, 0.762768998, 0.717174107, 0.662265673, 0.643885009, 0.576953954, 0.529132248, 0.462777345]
cfl_m_t = [0.927376667, 0.900996799, 0.869110453, 0.859537113, 0.871380895, 0.849421356, 0.816598541, 0.79515409]
apfl_m_t = [0.91, 0.889071607, 0.852037039, 0.840100209, 0.851944968, 0.840827298, 0.801356491, 0.789556316]
atp_m_t = [0.920706667, 0.891142233, 0.859148718, 0.849174764, 0.867996903, 0.849894114, 0.811398364, 0.799192462]


std_zzz_m_t_p = [0.003311274, 0.010239501, 0.007962698, 0.003178806, 0.011205795, 0.018774196, 0.013638556, 0.013310829]
std_zzz_m_t = [0.005702467, 0.014191212, 0.008820067, 0.015518107, 0.034316835, 0.028162492, 0.029583535, 0.02246802]
std_fedavg_m_t = [0.005774496, 0.008774135, 0.027653588, 0.02174304, 0.017548355, 0.02575846, 0.020874734, 0.029039238]
std_cfl_m_t = [0.011276485, 0.007706064, 0.024214372, 0.01919417, 0.019320947, 0.019504933, 0.02135467, 0.025423479]
std_ifca_m_t = [0.030719056, 0.015503516, 0.019543257, 0.043642925, 0.081466179, 0.077882485, 0.061346426, 0.058864697]
std_feddrift_m_t = [0.018802477, 0.030140446, 0.034568692, 0.034144997, 0.052953624, 0.050019969, 0.026852562, 0.036586219]
std_fesem_m_t = [0.017599144, 0.016023304, 0.030948681, 0.023666558, 0.048911274, 0.039614827, 0.04769621, 0.051712603]
std_fedem_m_t = [0.027804796, 0.047420206, 0.082822622, 0.04090284, 0.045537792, 0.033284681, 0.025322431, 0.032124977]
std_fedrc_m_t = [0.027727325, 0.047819618, 0.082825984, 0.040839549, 0.045772348, 0.035002891, 0.025041461, 0.028361476]
std_apfl_m_t = [0.009612221, 0.014834103, 0.021617178, 0.030907069, 0.022431657, 0.019456122, 0.02194168, 0.027177479]
std_atp_m_t = [0.006905922, 0.008734329, 0.023449601, 0.019931339, 0.014111212, 0.02362482, 0.019155652, 0.025792059]
 
 
# change here
two_column = True
setup_icml_plot(two_column=two_column)
 
 
# Create a figure with three horizontal subplots
fig, axes = plt.subplots(2, 1, figsize=(2.2, 4.0))  # Maintain two-column format with horizontal subplots
 
 
# First row, known cluster
 
# MNIST
axes[0].errorbar(x, zzz_m_k_p, yerr=std_zzz_m_k_p, label="_nolegend_", color="black", marker="o", markersize=2,
                    elinewidth=0.8, capsize=2, alpha=0.2, ecolor="black")
axes[0].errorbar(x, zzz_m_k, yerr=std_zzz_m_k, label="_nolegend_", color="blue", marker="o", markersize=2,
                    elinewidth=0.8, capsize=2, alpha=0.2, ecolor="blue")
axes[0].errorbar(x, fedavg_m_k, yerr=std_fedavg_m_k, label="_nolegend_", color="orange", marker="o", markersize=2,
                    elinewidth=0.8, capsize=2, alpha=0.2, ecolor="orange")  
axes[0].errorbar(x, ifca_m_k, yerr=std_ifca_m_k, label="_nolegend_", color="green", marker="o", markersize=2,
                    elinewidth=0.8, capsize=2, alpha=0.2, ecolor="green")
axes[0].errorbar(x, fedrc_m_k, yerr=std_fedrc_m_k, label="_nolegend_", color="red", marker="o", markersize=2,
                    elinewidth=0.8, capsize=2, alpha=0.2, ecolor="red")
axes[0].errorbar(x, fedem_m_k, yerr=std_fedem_m_k, label="_nolegend_", color="purple", marker="o", markersize=2,
                    elinewidth=0.8, capsize=2, alpha=0.2, ecolor="purple")
axes[0].errorbar(x, fesem_m_k, yerr=std_fesem_m_k, label="_nolegend_", color="brown", marker="o", markersize=2,
                    elinewidth=0.8, capsize=2, alpha=0.2, ecolor="brown")
axes[0].errorbar(x, feddrift_m_k, yerr=std_feddrift_m_k, label="_nolegend_", color="pink", marker="o", markersize=2,
                    elinewidth=0.8, capsize=2, alpha=0.2, ecolor="pink")
axes[0].errorbar(x, cfl_m_k, yerr=std_cfl_m_k, label="_nolegend_", color="cyan", marker="o", markersize=2,
                    elinewidth=0.8, capsize=2, alpha=0.2, ecolor="cyan")
axes[0].errorbar(x, pfedme_m_k, yerr=std_pfedme_m_k, label="_nolegend_", color="darkgoldenrod", marker="o", markersize=2,
                    elinewidth=0.8, capsize=2, alpha=0.2, ecolor="darkgoldenrod")
axes[0].errorbar(x, apfl_m_k, yerr=std_apfl_m_k, label="_nolegend_", color="teal", marker="o", markersize=2,
                    elinewidth=0.8, capsize=2, alpha=0.2, ecolor="teal")
axes[0].plot(x, fedavg_m_k, label="FedAvg", color="orange", marker='o', markersize=2)
axes[0].plot(x, ifca_m_k, label="IFCA", color="green", marker='o', markersize=2)
axes[0].plot(x, fedrc_m_k, label="FedRC", color="red", marker='o', markersize=2)
axes[0].plot(x, fedem_m_k, label="FedEM", color="purple", marker='o', markersize=2)
axes[0].plot(x, fesem_m_k, label="FeSEM", color="brown", marker='o', markersize=2)
axes[0].plot(x, feddrift_m_k, label="FedDrift", color="pink", marker='o', markersize=2)
axes[0].plot(x, cfl_m_k, label="CFL", color="cyan", marker='o', markersize=2)
axes[0].plot(x, pfedme_m_k, label="pFEDme", color="darkgoldenrod", marker='o', markersize=2)
axes[0].plot(x, apfl_m_k, label="APFL", color="teal", marker='o', markersize=2)
axes[0].plot(x, zzz_m_k, label="FLUX", color="blue", marker='o', markersize=2)
axes[0].plot(x, zzz_m_k_p, label="FLUX-pr", color="black", marker='o', markersize=2)
axes[0].set_title("Test Phase")
axes[0].set_ylabel("Accuracy")
axes[0].set_xticks(x)
axes[0].set_xticklabels([1,2,3,4,5,6,7,8])
axes[0].set_ylim(0.2, 1)
# axes[0].legend(loc="lower left")
 
# Second row, testing phase inference
 
# MNIST
axes[1].errorbar(x, zzz_m_t_p, yerr=std_zzz_m_t_p, label="_nolegend_", color="black", marker="o", markersize=2,
                    elinewidth=0.8, capsize=2, alpha=0.2, ecolor="black")
axes[1].errorbar(x, zzz_m_t, yerr=std_zzz_m_t, label="_nolegend_", color="blue", marker="o", markersize=2,
                    elinewidth=0.8, capsize=2, alpha=0.2, ecolor="blue")
axes[1].errorbar(x, fedavg_m_t, yerr=std_fedavg_m_t, label="_nolegend_", color="orange", marker="o", markersize=2,
                    elinewidth=0.8, capsize=2, alpha=0.2, ecolor="orange")
axes[1].errorbar(x, ifca_m_t, yerr=std_ifca_m_t, label="_nolegend_", color="green", marker="o", markersize=2,
                    elinewidth=0.8, capsize=2, alpha=0.2, ecolor="green")
axes[1].errorbar(x, fedrc_m_t, yerr=std_fedrc_m_t, label="_nolegend_", color="red", marker="o", markersize=2,
                    elinewidth=0.8, capsize=2, alpha=0.2, ecolor="red")
axes[1].errorbar(x, fedem_m_t, yerr=std_fedem_m_t, label="_nolegend_", color="purple", marker="o", markersize=2,
                    elinewidth=0.8, capsize=2, alpha=0.2, ecolor="purple")
axes[1].errorbar(x, fesem_m_t, yerr=std_fesem_m_t, label="_nolegend_", color="brown", marker="o", markersize=2,
                    elinewidth=0.8, capsize=2, alpha=0.2, ecolor="brown")
axes[1].errorbar(x, feddrift_m_t, yerr=std_feddrift_m_t, label="_nolegend_", color="pink", marker="o", markersize=2,
                    elinewidth=0.8, capsize=2, alpha=0.2, ecolor="pink")
axes[1].errorbar(x, cfl_m_t, yerr=std_cfl_m_t, label="_nolegend_", color="cyan", marker="o", markersize=2,
                    elinewidth=0.8, capsize=2, alpha=0.2, ecolor="cyan")
axes[1].errorbar(x, apfl_m_t, yerr=std_apfl_m_t, label="_nolegend_", color="teal", marker="o", markersize=2,
                    elinewidth=0.8, capsize=2, alpha=0.2, ecolor="teal")
axes[1].errorbar(x, atp_m_t, yerr=std_atp_m_t, label="_nolegend_", color="gold", marker="o", markersize=2,
                    elinewidth=0.8, capsize=2, alpha=0.2, ecolor="gold")
axes[1].plot(x, fedavg_m_t, label="FedAvg", color="orange", marker='o', markersize=2)
axes[1].plot(x, ifca_m_t, label="IFCA", color="green", marker='o', markersize=2)
axes[1].plot(x, fedrc_m_t, label="FedRC", color="red", marker='o', markersize=2)
axes[1].plot(x, fedem_m_t, label="FedEM", color="purple", marker='o', markersize=2)
axes[1].plot(x, fesem_m_t, label="FeSEM", color="brown", marker='o', markersize=2)
axes[1].plot(x, feddrift_m_t, label="FedDrift", color="pink", marker='o', markersize=2)
axes[1].plot(x, cfl_m_t, label="CFL", color="cyan", marker='o', markersize=2)
axes[1].plot(x, apfl_m_t, label="APFL", color="teal", marker='o', markersize=2)
axes[1].plot(x, atp_m_t, label="ATP", color="gold", marker='o', markersize=2)
axes[1].plot(x, zzz_m_t, label="FLUX", color="blue", marker='o', markersize=2)
axes[1].plot(x, zzz_m_t_p, label="FLUX-pr", color="black", marker='o', markersize=2)
axes[1].set_xlabel("Heterogeneity Level")
# axes[1].set_title("Test Phase")
axes[1].set_ylabel("Accuracy")
axes[1].set_xticks(x)
axes[1].set_xticklabels([1,2,3,4,5,6,7,8])
axes[1].set_ylim(0.2, 1)
# axes[1].legend(loc="lower left")
 
 
# Instead, collect all handles & labels from both axes:
handles, labels = [], []
for ax in axes:
    h_list, l_list = ax.get_legend_handles_labels()
    for h, l in zip(h_list, l_list):
        if l not in labels:
            handles.append(h)
            labels.append(l)

# place one legend slightly closer to the plots
fig.legend(handles, labels,
           loc="center left",
           bbox_to_anchor=(0.96, 0.55),
           borderaxespad=0,
           frameon=True)

# make room for that legend (extend rect width to 0.9)
plt.tight_layout(rect=[0, 0, 0.9, 1])
 
 
# Adjust layout
plt.tight_layout()
plt.savefig("/Users/dariofenoglio/Desktop/Hetero_mnist.pdf", bbox_inches="tight")
plt.show()
