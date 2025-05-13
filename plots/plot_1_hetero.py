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
        "axes.titlesize": 9,  # Font size for titles
        "legend.fontsize": 4,  # Font size for legends
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
 
# FMNIST Known cluster

zzz_f_k_p = [0.798825, 0.804902206, 0.803605147, 0.812669668, 0.818849002, 0.829607177, 0.839508584, 0.841645688]
zzz_f_k = [0.788855, 0.796550046, 0.793147994, 0.796582912, 0.791511176, 0.782692268, 0.803523294, 0.799993332]
fedavg_f_k = [0.772285, 0.733385627, 0.685230179, 0.655301071, 0.661428803, 0.621489727, 0.567678965, 0.540804696]
ifca_f_k = [0.792885, 0.757604681, 0.759096397, 0.72700935, 0.728388049, 0.715172979, 0.66268522, 0.651819309]
fedrc_f_k = [0.74536225, 0.692932303, 0.610307059, 0.548712761, 0.516914945, 0.463652725, 0.39860346, 0.365401208]
fedem_f_k = [0.74530675, 0.694946784, 0.611064014, 0.548987007, 0.520277379, 0.480444839, 0.437125142, 0.418074007]
fesem_f_k = [0.78742, 0.758717687, 0.715080619, 0.713423221, 0.714597806, 0.713098532, 0.687164579, 0.661590084]
feddrift_f_k = [0.79772, 0.78392753, 0.798080049, 0.804205803, 0.805597863, 0.815989466, 0.826408896, 0.825528264]
cfl_f_k = [0.776185, 0.737916831, 0.692706395, 0.661070632, 0.666312028, 0.626255883, 0.572341242, 0.546165155]
pfedme_f_k = [0.788575, 0.79736563, 0.807858357, 0.814542652, 0.823762313, 0.832452294, 0.842017594, 0.846255323]
std_pfedme_f_k = [0.008387366, 0.010163401, 0.009704337, 0.010969667, 0.011805615, 0.009448991, 0.007418304, 0.008999109]

std_zzz_f_k_p = [0.005645742, 0.010264198, 0.014654793, 0.011646817, 0.014218078, 0.015986313, 0.007932614, 0.014715123]
std_zzz_f_k = [0.006730557, 0.013996653, 0.016286261, 0.022328883, 0.030473601, 0.046463123, 0.035443401, 0.030612994]
std_fedavg_f_k = [0.007125419, 0.018814345, 0.022690081, 0.024577784, 0.024995722, 0.039928185, 0.019217604, 0.037210878]
std_ifca_f_k = [0.042636032, 0.057124892, 0.045058098, 0.059957705, 0.042495675, 0.043649222, 0.059259043, 0.057124741]
std_fedrc_f_k = [0.015043961, 0.028744163, 0.040736354, 0.03709924, 0.056929128, 0.083214618, 0.051672665, 0.05607771]
std_fedem_f_k = [0.014733613, 0.028699396, 0.041477632, 0.036662712, 0.059001495, 0.085070058, 0.053885076, 0.061105298]
std_fesem_f_k = [0.025335556, 0.033855554, 0.024539072, 0.03116499, 0.033767562, 0.049489687, 0.047558145, 0.041349479]
std_feddrift_f_k = [0.006514138, 0.01840895, 0.021564445, 0.010084098, 0.018383059, 0.023122641, 0.028562572, 0.029988524]
std_cfl_f_k = [0.008140172, 0.018538298, 0.021032615, 0.023188099, 0.025900024, 0.047648491, 0.029794036, 0.033363164]
apfl_f_k = [0.813616667, 0.810813233, 0.81208482, 0.81044719, 0.815427808, 0.820602032, 0.834719776, 0.838207823]
std_apfl_f_k = [0.005094388, 0.007261613, 0.009862702, 0.010603845, 0.015323006, 0.016410428, 0.009201396, 0.015770129]

# FMNIST Testing phase inference

zzz_f_t_p = [0.80102, 0.806973071, 0.813868487, 0.82595737, 0.833992003, 0.849236237, 0.862944779, 0.86709425]
zzz_f_t = [0.786466667, 0.795854073, 0.794383845, 0.804759778, 0.82230954, 0.805711283, 0.846917483, 0.842625929]
fedavg_f_t = [0.767413333, 0.74011417, 0.699360239, 0.684454762, 0.704191737, 0.675286303, 0.628085286, 0.606086261]
cfl_f_t = [0.771473333, 0.745029109, 0.707968527, 0.690860843, 0.710222704, 0.679601177, 0.63311499, 0.61211354]
ifca_f_t = [0.762906667, 0.702960792, 0.653403476, 0.646603022, 0.65123948, 0.612416039, 0.525270192, 0.525837628]
feddrift_f_t = [0.69908, 0.621918128, 0.558483351, 0.508424722, 0.49905857, 0.454759741, 0.392990468, 0.35370697]
fesem_f_t = [0.764753333, 0.722453456, 0.697673783, 0.651199764, 0.658787207, 0.642285447, 0.593758532, 0.566436889]
fedem_f_t = [0.7292, 0.671428763, 0.59351775, 0.555836687, 0.551529867, 0.500017638, 0.446711838, 0.429304984]
fedrc_f_t = [0.72924, 0.670962935, 0.593228933, 0.555859388, 0.552216901, 0.500821221, 0.442890566, 0.408274176]
apfl_f_t = [0.781277778, 0.745715909, 0.703591082, 0.680831344, 0.705481963, 0.684098603, 0.636114049, 0.598986477]
atp_f_t = [0.769106667, 0.733385202, 0.694258148, 0.672229921, 0.70513123, 0.666526796, 0.618683868, 0.614913977]


std_zzz_f_t_p = [0.00467581, 0.011701623, 0.015259225, 0.014060825, 0.014928977, 0.018255038, 0.009331278, 0.016887188]
std_zzz_f_t = [0.00683134, 0.019818943, 0.019580191, 0.027980805, 0.019329496, 0.043155699, 0.034264614, 0.030842948]
std_fedavg_f_t = [0.008016616, 0.015218085, 0.022263578, 0.027366692, 0.02115478, 0.043723601, 0.016359694, 0.039753589]
std_cfl_f_t = [0.009363689, 0.014177029, 0.02084727, 0.026085984, 0.022169825, 0.05308891, 0.031256726, 0.035526203]
std_ifca_f_t = [0.017114111, 0.028835968, 0.052292464, 0.056013176, 0.040319963, 0.060611264, 0.047525647, 0.068047815]
std_feddrift_f_t = [0.007395562, 0.024081944, 0.019652941, 0.019691357, 0.040352555, 0.033967437, 0.028761588, 0.023574049]
std_fesem_f_t = [0.013316887, 0.02815826, 0.024044429, 0.05324666, 0.056947753, 0.053183219, 0.042289516, 0.059524909]
std_fedem_f_t = [0.016688463, 0.030626872, 0.04131437, 0.047450213, 0.055940043, 0.086484852, 0.049151852, 0.055864525]
std_fedrc_f_t = [0.017365506, 0.029324912, 0.040496498, 0.047808416, 0.057594262, 0.088889309, 0.047691157, 0.0607463]
std_atp_f_t = [0.007900882, 0.014999518, 0.025835112, 0.023324552, 0.024094668, 0.047744319, 0.028622909, 0.050660286]


# CIFAR10 Known cluster

zzz_c_k_p = [0.373925, 0.376923532, 0.381931681, 0.395209187, 0.402519788, 0.411331391, 0.422125701, 0.442849716]
zzz_c_k = [0.36651, 0.374861876, 0.382870177, 0.378233346, 0.396545814, 0.385953854, 0.397788209, 0.408444952]
fedavg_c_k = [0.3668625, 0.349049151, 0.34190969, 0.331556864, 0.303721509, 0.280722211, 0.253034648, 0.245326412]
ifca_c_k = [0.400105, 0.39370889, 0.385634705, 0.386559478, 0.370765503, 0.366365397, 0.38375965, 0.378972432]
fedrc_c_k = [0.24883, 0.212930074, 0.169543771, 0.146550528, 0.153488599, 0.155807049, 0.150687542, 0.147599444]
fedem_c_k = [0.248415, 0.212930144, 0.169744639, 0.146777192, 0.15827582, 0.160952682, 0.164704123, 0.181342605]
fesem_c_k = [0.3764725, 0.372269321, 0.375203893, 0.37928538, 0.362826489, 0.363214087, 0.344959912, 0.346117496]
feddrift_c_k = [0.3808425, 0.362794621, 0.353958551, 0.349772747, 0.347133955, 0.332525582, 0.334756385, 0.342708771]
cfl_c_k = [0.3783175, 0.360400668, 0.352674763, 0.34103406, 0.315123642, 0.296469583, 0.26907676, 0.259450745]
pfedme_c_k = [0.382766667, 0.393601913, 0.404095293, 0.417107718, 0.430255541, 0.437809737, 0.454899412, 0.473382561]
std_pfedme_c_k = [0.010709768, 0.008829074, 0.008323734, 0.008714481, 0.006907408, 0.013635969, 0.01229199, 0.007727973]

std_zzz_c_k_p = [0.008243422, 0.011216848, 0.010674407, 0.012920749, 0.013017158, 0.020116323, 0.024904674, 0.021368303]
std_zzz_c_k = [0.008491949, 0.01024545, 0.010528988, 0.016936639, 0.014072441, 0.033336509, 0.030031916, 0.035545629]
std_fedavg_c_k = [0.001859145, 0.01082384, 0.012230028, 0.02276387, 0.011646489, 0.031633909, 0.029252426, 0.025916977]
std_ifca_c_k = [0.012044712, 0.017811418, 0.01911359, 0.021271982, 0.020549964, 0.031242931, 0.034558065, 0.03508037]
std_fedrc_c_k = [0.019710658, 0.023262677, 0.03542611, 0.024494047, 0.013139043, 0.018852351, 0.012061035, 0.022522658]
std_fedem_c_k = [0.020101416, 0.023073193, 0.03555906, 0.024972581, 0.018858393, 0.018702659, 0.016527594, 0.020743469]
std_fesem_c_k = [0.008307973, 0.012275412, 0.014695247, 0.024263122, 0.020566674, 0.023035191, 0.02984537, 0.056041303]
std_feddrift_c_k = [0.008800434, 0.011809014, 0.01070004, 0.0174051, 0.020552669, 0.026934764, 0.035843242, 0.04676273]
std_cfl_c_k = [0.003376719, 0.011052322, 0.011065054, 0.022909628, 0.011106158, 0.034411528, 0.030790938, 0.022927227]
std_apfl_f_t = [0.011209395, 0.019195069, 0.025311563, 0.029146065, 0.019748552, 0.033216187, 0.024975379, 0.030896124]
apfl_c_k = [0.427116667, 0.430057061, 0.434485427, 0.444398162, 0.44003673, 0.44804307, 0.464727644, 0.484471208]
std_apfl_c_k = [0.006152415, 0.006574644, 0.007057378, 0.008103857, 0.007218696, 0.014039081, 0.015409901, 0.011412385]

# CIFAR10 Testing phase inference

zzz_c_t_p = [0.371213333, 0.371702261, 0.373220395, 0.386766339, 0.393968752, 0.407198512, 0.4184912, 0.420309068]
zzz_c_t = [0.361306667, 0.369206525, 0.376292234, 0.383517495, 0.389119545, 0.405053278, 0.40984764, 0.397591429] #  0.397591429-0.326604003=7
fedavg_c_t = [0.363343333, 0.349212201, 0.351259586, 0.348115819, 0.316135345, 0.295049615, 0.264692864, 0.260861883]
ifca_c_t = [0.399623333, 0.389945926, 0.390916119, 0.392170226, 0.35195458, 0.350797094, 0.328711193, 0.319649452]
fedrc_c_t = [0.205206667, 0.178193045, 0.172277413, 0.159421009, 0.166173493, 0.172904227, 0.163698659, 0.15924944]
fedem_c_t = [0.204353333, 0.178188246, 0.172171904, 0.159585799, 0.170225361, 0.175135831, 0.166919131, 0.172476694]
fesem_c_t = [0.383293333, 0.375178825, 0.378458546, 0.376778153, 0.356408652, 0.331961849, 0.313744706, 0.306214873]
feddrift_c_t = [0.372683333, 0.355102828, 0.357030582, 0.341557253, 0.317716945, 0.286770764, 0.267532712, 0.265756266]
cfl_c_t = [0.373943333, 0.360414224, 0.361966351, 0.35677208, 0.328651522, 0.313619445, 0.28423568, 0.27830766]
apfl_c_t = [0.411344444, 0.389328411, 0.387014501, 0.385463547, 0.353417332, 0.344389853, 0.328434434, 0.326604003]
atp_c_t = [0.368526667, 0.36196948, 0.361680185, 0.357101994, 0.327796809, 0.32410911, 0.294982152, 0.29111413]


std_zzz_c_t_p = [0.009518389, 0.013967282, 0.01600947, 0.018930425, 0.02134424, 0.030427669, 0.039320529, 0.061079427]
std_zzz_c_t = [0.011745637, 0.01100267, 0.018471237, 0.017573411, 0.021623648, 0.035945146, 0.043488837, 0.063750885]
std_fedavg_c_t = [0.001871548, 0.009723574, 0.013705959, 0.025968811, 0.011372616, 0.035556108, 0.033095274, 0.028994926]
std_ifca_c_t = [0.010819391, 0.01132119, 0.01362378, 0.019617728, 0.022118354, 0.023322017, 0.021585934, 0.027345622]
std_fedrc_c_t = [0.024443214, 0.025186718, 0.036566038, 0.026695796, 0.011838823, 0.020526456, 0.011441802, 0.02539215]
std_fedem_c_t = [0.023007928, 0.024831811, 0.036530598, 0.026865526, 0.015558093, 0.022441341, 0.012505124, 0.020175904]
std_fesem_c_t = [0.007882247, 0.011600606, 0.014590076, 0.02997792, 0.022961089, 0.033251024, 0.031046321, 0.032329288]
std_feddrift_c_t = [0.009269586, 0.014183549, 0.020053364, 0.021524106, 0.017231699, 0.024406084, 0.033066257, 0.025134627]
std_cfl_c_t = [0.006108807, 0.009792325, 0.012013905, 0.026189149, 0.010009938, 0.038529501, 0.034841965, 0.025233321]
std_apfl_c_t = [0.012347131, 0.010837803, 0.008486007, 0.011753117, 0.023926558, 0.028491835, 0.02679871, 0.016112333]
std_atp_c_t = [0.006521967, 0.0107566, 0.011941588, 0.020582492, 0.013855301, 0.027137339, 0.027626913, 0.027303887]

 
# CIFAR100 Known cluster

zzz_cc_k_p = [0.4335585, 0.4099295, 0.422284, 0.4175525, 0.4397475, 0.42837375, 0.43579525, 0.43305275]
zzz_cc_k = [0.42845, 0.40456575, 0.4206575, 0.40883375, 0.43706675, 0.41147875, 0.412059, 0.41119875]
fedavg_cc_k = [0.48303325, 0.42815175, 0.42013125, 0.373981, 0.298628, 0.2624005, 0.23336, 0.20355925]
ifca_cc_k = [0.48925825, 0.42581475, 0.401946, 0.345209, 0.35795225, 0.27148, 0.315184, 0.28062825]
fedrc_cc_k = [0.4596, 0.41465175, 0.40128825, 0.3621341, 0.3184595, 0.2618705, 0.25817775, 0.22919375]
fedem_cc_k = [0.46604175, 0.4214935, 0.40806875, 0.37107975, 0.31930375, 0.281167, 0.27476725, 0.246045]
fesem_cc_k = [0.4426835, 0.4134, 0.4105455, 0.3727195, 0.337384, 0.29943675, 0.2968285, 0.276206]
feddrift_cc_k = [0.4533, 0.42418425, 0.420049, 0.38622175, 0.385086, 0.362803, 0.38327775, 0.35881475]
cfl_cc_k = [0.477925, 0.43018125, 0.424095, 0.38298925, 0.318496, 0.27311275, 0.247989, 0.214669]
pfedme_cc_k = [0.33298325, 0.344083, 0.3453145, 0.3442545, 0.37378775, 0.37536225, 0.386658, 0.38338875]
apfl_cc_k = [0.4964165, 0.4773205, 0.478312, 0.44277875, 0.425857, 0.41146, 0.4077575, 0.3995195]

std_zzz_cc_k_p = [0.00522725, 0.0147805, 0.00745225, 0.01018025, 0.010455, 0.011429, 0.00415175, 0.00660825]
std_zzz_cc_k = [0.01174575, 0.01232375, 0.00776675, 0.009116, 0.012493, 0.016513, 0.022556, 0.01887275]
std_fedavg_cc_k = [0.006489, 0.01087825, 0.0094275, 0.01442325, 0.0258825, 0.00495725, 0.00696775, 0.01564575]
std_ifca_cc_k = [0.0078735, 0.02580125, 0.0246765, 0.02211425, 0.05604675, 0.039252, 0.03882425, 0.0341835]
std_fedrc_cc_k = [0.00779, 0.0085475, 0.00800125, 0.01318025, 0.01447575, 0.01378675, 0.007138, 0.00579125]
std_fedem_cc_k = [0.00646375, 0.0084105, 0.00916, 0.01095775, 0.01035875, 0.01020175, 0.008512, 0.00749725]
std_fesem_cc_k = [0.00752925, 0.0177615, 0.01351825, 0.01333675, 0.02165, 0.025029, 0.0227285, 0.03382725]
std_feddrift_cc_k = [0.02177775, 0.02233475, 0.00813125, 0.0154245, 0.02921125, 0.02863875, 0.02144275, 0.048843]
std_cfl_cc_k = [0.00392275, 0.007896, 0.00852925, 0.00936825, 0.01518175, 0.00607725, 0.00379175, 0.01732675]
std_pfedme_cc_k = [0.00230475, 0.00418, 0.016411, 0.026385, 0.00498025, 0.00638325, 0.0037525, 0.00743225]
std_apfl_cc_k = [0.0030275, 0.006577, 0.00825475, 0.026043, 0.008476, 0.00539975, 0.00341325, 0.00487275]

# CIFAR100 Testing phase inference

zzz_cc_t_p = [0.430166667, 0.390503333, 0.416043, 0.389482, 0.428047, 0.395931333, 0.423699, 0.431134667]
zzz_cc_t = [0.432911333, 0.396262333, 0.404349, 0.392397667, 0.430892667, 0.392571333, 0.424191, 0.428854333] #0.428854333-0.40323=3
fedavg_cc_t = [0.510566667, 0.460158, 0.466941667, 0.424774667, 0.337648333, 0.303167333, 0.280946667, 0.256301333]
ifca_cc_t = [0.518477667, 0.440805667, 0.439948667, 0.405376667, 0.380018667, 0.303617667, 0.315320667, 0.282965]
fedrc_cc_t = [0.499833333, 0.452624667, 0.449039333, 0.408016667, 0.342762, 0.301733, 0.295702, 0.280832667]
fedem_cc_t = [0.500367, 0.453045, 0.449578333, 0.414197667, 0.344235, 0.304907333, 0.296806333, 0.273292667]
fesem_cc_t = [0.5086, 0.459677333, 0.468475333, 0.430832667, 0.377652, 0.323370667, 0.327789667, 0.288838]
feddrift_cc_t = [0.411596, 0.362788333, 0.350827667, 0.281402, 0.228041667, 0.204854333, 0.187599333, 0.156912]
cfl_cc_t = [0.504655667, 0.456086, 0.464182, 0.423974667, 0.356128, 0.312817, 0.298052, 0.271192]
apfl_cc_t = [0.336533333, 0.347213, 0.366318333, 0.364961, 0.380263333, 0.386721667, 0.397098333, 0.40323]
atp_cc_t = [0.499133334, 0.450152152, 0.45666119, 0.41649339, 0.342430782, 0.300379296, 0.276956807, 0.255788156]


std_zzz_cc_t_p = [0.028381667, 0.011245333, 0.021241333, 0.033620667, 0.022000667, 0.04013, 0.030240667, 0.034396333]
std_zzz_cc_t = [0.031887667, 0.013833333, 0.026817667, 0.027236333, 0.023273, 0.032559667, 0.028849, 0.033126]
std_fedavg_cc_t = [0.006301, 0.011036333, 0.009998333, 0.011962, 0.031652333, 0.005793333, 0.006813667, 0.020561333]
std_ifca_cc_t = [0.005747333, 0.013710667, 0.009738667, 0.008059, 0.036614667, 0.032948, 0.014658, 0.029752333]
std_fedrc_cc_t = [0.005133, 0.008603333, 0.007703667, 0.011691667, 0.012502333, 0.012235333, 0.008127, 0.009131333]
std_fedem_cc_t = [0.006207, 0.008830667, 0.008595, 0.009815333, 0.012121333, 0.012243333, 0.007318, 0.007742]
std_fesem_cc_t = [0.00396, 0.013132333, 0.010828333, 0.015590667, 0.020232, 0.023148667, 0.014620333, 0.012121333]
std_feddrift_cc_t = [0.018565, 0.020436667, 0.014936333, 0.011806, 0.013968667, 0.021146667, 0.008296, 0.011297333]
std_cfl_cc_t = [0.003675333, 0.007724333, 0.007739667, 0.007088333, 0.017076333, 0.006845, 0.003538, 0.022192]
std_apfl_cc_t = [0.002944667, 0.003289, 0.005758333, 0.012289, 0.005880667, 0.003617333, 0.004511, 0.004855667]
std_atp_cc_t = [0.006995694, 0.007416071, 0.009069888, 0.009099284, 0.024418394, 0.006805274, 0.006123036, 0.02001765]
 
 
# change here
two_column = True
setup_icml_plot(two_column=two_column)
 
 
# Create a figure with three horizontal subplots
fig, axes = plt.subplots(2, 4, figsize=(10, 4.7))  # Maintain two-column format with horizontal subplots
 
 
# First row, known cluster
 
# MNIST
axes[0,0].errorbar(x, zzz_m_k_p, yerr=std_zzz_m_k_p, label="_nolegend_", color="black", marker="o", markersize=2,
                    elinewidth=0.8, capsize=2, alpha=0.2, ecolor="black")
axes[0,0].errorbar(x, zzz_m_k, yerr=std_zzz_m_k, label="_nolegend_", color="blue", marker="o", markersize=2,
                    elinewidth=0.8, capsize=2, alpha=0.2, ecolor="blue")
axes[0,0].errorbar(x, fedavg_m_k, yerr=std_fedavg_m_k, label="_nolegend_", color="orange", marker="o", markersize=2,
                    elinewidth=0.8, capsize=2, alpha=0.2, ecolor="orange")  
axes[0,0].errorbar(x, ifca_m_k, yerr=std_ifca_m_k, label="_nolegend_", color="green", marker="o", markersize=2,
                    elinewidth=0.8, capsize=2, alpha=0.2, ecolor="green")
axes[0,0].errorbar(x, fedrc_m_k, yerr=std_fedrc_m_k, label="_nolegend_", color="red", marker="o", markersize=2,
                    elinewidth=0.8, capsize=2, alpha=0.2, ecolor="red")
axes[0,0].errorbar(x, fedem_m_k, yerr=std_fedem_m_k, label="_nolegend_", color="purple", marker="o", markersize=2,
                    elinewidth=0.8, capsize=2, alpha=0.2, ecolor="purple")
axes[0,0].errorbar(x, fesem_m_k, yerr=std_fesem_m_k, label="_nolegend_", color="brown", marker="o", markersize=2,
                    elinewidth=0.8, capsize=2, alpha=0.2, ecolor="brown")
axes[0,0].errorbar(x, feddrift_m_k, yerr=std_feddrift_m_k, label="_nolegend_", color="pink", marker="o", markersize=2,
                    elinewidth=0.8, capsize=2, alpha=0.2, ecolor="pink")
axes[0,0].errorbar(x, cfl_m_k, yerr=std_cfl_m_k, label="_nolegend_", color="cyan", marker="o", markersize=2,
                    elinewidth=0.8, capsize=2, alpha=0.2, ecolor="cyan")
axes[0,0].errorbar(x, pfedme_m_k, yerr=std_pfedme_m_k, label="_nolegend_", color="darkgoldenrod", marker="o", markersize=2,
                    elinewidth=0.8, capsize=2, alpha=0.2, ecolor="darkgoldenrod")
axes[0,0].errorbar(x, apfl_m_k, yerr=std_apfl_m_k, label="_nolegend_", color="teal", marker="o", markersize=2,
                    elinewidth=0.8, capsize=2, alpha=0.2, ecolor="teal")
axes[0,0].plot(x, fedavg_m_k, label="FedAvg", color="orange", marker='o', markersize=2)
axes[0,0].plot(x, ifca_m_k, label="IFCA", color="green", marker='o', markersize=2)
axes[0,0].plot(x, fedrc_m_k, label="FedRC", color="red", marker='o', markersize=2)
axes[0,0].plot(x, fedem_m_k, label="FedEM", color="purple", marker='o', markersize=2)
axes[0,0].plot(x, fesem_m_k, label="FeSEM", color="brown", marker='o', markersize=2)
axes[0,0].plot(x, feddrift_m_k, label="FedDrift", color="pink", marker='o', markersize=2)
axes[0,0].plot(x, cfl_m_k, label="CFL", color="cyan", marker='o', markersize=2)
axes[0,0].plot(x, pfedme_m_k, label="pFEDme", color="darkgoldenrod", marker='o', markersize=2)
axes[0,0].plot(x, apfl_m_k, label="APFL", color="teal", marker='o', markersize=2)
axes[0,0].plot(x, zzz_m_k, label="FLUX", color="blue", marker='o', markersize=2)
axes[0,0].plot(x, zzz_m_k_p, label="FLUX-prior", color="black", marker='o', markersize=2)
axes[0,0].set_title("MNIST")
axes[0,0].set_ylabel("Accuracy (Known Association)")
axes[0,0].set_xticks(x)
axes[0,0].set_xticklabels([1,2,3,4,5,6,7,8])
axes[0,0].set_ylim(0.2, 1)
axes[0,0].legend(loc="lower left")
 
# FMNIST
axes[0,1].errorbar(x, zzz_f_k_p, yerr=std_zzz_f_k_p, label="_nolegend_", color="black", marker="o", markersize=2,
                    elinewidth=0.8, capsize=2, alpha=0.2, ecolor="black")
axes[0,1].errorbar(x, zzz_f_k, yerr=std_zzz_f_k, label="_nolegend_", color="blue", marker="o", markersize=2,
                    elinewidth=0.8, capsize=2, alpha=0.2, ecolor="blue")
axes[0,1].errorbar(x, fedavg_f_k, yerr=std_fedavg_f_k, label="_nolegend_", color="orange", marker="o", markersize=2,
                    elinewidth=0.8, capsize=2, alpha=0.2, ecolor="orange")
axes[0,1].errorbar(x, ifca_f_k, yerr=std_ifca_f_k, label="_nolegend_", color="green", marker="o", markersize=2,
                    elinewidth=0.8, capsize=2, alpha=0.2, ecolor="green")
axes[0,1].errorbar(x, fedrc_f_k, yerr=std_fedrc_f_k, label="_nolegend_", color="red", marker="o", markersize=2,
                    elinewidth=0.8, capsize=2, alpha=0.2, ecolor="red")
axes[0,1].errorbar(x, fedem_f_k, yerr=std_fedem_f_k, label="_nolegend_", color="purple", marker="o", markersize=2,
                    elinewidth=0.8, capsize=2, alpha=0.2, ecolor="purple")
axes[0,1].errorbar(x, fesem_f_k, yerr=std_fesem_f_k, label="_nolegend_", color="brown", marker="o", markersize=2,
                    elinewidth=0.8, capsize=2, alpha=0.2, ecolor="brown")
axes[0,1].errorbar(x, feddrift_f_k, yerr=std_feddrift_f_k, label="_nolegend_", color="pink", marker="o", markersize=2,
                    elinewidth=0.8, capsize=2, alpha=0.2, ecolor="pink")
axes[0,1].errorbar(x, cfl_f_k, yerr=std_cfl_f_k, label="_nolegend_", color="cyan", marker="o", markersize=2,
                    elinewidth=0.8, capsize=2, alpha=0.2, ecolor="cyan")
axes[0,1].errorbar(x, pfedme_f_k, yerr=std_pfedme_f_k, label="_nolegend_", color="darkgoldenrod", marker="o", markersize=2,
                    elinewidth=0.8, capsize=2, alpha=0.2, ecolor="darkgoldenrod")
axes[0,1].errorbar(x, apfl_f_k, yerr=std_apfl_f_k, label="_nolegend_", color="teal", marker="o", markersize=2,
                    elinewidth=0.8, capsize=2, alpha=0.2, ecolor="teal")
axes[0,1].plot(x, fedavg_f_k, label="FedAvg", color="orange", marker='o', markersize=2)
axes[0,1].plot(x, ifca_f_k, label="IFCA", color="green", marker='o', markersize=2)
axes[0,1].plot(x, fedrc_f_k, label="FedRC", color="red", marker='o', markersize=2)
axes[0,1].plot(x, fedem_f_k, label="FedEM", color="purple", marker='o', markersize=2)
axes[0,1].plot(x, fesem_f_k, label="FeSEM", color="brown", marker='o', markersize=2)
axes[0,1].plot(x, feddrift_f_k, label="FedDrift", color="pink", marker='o', markersize=2)
axes[0,1].plot(x, cfl_f_k, label="CFL", color="cyan", marker='o', markersize=2)
axes[0,1].plot(x, pfedme_f_k, label="pFEDme", color="darkgoldenrod", marker='o', markersize=2)
axes[0,1].plot(x, apfl_f_k, label="APFL", color="teal", marker='o', markersize=2)
axes[0,1].plot(x, zzz_f_k, label="FLUX", color="blue", marker='o', markersize=2)
axes[0,1].plot(x, zzz_f_k_p, label="FLUX-prior", color="black", marker='o', markersize=2)
axes[0,1].set_title("FMNIST")
axes[0,1].set_ylabel("Accuracy (Known Association)")
axes[0,1].set_xticks(x)
axes[0,1].set_xticklabels([1,2,3,4,5,6,7,8])
axes[0,1].set_ylim(0.3, 0.9)
axes[0,1].legend(loc="lower left")
 
# CIFAR10
axes[0,2].errorbar(x, zzz_c_k_p, yerr=std_zzz_c_k_p, label="_nolegend_", color="black", marker="o", markersize=2,
                    elinewidth=0.8, capsize=2, alpha=0.2, ecolor="black")
axes[0,2].errorbar(x, zzz_c_k, yerr=std_zzz_c_k, label="_nolegend_", color="blue", marker="o", markersize=2,
                    elinewidth=0.8, capsize=2, alpha=0.2, ecolor="blue")
axes[0,2].errorbar(x, fedavg_c_k, yerr=std_fedavg_c_k, label="_nolegend_", color="orange", marker="o", markersize=2,
                    elinewidth=0.8, capsize=2, alpha=0.2, ecolor="orange")
axes[0,2].errorbar(x, ifca_c_k, yerr=std_ifca_c_k, label="_nolegend_", color="green", marker="o", markersize=2,
                    elinewidth=0.8, capsize=2, alpha=0.2, ecolor="green")
axes[0,2].errorbar(x, fedrc_c_k, yerr=std_fedrc_c_k, label="_nolegend_", color="red", marker="o", markersize=2,
                    elinewidth=0.8, capsize=2, alpha=0.2, ecolor="red")
axes[0,2].errorbar(x, fedem_c_k, yerr=std_fedem_c_k, label="_nolegend_", color="purple", marker="o", markersize=2,
                    elinewidth=0.8, capsize=2, alpha=0.2, ecolor="purple")
axes[0,2].errorbar(x, fesem_c_k, yerr=std_fesem_c_k, label="_nolegend_", color="brown", marker="o", markersize=2,
                    elinewidth=0.8, capsize=2, alpha=0.2, ecolor="brown")
axes[0,2].errorbar(x, feddrift_c_k, yerr=std_feddrift_c_k, label="_nolegend_", color="pink", marker="o", markersize=2,
                    elinewidth=0.8, capsize=2, alpha=0.2, ecolor="pink")
axes[0,2].errorbar(x, cfl_c_k, yerr=std_cfl_c_k, label="_nolegend_", color="cyan", marker="o", markersize=2,
                    elinewidth=0.8, capsize=2, alpha=0.2, ecolor="cyan")
axes[0,2].errorbar(x, pfedme_c_k, yerr=std_pfedme_c_k, label="_nolegend_", color="darkgoldenrod", marker="o", markersize=2,
                    elinewidth=0.8, capsize=2, alpha=0.2, ecolor="darkgoldenrod")
axes[0,2].errorbar(x, apfl_c_k, yerr=std_apfl_c_k, label="_nolegend_", color="teal", marker="o", markersize=2,
                    elinewidth=0.8, capsize=2, alpha=0.2, ecolor="teal")
axes[0,2].plot(x, fedavg_c_k, label="FedAvg", color="orange", marker='o', markersize=2)
axes[0,2].plot(x, ifca_c_k, label="IFCA", color="green", marker='o', markersize=2)
axes[0,2].plot(x, fedrc_c_k, label="FedRC", color="red", marker='o', markersize=2)
axes[0,2].plot(x, fedem_c_k, label="FedEM", color="purple", marker='o', markersize=2)
axes[0,2].plot(x, fesem_c_k, label="FeSEM", color="brown", marker='o', markersize=2)
axes[0,2].plot(x, feddrift_c_k, label="FedDrift", color="pink", marker='o', markersize=2)
axes[0,2].plot(x, cfl_c_k, label="CFL", color="cyan", marker='o', markersize=2)
axes[0,2].plot(x, pfedme_c_k, label="pFEDme", color="darkgoldenrod", marker='o', markersize=2)
axes[0,2].plot(x, apfl_c_k, label="APFL", color="teal", marker='o', markersize=2)
axes[0,2].plot(x, zzz_c_k, label="FLUX", color="blue", marker='o', markersize=2)
axes[0,2].plot(x, zzz_c_k_p, label="FLUX-prior", color="black", marker='o', markersize=2)
axes[0,2].set_title("CIFAR-10")
axes[0,2].set_ylabel("Accuracy (Known Association)")
axes[0,2].set_xticks(x)
axes[0,2].set_xticklabels([1,2,3,4,5,6,7,8])
axes[0,2].set_ylim(0.1, 0.5)
axes[0,2].legend(loc="lower left")

# CIFAR100
axes[0,3].errorbar(x, zzz_cc_k_p, yerr=std_zzz_cc_k_p, label="_nolegend_", color="black", marker="o", markersize=2,
                    elinewidth=0.8, capsize=2, alpha=0.2, ecolor="black")
axes[0,3].errorbar(x, zzz_cc_k, yerr=std_zzz_cc_k, label="_nolegend_", color="blue", marker="o", markersize=2,
                    elinewidth=0.8, capsize=2, alpha=0.2, ecolor="blue")
axes[0,3].errorbar(x, fedavg_cc_k, yerr=std_fedavg_cc_k, label="_nolegend_", color="orange", marker="o", markersize=2,
                    elinewidth=0.8, capsize=2, alpha=0.2, ecolor="orange")
axes[0,3].errorbar(x, ifca_cc_k, yerr=std_ifca_cc_k, label="_nolegend_", color="green", marker="o", markersize=2,
                    elinewidth=0.8, capsize=2, alpha=0.2, ecolor="green")
axes[0,3].errorbar(x, fedrc_cc_k, yerr=std_fedrc_cc_k, label="_nolegend_", color="red", marker="o", markersize=2,
                    elinewidth=0.8, capsize=2, alpha=0.2, ecolor="red")
axes[0,3].errorbar(x, fedem_cc_k, yerr=std_fedem_cc_k, label="_nolegend_", color="purple", marker="o", markersize=2,
                    elinewidth=0.8, capsize=2, alpha=0.2, ecolor="purple")
axes[0,3].errorbar(x, fesem_cc_k, yerr=std_fesem_cc_k, label="_nolegend_", color="brown", marker="o", markersize=2,
                    elinewidth=0.8, capsize=2, alpha=0.2, ecolor="brown")
axes[0,3].errorbar(x, feddrift_cc_k, yerr=std_feddrift_cc_k, label="_nolegend_", color="pink", marker="o", markersize=2,
                    elinewidth=0.8, capsize=2, alpha=0.2, ecolor="pink")
axes[0,3].errorbar(x, cfl_cc_k, yerr=std_cfl_cc_k, label="_nolegend_", color="cyan", marker="o", markersize=2,
                    elinewidth=0.8, capsize=2, alpha=0.2, ecolor="cyan")
axes[0,3].errorbar(x, pfedme_cc_k, yerr=std_pfedme_cc_k, label="_nolegend_", color="darkgoldenrod", marker="o", markersize=2,
                    elinewidth=0.8, capsize=2, alpha=0.2, ecolor="darkgoldenrod")
axes[0,3].errorbar(x, apfl_cc_k, yerr=std_apfl_cc_k, label="_nolegend_", color="teal", marker="o", markersize=2,
                    elinewidth=0.8, capsize=2, alpha=0.2, ecolor="teal")
axes[0,3].plot(x, fedavg_cc_k, label="FedAvg", color="orange", marker='o', markersize=2)
axes[0,3].plot(x, ifca_cc_k, label="IFCA", color="green", marker='o', markersize=2)
axes[0,3].plot(x, fedrc_cc_k, label="FedRC", color="red", marker='o', markersize=2)
axes[0,3].plot(x, fedem_cc_k, label="FedEM", color="purple", marker='o', markersize=2)
axes[0,3].plot(x, fesem_cc_k, label="FeSEM", color="brown", marker='o', markersize=2)
axes[0,3].plot(x, feddrift_cc_k, label="FedDrift", color="pink", marker='o', markersize=2)
axes[0,3].plot(x, cfl_cc_k, label="CFL", color="cyan", marker='o', markersize=2)
axes[0,3].plot(x, pfedme_cc_k, label="pFEDme", color="darkgoldenrod", marker='o', markersize=2)
axes[0,3].plot(x, apfl_cc_k, label="APFL", color="teal", marker='o', markersize=2)
axes[0,3].plot(x, zzz_cc_k, label="FLUX", color="blue", marker='o', markersize=2)
axes[0,3].plot(x, zzz_cc_k_p, label="FLUX-prior", color="black", marker='o', markersize=2)
axes[0,3].set_title("CIFAR-100")
axes[0,3].set_ylabel("Accuracy (Known Association)")
axes[0,3].set_xticks(x)
axes[0,3].set_xticklabels([1,2,3,4,5,6,7,8])
axes[0,3].set_ylim(0.1, 0.56)
axes[0,3].legend(loc="lower left")
 
# Second row, testing phase inference
 
# MNIST
axes[1,0].errorbar(x, zzz_m_t_p, yerr=std_zzz_m_t_p, label="_nolegend_", color="black", marker="o", markersize=2,
                    elinewidth=0.8, capsize=2, alpha=0.2, ecolor="black")
axes[1,0].errorbar(x, zzz_m_t, yerr=std_zzz_m_t, label="_nolegend_", color="blue", marker="o", markersize=2,
                    elinewidth=0.8, capsize=2, alpha=0.2, ecolor="blue")
axes[1,0].errorbar(x, fedavg_m_t, yerr=std_fedavg_m_t, label="_nolegend_", color="orange", marker="o", markersize=2,
                    elinewidth=0.8, capsize=2, alpha=0.2, ecolor="orange")
axes[1,0].errorbar(x, ifca_m_t, yerr=std_ifca_m_t, label="_nolegend_", color="green", marker="o", markersize=2,
                    elinewidth=0.8, capsize=2, alpha=0.2, ecolor="green")
axes[1,0].errorbar(x, fedrc_m_t, yerr=std_fedrc_m_t, label="_nolegend_", color="red", marker="o", markersize=2,
                    elinewidth=0.8, capsize=2, alpha=0.2, ecolor="red")
axes[1,0].errorbar(x, fedem_m_t, yerr=std_fedem_m_t, label="_nolegend_", color="purple", marker="o", markersize=2,
                    elinewidth=0.8, capsize=2, alpha=0.2, ecolor="purple")
axes[1,0].errorbar(x, fesem_m_t, yerr=std_fesem_m_t, label="_nolegend_", color="brown", marker="o", markersize=2,
                    elinewidth=0.8, capsize=2, alpha=0.2, ecolor="brown")
axes[1,0].errorbar(x, feddrift_m_t, yerr=std_feddrift_m_t, label="_nolegend_", color="pink", marker="o", markersize=2,
                    elinewidth=0.8, capsize=2, alpha=0.2, ecolor="pink")
axes[1,0].errorbar(x, cfl_m_t, yerr=std_cfl_m_t, label="_nolegend_", color="cyan", marker="o", markersize=2,
                    elinewidth=0.8, capsize=2, alpha=0.2, ecolor="cyan")
axes[1,0].errorbar(x, apfl_m_t, yerr=std_apfl_m_t, label="_nolegend_", color="teal", marker="o", markersize=2,
                    elinewidth=0.8, capsize=2, alpha=0.2, ecolor="teal")
axes[1,0].errorbar(x, atp_m_t, yerr=std_atp_m_t, label="_nolegend_", color="gold", marker="o", markersize=2,
                    elinewidth=0.8, capsize=2, alpha=0.2, ecolor="gold")
axes[1,0].plot(x, fedavg_m_t, label="FedAvg", color="orange", marker='o', markersize=2)
axes[1,0].plot(x, ifca_m_t, label="IFCA", color="green", marker='o', markersize=2)
axes[1,0].plot(x, fedrc_m_t, label="FedRC", color="red", marker='o', markersize=2)
axes[1,0].plot(x, fedem_m_t, label="FedEM", color="purple", marker='o', markersize=2)
axes[1,0].plot(x, fesem_m_t, label="FeSEM", color="brown", marker='o', markersize=2)
axes[1,0].plot(x, feddrift_m_t, label="FedDrift", color="pink", marker='o', markersize=2)
axes[1,0].plot(x, cfl_m_t, label="CFL", color="cyan", marker='o', markersize=2)
axes[1,0].plot(x, apfl_m_t, label="APFL", color="teal", marker='o', markersize=2)
axes[1,0].plot(x, atp_m_t, label="ATP", color="gold", marker='o', markersize=2)
axes[1,0].plot(x, zzz_m_t, label="FLUX", color="blue", marker='o', markersize=2)
axes[1,0].plot(x, zzz_m_t_p, label="FLUX-prior", color="black", marker='o', markersize=2)
axes[1,0].set_xlabel("Heterogeneity Level")
axes[1,0].set_ylabel("Accuracy (Test Phase)")
axes[1,0].set_xticks(x)
axes[1,0].set_xticklabels([1,2,3,4,5,6,7,8])
axes[1,0].set_ylim(0.2, 1)
axes[1,0].legend(loc="lower left")
 
# FMNIST
axes[1,1].errorbar(x, zzz_f_t_p, yerr=std_zzz_f_t_p, label="_nolegend_", color="black", marker="o", markersize=2,
                    elinewidth=0.8, capsize=2, alpha=0.2, ecolor="black")
axes[1,1].errorbar(x, zzz_f_t, yerr=std_zzz_f_t, label="_nolegend_", color="blue", marker="o", markersize=2,
                    elinewidth=0.8, capsize=2, alpha=0.2, ecolor="blue")
axes[1,1].errorbar(x, fedavg_f_t, yerr=std_fedavg_f_t, label="_nolegend_", color="orange", marker="o", markersize=2,
                    elinewidth=0.8, capsize=2, alpha=0.2, ecolor="orange")
axes[1,1].errorbar(x, ifca_f_t, yerr=std_ifca_f_t, label="_nolegend_", color="green", marker="o", markersize=2,
                    elinewidth=0.8, capsize=2, alpha=0.2, ecolor="green")
axes[1,1].errorbar(x, fedrc_f_t, yerr=std_fedrc_f_t, label="_nolegend_", color="red", marker="o", markersize=2,
                    elinewidth=0.8, capsize=2, alpha=0.2, ecolor="red")
axes[1,1].errorbar(x, fedem_f_t, yerr=std_fedem_f_t, label="_nolegend_", color="purple", marker="o", markersize=2,
                    elinewidth=0.8, capsize=2, alpha=0.2, ecolor="purple")
axes[1,1].errorbar(x, fesem_f_t, yerr=std_fesem_f_t, label="_nolegend_", color="brown", marker="o", markersize=2,
                    elinewidth=0.8, capsize=2, alpha=0.2, ecolor="brown")
axes[1,1].errorbar(x, feddrift_f_t, yerr=std_feddrift_f_t, label="_nolegend_", color="pink", marker="o", markersize=2,
                    elinewidth=0.8, capsize=2, alpha=0.2, ecolor="pink")
axes[1,1].errorbar(x, cfl_f_t, yerr=std_cfl_f_t, label="_nolegend_", color="cyan", marker="o", markersize=2,
                    elinewidth=0.8, capsize=2, alpha=0.2, ecolor="cyan")
axes[1,1].errorbar(x, apfl_f_t, yerr=std_apfl_f_t, label="_nolegend_", color="teal", marker="o", markersize=2,
                    elinewidth=0.8, capsize=2, alpha=0.2, ecolor="teal")
axes[1,1].errorbar(x, atp_f_t, yerr=std_atp_f_t, label="_nolegend_", color="gold", marker="o", markersize=2,
                    elinewidth=0.8, capsize=2, alpha=0.2, ecolor="gold")
axes[1,1].plot(x, fedavg_f_t, label="FedAvg", color="orange", marker='o', markersize=2)
axes[1,1].plot(x, ifca_f_t, label="IFCA", color="green", marker='o', markersize=2)
axes[1,1].plot(x, fedrc_f_t, label="FedRC", color="red", marker='o', markersize=2)
axes[1,1].plot(x, fedem_f_t, label="FedEM", color="purple", marker='o', markersize=2)
axes[1,1].plot(x, fesem_f_t, label="FeSEM", color="brown", marker='o', markersize=2)
axes[1,1].plot(x, feddrift_f_t, label="FedDrift", color="pink", marker='o', markersize=2)
axes[1,1].plot(x, cfl_f_t, label="CFL", color="cyan", marker='o', markersize=2)
axes[1,1].plot(x, apfl_f_t, label="APFL", color="teal", marker='o', markersize=2)
axes[1,1].plot(x, atp_f_t, label="ATP", color="gold", marker='o', markersize=2)
axes[1,1].plot(x, zzz_f_t, label="FLUX", color="blue", marker='o', markersize=2)
axes[1,1].plot(x, zzz_f_t_p, label="FLUX-prior", color="black", marker='o', markersize=2)
axes[1,1].set_xlabel("Heterogeneity Level")
axes[1,1].set_ylabel("Accuracy (Test Phase)")
axes[1,1].set_xticks(x)
axes[1,1].set_xticklabels([1,2,3,4,5,6,7,8])
axes[1,1].set_ylim(0.3, 0.9)
axes[1,1].legend(loc="lower left")
 
# CIFAR10
axes[1,2].errorbar(x, zzz_c_t_p, yerr=std_zzz_c_t_p, label="_nolegend_", color="blue", marker="o", markersize=2,
                    elinewidth=0.8, capsize=2, alpha=0.2, ecolor="blue")
axes[1,2].errorbar(x, zzz_c_t, yerr=std_zzz_c_t, label="_nolegend_", color="blue", marker="o", markersize=2,
                    elinewidth=0.8, capsize=2, alpha=0.2, ecolor="blue")
axes[1,2].errorbar(x, fedavg_c_t, yerr=std_fedavg_c_t, label="_nolegend_", color="orange", marker="o", markersize=2,
                    elinewidth=0.8, capsize=2, alpha=0.2, ecolor="orange")
axes[1,2].errorbar(x, ifca_c_t, yerr=std_ifca_c_t, label="_nolegend_", color="green", marker="o", markersize=2,
                    elinewidth=0.8, capsize=2, alpha=0.2, ecolor="green")
axes[1,2].errorbar(x, fedrc_c_t, yerr=std_fedrc_c_t, label="_nolegend_", color="red", marker="o", markersize=2,
                    elinewidth=0.8, capsize=2, alpha=0.2, ecolor="red")
axes[1,2].errorbar(x, fedem_c_t, yerr=std_fedem_c_t, label="_nolegend_", color="purple", marker="o", markersize=2,
                    elinewidth=0.8, capsize=2, alpha=0.2, ecolor="purple")
axes[1,2].errorbar(x, fesem_c_t, yerr=std_fesem_c_t, label="_nolegend_", color="brown", marker="o", markersize=2,
                    elinewidth=0.8, capsize=2, alpha=0.2, ecolor="brown")
axes[1,2].errorbar(x, feddrift_c_t, yerr=std_feddrift_c_t, label="_nolegend_", color="pink", marker="o", markersize=2,
                    elinewidth=0.8, capsize=2, alpha=0.2, ecolor="pink")
axes[1,2].errorbar(x, cfl_c_t, yerr=std_cfl_c_t, label="_nolegend_", color="cyan", marker="o", markersize=2,
                    elinewidth=0.8, capsize=2, alpha=0.2, ecolor="cyan")
axes[1,2].errorbar(x, apfl_c_t, yerr=std_apfl_c_t, label="_nolegend_", color="teal", marker="o", markersize=2,
                    elinewidth=0.8, capsize=2, alpha=0.2, ecolor="teal")
axes[1,2].errorbar(x, atp_c_t, yerr=std_atp_c_t, label="_nolegend_", color="gold", marker="o", markersize=2,
                    elinewidth=0.8, capsize=2, alpha=0.2, ecolor="gold")
axes[1,2].plot(x, fedavg_c_t, label="FedAvg", color="orange", marker='o', markersize=2)
axes[1,2].plot(x, ifca_c_t, label="IFCA", color="green", marker='o', markersize=2)
axes[1,2].plot(x, fedrc_c_t, label="FedRC", color="red", marker='o', markersize=2)
axes[1,2].plot(x, fedem_c_t, label="FedEM", color="purple", marker='o', markersize=2)
axes[1,2].plot(x, fesem_c_t, label="FeSEM", color="brown", marker='o', markersize=2)
axes[1,2].plot(x, feddrift_c_t, label="FedDrift", color="pink", marker='o', markersize=2)
axes[1,2].plot(x, cfl_c_t, label="CFL", color="cyan", marker='o', markersize=2)
axes[1,2].plot(x, apfl_c_t, label="APFL", color="teal", marker='o', markersize=2)
axes[1,2].plot(x, atp_c_t, label="ATP", color="gold", marker='o', markersize=2)
axes[1,2].plot(x, zzz_c_t, label="FLUX", color="blue", marker='o', markersize=2)
axes[1,2].plot(x, zzz_c_t_p, label="FLUX-prior", color="black", marker='o', markersize=2)
axes[1,2].set_xlabel("Heterogeneity Level")
axes[1,2].set_ylabel("Accuracy (Test Phase)")
axes[1,2].set_xticks(x)
axes[1,2].set_xticklabels([1,2,3,4,5,6,7,8])
axes[1,2].set_ylim(0.1, 0.5)
axes[1,2].legend(loc="lower left")

# CIFAR100
axes[1,3].errorbar(x, zzz_cc_t_p, yerr=std_zzz_cc_t_p, label="_nolegend_", color="blue", marker="o", markersize=2,
                    elinewidth=0.8, capsize=2, alpha=0.2, ecolor="blue")
axes[1,3].errorbar(x, zzz_cc_t, yerr=std_zzz_cc_t, label="_nolegend_", color="blue", marker="o", markersize=2,
                    elinewidth=0.8, capsize=2, alpha=0.2, ecolor="blue")
axes[1,3].errorbar(x, fedavg_cc_t, yerr=std_fedavg_cc_t, label="_nolegend_", color="orange", marker="o", markersize=2,
                    elinewidth=0.8, capsize=2, alpha=0.2, ecolor="orange")
axes[1,3].errorbar(x, ifca_cc_t, yerr=std_ifca_cc_t, label="_nolegend_", color="green", marker="o", markersize=2,
                    elinewidth=0.8, capsize=2, alpha=0.2, ecolor="green")
axes[1,3].errorbar(x, fedrc_cc_t, yerr=std_fedrc_cc_t, label="_nolegend_", color="red", marker="o", markersize=2,
                    elinewidth=0.8, capsize=2, alpha=0.2, ecolor="red")
axes[1,3].errorbar(x, fedem_cc_t, yerr=std_fedem_cc_t, label="_nolegend_", color="purple", marker="o", markersize=2,
                    elinewidth=0.8, capsize=2, alpha=0.2, ecolor="purple")
axes[1,3].errorbar(x, fesem_cc_t, yerr=std_fesem_cc_t, label="_nolegend_", color="brown", marker="o", markersize=2,
                    elinewidth=0.8, capsize=2, alpha=0.2, ecolor="brown")
axes[1,3].errorbar(x, feddrift_cc_t, yerr=std_feddrift_cc_t, label="_nolegend_", color="pink", marker="o", markersize=2,
                    elinewidth=0.8, capsize=2, alpha=0.2, ecolor="pink")
axes[1,3].errorbar(x, cfl_cc_t, yerr=std_cfl_cc_t, label="_nolegend_", color="cyan", marker="o", markersize=2,
                    elinewidth=0.8, capsize=2, alpha=0.2, ecolor="cyan")
axes[1,3].errorbar(x, apfl_cc_t, yerr=std_apfl_cc_t, label="_nolegend_", color="teal", marker="o", markersize=2,
                    elinewidth=0.8, capsize=2, alpha=0.2, ecolor="teal")
axes[1,3].errorbar(x, atp_cc_t, yerr=std_atp_cc_t, label="_nolegend_", color="gold", marker="o", markersize=2,
                    elinewidth=0.8, capsize=2, alpha=0.2, ecolor="gold")
axes[1,3].plot(x, fedavg_cc_t, label="FedAvg", color="orange", marker='o', markersize=2)
axes[1,3].plot(x, ifca_cc_t, label="IFCA", color="green", marker='o', markersize=2)
axes[1,3].plot(x, fedrc_cc_t, label="FedRC", color="red", marker='o', markersize=2)
axes[1,3].plot(x, fedem_cc_t, label="FedEM", color="purple", marker='o', markersize=2)
axes[1,3].plot(x, fesem_cc_t, label="FeSEM", color="brown", marker='o', markersize=2)
axes[1,3].plot(x, feddrift_cc_t, label="FedDrift", color="pink", marker='o', markersize=2)
axes[1,3].plot(x, cfl_cc_t, label="CFL", color="cyan", marker='o', markersize=2)
axes[1,3].plot(x, atp_cc_t, label="ATP", color="gold", marker='o', markersize=2)
axes[1,3].plot(x, apfl_cc_t, label="APFL", color="teal", marker='o', markersize=2)
axes[1,3].plot(x, zzz_cc_t, label="FLUX", color="blue", marker='o', markersize=2)
axes[1,3].plot(x, zzz_cc_t_p, label="FLUX-prior", color="black", marker='o', markersize=2)
axes[1,3].set_xlabel("Heterogeneity Level")
axes[1,3].set_ylabel("Accuracy (Test Phase)")
axes[1,3].set_xticks(x)
axes[1,3].set_xticklabels([1,2,3,4,5,6,7,8])
axes[1,3].set_ylim(0.1, 0.56)
axes[1,3].legend(loc="lower left")
 
# Adjust layout
plt.tight_layout()
plt.savefig("/Users/dariofenoglio/Desktop/Hetero.pdf", bbox_inches="tight")
plt.show()
