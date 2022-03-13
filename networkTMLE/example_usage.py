############################################################################################
# Example of Targetula with simulated data generating mechanisms
#   This file demonstrates basic usage of the Targetula class with some of the available
#       data generating mechanisms
############################################################################################

from amonhen import NetworkTMLE
from beowulf import (sofrygin_observational, generate_sofrygin_network,
                     load_uniform_statin, load_random_naloxone, load_uniform_diet, load_random_vaccine)
from beowulf.dgm import statin_dgm, naloxone_dgm, diet_dgm, vaccine_dgm


###########################################
# Sofrygin & van der Laan (2017) Example

# generating data
sample_size = 1000
G = generate_sofrygin_network(n=sample_size, max_degree=2, seed=20200115)
H = sofrygin_observational(G)

# network-TMLE applied to generated data
tmle = NetworkTMLE(network=H, exposure='A', outcome='Y', verbose=False)
tmle.exposure_model('W + W_sum')
tmle.exposure_map_model('A + W + W_sum')  # by default a series of logistic models is used
tmle.outcome_model('A + A_sum + W + W_sum')
# Policy of setting everyone's probability of exposure to 0.35
tmle.fit(p=0.35, samples=100)
tmle.summary()
# Policy of setting everyone's probability of exposure to 0.65
tmle.fit(p=0.65, samples=100)
tmle.summary()

###########################################
# Statin-ASCVD -- DGM

# Loading uniform network with statin W
G = load_uniform_statin()
# Simulation single instance of exposure and outcome
H = statin_dgm(network=G)

# network-TMLE applies to generated data
tmle = NetworkTMLE(H, exposure='statin', outcome='cvd')
tmle.exposure_model("L + A_30 + R_1 + R_2 + R_3")
tmle.exposure_map_model("statin + L + A_30 + R_1 + R_2 + R_3",
                        measure='sum', distribution='poisson')  # Applying a Poisson model
tmle.outcome_model("statin + statin_sum + A_sqrt + R + L")
tmle.fit(p=0.35, bound=0.01)
tmle.summary()

###########################################
# Naloxone-Overdose -- DGM

# Loading clustered power-law network with naloxone W
G = load_random_naloxone()
# Simulation single instance of exposure and outcome
H = naloxone_dgm(network=G)

# network-TMLE applies to generated data
tmle = NetworkTMLE(H, exposure='naloxone', outcome='overdose',
                   degree_restrict=(0, 18))  # Applying restriction by degree
tmle.exposure_model("P + P:G + O_mean + G_mean")
tmle.exposure_map_model("naloxone + P + P:G + O_mean + G_mean",
                        measure='sum', distribution='poisson')  # Applying a Poisson model
tmle.outcome_model("naloxone_sum + P + G + O_mean + G_mean")
tmle.fit(p=0.35, bound=0.01)
tmle.summary()

###########################################
# Diet-BMI -- DGM

# Loading clustered power-law network with naloxone W
G = load_uniform_diet()
# Simulation single instance of exposure and outcome
H = diet_dgm(network=G)

# network-TMLE applies to generated data
tmle = NetworkTMLE(H, exposure='diet', outcome='bmi')
tmle.define_threshold(variable='diet', threshold=3,
                      definition='sum')  # Defining threshold measure of at least 3 contacts with a diet
tmle.exposure_model("B_30 + G:E + E_mean")
tmle.exposure_map_model("diet + B_30 + G:E + E_mean", measure='t3',
                        distribution='threshold')  # Logistic model for the threshold summary measure
tmle.outcome_model("diet + diet_t3 + B + G + E + E_sum + B_mean_dist")
tmle.fit(p=0.65, bound=0.01)
tmle.summary()

###########################################
# Vaccine-Infection -- DGM

# Loading clustered power-law network with naloxone W
G = load_random_vaccine()
# Simulation single instance of exposure and outcome
H = vaccine_dgm(network=G)

tmle = NetworkTMLE(H, exposure='vaccine', outcome='D', verbose=False, degree_restrict=(0, 18))
tmle.exposure_model("A + H + H_mean + degree")
tmle.exposure_map_model("vaccine + A + H + H_mean + degree",
                        measure='sum', distribution='poisson')
tmle.outcome_model("vaccine + vaccine_mean + A + H + A_mean + H_mean + degree")
tmle.fit(p=0.55, bound=0.01)
tmle.summary()
