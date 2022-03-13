"""Beowulf is a library containing data generating mechanisms. This library is to manage the various data sets and
streamline their loading procedures.
"""

from .version import __version__

from .dgm import (sofrygin_observational, sofrygin_randomized,
                  modified_randomized, modified_observational,
                  continuous_observational, continuous_randomized,
                  direct_observational, direct_randomized,
                  indirect_observational, indirect_randomized,
                  independent_observational, independent_randomized,
                  threshold_observational, threshold_randomized)
from .load_networks import (load_random_network, load_uniform_network, generate_sofrygin_network,
                            load_uniform_naloxone, load_uniform_statin, load_uniform_diet, load_uniform_vaccine,
                            load_random_naloxone, load_random_statin, load_random_diet, load_random_vaccine)
from .truth import truth_values


def simulation_setup(slurm_id_str):
    """Takes an input string (of integers) and returns the simulation setup information.
    """
    # Setting up network
    if slurm_id_str[0] == "1":
        network = "uniform"
        n_nodes = 500
        save = network
    elif slurm_id_str[0] == "2":
        network = "random"
        n_nodes = 500
        save = network
    elif slurm_id_str[0] == "4":
        network = "uniform"
        n_nodes = 1000
        save = network + "1k"
    elif slurm_id_str[0] == "5":
        network = "random"
        n_nodes = 1000
        save = network + "1k"
    elif slurm_id_str[0] == "6":
        network = "uniform"
        n_nodes = 2000
        save = network + "2k"
    elif slurm_id_str[0] == "7":
        network = "random"
        n_nodes = 2000
        save = network + "2k"
    else:
        raise ValueError("Invalid network specification. Got " + slurm_id_str[0] +
                         " but expected either 1, 2, 4, 5, 6, or 7.")

    # Determining if degree restriction
    if slurm_id_str[1] == "0":
        save = save + "_u"
        degree_restrict = None
    elif slurm_id_str[1] == "1":
        save = save + "_r"
        if network == "random" and n_nodes == 500:
            degree_restrict = (0, 18)
        elif network == "random":
            degree_restrict = (0, 22)
        else:
            raise ValueError("That network - degree restriction is not available :(")
    else:
        raise ValueError("Invalid degree restriction specification. Got " + slurm_id_str[1] +
                         " but expected either 1 or 0.")

    # Determining if shift
    if slurm_id_str[2] == "0":
        save = save + "_u"
        shift = False
    elif slurm_id_str[2] == "1":
        save = save + "_s"
        shift = True
    else:
        raise ValueError("Invalid shift specification. Got '" + slurm_id_str[2] +
                         "' but expected either 1 or 0.")

    # Determining model specification
    if slurm_id_str[3] == "1":
        model = "cc"
        save = save + "_" + model
    elif slurm_id_str[3] == "2":
        model = "cw"
        save = save + "_" + model
    elif slurm_id_str[3] == "3":
        model = "wc"
        save = save + "_" + model
    elif slurm_id_str[3] == "4":
        model = "np"
        save = save + "_" + model
    elif slurm_id_str[3] == "9":
        model = "ind"
        save = save + "_" + model
    else:
        raise ValueError("Invalid model specification. Got '" + slurm_id_str[3] +
                         "' but expected 1, 2, 4 or 9.")

    return network, n_nodes, degree_restrict, shift, model, save
