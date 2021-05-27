import json
import os

from aalpy.utils import visualize_automaton, get_faulty_coffee_machine_MDP
from Examples import faulty_coffee_machine_mdp_example
from aalpy.utils.ModelChecking import mdp_2_prism_format

dir_path = os.path.dirname(os.path.realpath(__file__))

faulty_coffee_machine = faulty_coffee_machine_mdp_example()
visualize_automaton(faulty_coffee_machine)

mdp = get_faulty_coffee_machine_MDP()
test = mdp_2_prism_format(mdp, "test2", dir_path + "/test.txt")

with open("testfile", "w+") as infile:
    json.dump(test, infile)
