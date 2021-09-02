from PDS import PropertyDirectedSampling as PDS
from aalpy.utils import load_automaton_from_file
import json
from aalpy.utils.ModelChecking import mdp_2_prism_format
from aalpy.SULs import MdpSUL

original_mdp = load_automaton_from_file("/home/manuel/TUGraz/SPST/AALpy/DotModels/MDPs/first_grid.dot", automaton_type='mdp')

test = mdp_2_prism_format(original_mdp, "test2", "/home/manuel/TUGraz/SPST/AALpy/PDS/prism_files/mdp_test.txt")

with open("testfile", "w+") as infile:
    json.dump(test, infile)

input_alphabet = original_mdp.get_input_alphabet()
mdp_sul = MdpSUL(original_mdp)
pdss = PDS.PDS(target="goal", sul=mdp_sul, alphabet=input_alphabet, k=10, quit_prob=0.1)
p_rand_new, s_new, s_all_new = pdss.execute_pds(original_mdp)

print(f"p_rand = {p_rand_new}\n")
print("s_new:")
for entry in s_new:
    print(entry)
print(f"\ns_all_new:")
for entry in s_all_new:
    print(f"s_new_{s_all_new.index(entry)}:")
    for e in entry:
        print(e)
