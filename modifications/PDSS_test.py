from modifications import PropertyDirectedSamplingStormpy as PDSS
from aalpy.utils import get_faulty_coffee_machine_MDP, get_weird_coffee_machine_MDP, load_automaton_from_file
from aalpy.SULs import MdpSUL

# pdss = PDSS.PDS(target="coffee", sul=MdpSUL(get_faulty_coffee_machine_MDP()), alphabet=get_faulty_coffee_machine_MDP().get_input_alphabet(), k=5, quit_prob=0.3)
# p_rand_new, s_new, s_all_new = pdss.execute_pds(get_faulty_coffee_machine_MDP())

original_mdp = load_automaton_from_file("/home/manuel/TUGraz/SPST/AALpy/DotModels/MDPs/first_grid.dot", automaton_type='mdp')
input_alphabet = original_mdp.get_input_alphabet()
mdp_sul = MdpSUL(original_mdp)
pdss = PDSS.PDS(target="grass__goal", sul=mdp_sul, alphabet=input_alphabet, k=10, quit_prob=0.1)
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
