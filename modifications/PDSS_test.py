from modifications import PropertyDirectedSamplingStormpy as PDSS
from aalpy.utils import get_faulty_coffee_machine_MDP
from aalpy.SULs import MdpSUL

pdss = PDSS.PDS(target="coffee", sul=MdpSUL(get_faulty_coffee_machine_MDP()), alphabet=get_faulty_coffee_machine_MDP().get_input_alphabet(), k=5, quit_prob=0.3)
p_rand_new, s_new, s_all_new = pdss.find_cex(get_faulty_coffee_machine_MDP())
print(f"p_rand = {p_rand_new}\n")
print("s_new:")
for entry in s_new:
    print(entry)
print(f"\ns_all_new:")
for entry in s_all_new:
    print(f"s_new_{s_all_new.index(entry)}:")
    for e in entry:
        print(e)
