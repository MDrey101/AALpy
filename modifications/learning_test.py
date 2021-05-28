from aalpy.SULs import MdpSUL
from aalpy.learning_algs import run_stochastic_Lstar
import PropertyDirectedSamplingStormpy as PDSS
from aalpy.utils import get_faulty_coffee_machine_MDP

input_alphabet = get_faulty_coffee_machine_MDP().get_input_alphabet()
mdp_sul = MdpSUL(get_faulty_coffee_machine_MDP())
oracle = PDSS.PDS(target="coffee", sul=mdp_sul, alphabet=input_alphabet, k=5, quit_prob=0.3, p_rand=1.0, stop_on_cex=True)

run_stochastic_Lstar(input_alphabet=input_alphabet, sul=mdp_sul, eq_oracle=oracle, samples_cex_strategy=None)
