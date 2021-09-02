from aalpy.SULs import MdpSUL
from aalpy.learning_algs import run_stochastic_Lstar
from PropertyDirectedSampling import PDS
from aalpy.utils import load_automaton_from_file

original_mdp = load_automaton_from_file("/home/manuel/TUGraz/SPST/AALpy/DotModels/MDPs/first_grid.dot", automaton_type='mdp')
input_alphabet = original_mdp.get_input_alphabet()
mdp_sul = MdpSUL(original_mdp)

oracle = PDS(target="goal", sul=mdp_sul, alphabet=input_alphabet, k=12, quit_prob=0.3, p_rand=0.5, stop_on_cex=True)

run_stochastic_Lstar(input_alphabet=input_alphabet, sul=mdp_sul, eq_oracle=oracle, samples_cex_strategy=None)
