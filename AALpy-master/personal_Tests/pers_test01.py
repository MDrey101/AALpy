from aalpy.oracles import StatePrefixEqOracle
from aalpy.SULs import RegexSUL
from aalpy.learning_algs import run_Lstar
from aalpy.utils import visualize_automaton

regex = 'ab(b|a)'
alphabet = ['a', 'b']

regex_sul = RegexSUL(regex)

eq_oracle = StatePrefixEqOracle(alphabet, regex_sul, walks_per_state=100, walk_len=20)

learned_regex = run_Lstar(alphabet, regex_sul, eq_oracle,  cex_processing=None, automaton_type='dfa', print_level=3)

visualize_automaton(learned_regex)
