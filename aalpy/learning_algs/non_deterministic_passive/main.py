import time
from random import choice, randint

from aalpy.SULs import OnfsmSUL
from aalpy.learning_algs.non_deterministic_passive.OnfsmAlternative import run_non_det_Lstar_alternative
from aalpy.learning_algs.non_deterministic_passive.OnfsmObservationTable import NonDetObservationTable
from aalpy.learning_algs.non_deterministic_passive.TraceTree import TraceTree, SULWrapper
from aalpy.oracles import RandomWalkEqOracle, RandomWordEqOracle
from aalpy.utils import load_automaton_from_file

#original_model = load_automaton_from_file('../../../fail_safe_model.dot', 'onfsm')
original_model = load_automaton_from_file('original.dot', 'onfsm')
alphabet = original_model.get_input_alphabet()

sul = OnfsmSUL(original_model)

from random import seed
# seed(1)
n_samples = 0

samples = []
for _ in range(n_samples):
    inputs = tuple(choice(alphabet) for _ in range(randint(5, 10)))
    outputs = sul.query(inputs)

eq_oracle = RandomWordEqOracle(alphabet, sul, num_walks=1000, min_walk_len=4, max_walk_len=10)

model = run_non_det_Lstar_alternative(alphabet, sul, eq_oracle, samples=samples, n_sampling=10)
model.visualize()