from random import choice, randint

from aalpy.SULs import OnfsmSUL
from aalpy.learning_algs.non_deterministic.OnfsmObservationTable import NonDetObservationTable
from aalpy.learning_algs.non_deterministic.TraceTree import TraceTree, SULWrapper
from aalpy.oracles import RandomWordEqOracle
from aalpy.utils import load_automaton_from_file

original_model = load_automaton_from_file('models_with_undesired_transitions/fail_safe_model.dot', 'onfsm')
alphabet = original_model.get_input_alphabet()

sul = OnfsmSUL(original_model)
cache = TraceTree()

sul = SULWrapper(sul)
eq_oracle = RandomWordEqOracle(alphabet, sul, num_walks=1000, min_walk_len=4, max_walk_len=10)
eq_oracle.sul = sul

observation_table = NonDetObservationTable(alphabet, sul, 10)

n_samples = 100

for _ in range(n_samples):
    inputs = tuple(choice(alphabet) for _ in range(randint(5, 10)))
    outputs = sul.query(inputs)
    sul.pta.add_trace(inputs, outputs)

hypothesis = observation_table.reconstruct_obs_table()
hypothesis.visualize()