import time
from random import choice, randint

from aalpy.SULs import OnfsmSUL
from aalpy.learning_algs.non_deterministic_passive.OnfsmObservationTable import NonDetObservationTable
from aalpy.learning_algs.non_deterministic_passive.TraceTree import TraceTree, SULWrapper
from aalpy.utils import load_automaton_from_file

original_model = load_automaton_from_file('../../../fail_safe_model.dot', 'onfsm')
alphabet = original_model.get_input_alphabet()

sul = OnfsmSUL(original_model)
cache = TraceTree()

sul = SULWrapper(sul)

observation_table = NonDetObservationTable(alphabet, sul, 10)

n_samples = 1000

for _ in range(n_samples):
    inputs = tuple(choice(alphabet) for _ in range(randint(5, 10)))
    outputs = sul.query(inputs)
    sul.pta.add_trace(inputs, outputs)


start_time = time.time()
hypothesis = observation_table.reconstruct_obs_table()
learning_time = round(time.time() - start_time, 2)

info = {
        'automaton_size': len(hypothesis.states),
        'queries_learning': sul.num_queries,
        'steps_learning': sul.num_steps,
        'learning_time': learning_time,
        }

for key, value in info.items():
    print(f'{key}: {value}')

hypothesis.visualize()