from aalpy.SULs import OnfsmSUL
from aalpy.oracles import UnseenOutputRandomWalkEqOracle, UnseenOutputRandomWordEqOracle
from aalpy.learning_algs import run_non_det_Lstar
from aalpy.utils import visualize_automaton
from aalpy.automata import Dfa, DfaState, OnfsmState, Onfsm, MarkovChain

states = []
for i in range(4):
  state = OnfsmState(f's{i}')
  states.append(state)

states[0].transitions['a'].append(('AB', states[1]))
states[0].transitions['b'].append(('B', states[2]))

states[1].transitions['a'].append(('A', states[3]))
states[1].transitions['b'].append(('B', states[3]))
states[1].transitions['a'].append(('AB', states[1]))

states[2].transitions['a'].append(('A', states[1]))
states[2].transitions['b'].append(('B', states[2]))


states[3].transitions['a'].append(('A', states[3]))
states[3].transitions['b'].append(('AB', states[3]))


onfsm = Onfsm(states[0], states)
alphabet = onfsm.get_input_alphabet()

sul = OnfsmSUL(onfsm)
eq_oracle = UnseenOutputRandomWordEqOracle(alphabet, sul, num_walks=100, min_walk_len=10, max_walk_len=30)
#eq_oracle = UnseenOutputRandomWalkEqOracle(alphabet, sul, num_steps=5000, reset_prob=0.15, reset_after_cex=True)

learned_model = run_non_det_Lstar(alphabet, sul, eq_oracle=eq_oracle, n_sampling=5, print_level=3, trace_tree=True)

#visualize_automaton(learned_model)