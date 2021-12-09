def random_onfsm_example(num_states, input_size, output_size, n_sampling):
  """
  Generate and learn random ONFSM.
  :param num_states: number of states of the randomly generated automaton
  :param input_size: size of the input alphabet
  :param output_size: size of the output alphabet
  :param n_sampling: number of times each query will be repeated to ensure that all non-determinist outputs are
  observed
  :return: learned ONFSM
  """
  from aalpy.SULs import OnfsmSUL
  from aalpy.utils import generate_random_ONFSM
  from aalpy.oracles import UnseenOutputRandomWalkEqOracle, UnseenOutputRandomWordEqOracle
  from aalpy.learning_algs import run_non_det_Lstar
  from aalpy.utils import visualize_automaton

  onfsm = generate_random_ONFSM(num_states=num_states, num_inputs=input_size, num_outputs=output_size, multiple_out_prob=0.75)
  alphabet = onfsm.get_input_alphabet()

  sul = OnfsmSUL(onfsm)
  eq_oracle = UnseenOutputRandomWordEqOracle(alphabet, sul, num_walks=500, min_walk_len=10, max_walk_len=50)
  eq_oracle = UnseenOutputRandomWalkEqOracle(alphabet, sul, num_steps=5000, reset_prob=0.15, reset_after_cex=True)

  learned_model = run_non_det_Lstar(alphabet, sul, eq_oracle=eq_oracle, n_sampling=n_sampling, print_level=3, trace_tree=True)

  #visualize_automaton(learned_model)

  return learned_model


random_onfsm_example(5, 3, 4, 25)