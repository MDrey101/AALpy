import random

from aalpy.base.Oracle import Oracle
from aalpy.base.SUL import SUL


class StatePrefixEqOracle(Oracle):
    """
    Equivalence oracle that achieves guided exploration by starting random walks from each state a walk_per_state
    times. Starting the random walk ensures that all states are reached at least walk_per_state times and that their
    surrounding is randomly explored. Note that each state serves as a root of random exploration of maximum length
    rand_walk_len exactly walk_per_state times during learning. Therefore excessive testing of initial states is
    avoided.
    """
    def __init__(self, alphabet: list, sul: SUL, walks_per_state=10, walk_len=30, depth_first=False):
        """
        Args:

            alphabet: input alphabet

            sul: system under learning

            walks_per_state:individual walks per state of the automaton over the whole learning process

            walk_len:length of random walk

            depth_first:first explore newest states
        """

        super().__init__(alphabet, sul)
        self.walks_per_state = walks_per_state
        self.steps_per_walk = walk_len
        self.depth_first = depth_first

        self.freq_dict = dict()

    def find_cex(self, hypothesis):

        states_to_cover = []

        print(f"num states: {len(hypothesis.states)}: {hypothesis.states}")
        if len(states_to_cover) > 6: #for CC2650.dot
            print("ERROR: found more states than should be expected!")
        for state in hypothesis.states:
            if state.prefix not in self.freq_dict.keys():
                self.freq_dict[state.prefix] = 0

            states_to_cover.extend([state] * (self.walks_per_state - self.freq_dict[state.prefix]))

        if self.depth_first:
            # reverse sort the states by length of their access sequences
            # first do the random walk on the state with longest access sequence
            states_to_cover.sort(key=lambda x: len(x.prefix), reverse=True)
        else:
            random.shuffle(states_to_cover)

        print(f"num states to cover: {len(states_to_cover)}: {states_to_cover}")
        for state in states_to_cover:
            self.freq_dict[state.prefix] = self.freq_dict[state.prefix] + 1
            self.reset_hyp_and_sul(hypothesis)

            prefix = state.prefix
            out_sul = "ERROR"
            error_counter = 0
            while out_sul == "ERROR" and error_counter < 20:
                self.reset_hyp_and_sul(hypothesis)

                for p in prefix:
                    out_sul = self.sul.step(p)
                    if out_sul == "ERROR":
                        error_counter += 1
                        break

                    output_list = hypothesis.outputs_on_input(p)
                    print(output_list)

                    self.num_steps += 1
                    if out_sul not in output_list:
                        break

                if out_sul == "ERROR":
                    error_counter += 1
                    continue

                suffix = ()
                for _ in range(self.steps_per_walk):
                    suffix += (random.choice(self.alphabet),)

                    out_sul = self.sul.step(suffix[-1])
                    print(out_sul)
                    if "ERROR" == out_sul:
                        break

                    output_list = hypothesis.outputs_on_input(suffix[-1])
                    print(output_list)

                    self.num_steps += 1
                    if out_sul in output_list:
                        hypothesis.step_to(suffix[-1], out_sul)
                        # return prefix + suffix
                    else:
                        self.sul.post()
                        return prefix + suffix

                if out_sul == "ERROR":
                    error_counter += 1
                    continue

        return None