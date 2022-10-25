import math
from collections import Counter

from aalpy.automata import Onfsm, OnfsmState
from aalpy.base import Automaton
from colorama import Fore
import constant
# from aalpy.learning_algs.non_deterministic.TraceTree import SULWrapper


class NonDetObservationTable:

    def __init__(self, alphabet: list, sul, n_sampling):
        """
        Construction of the non-deterministic observation table.

        Args:

            alphabet: input alphabet
            sul: system under learning
            n_sampling: number of samples to be performed for each cell
        """
        assert alphabet is not None and sul is not None

        self.A = [tuple([a]) for a in alphabet]
        self.S = list()  # prefixes of S

        self.E = [tuple([a]) for a in alphabet]

        self.n_samples = n_sampling
        self.closing_counter = 0

        self.sul = sul

        self.sampling_counter = Counter()

        empty_word = tuple()

        # Elements of S are in form that is presented in 'Learning Finite State Models of Observable Nondeterministic
        # Systems in a Testing Context'. Each element of S is a (inputs, outputs) tuple, where first element of the
        # tuple are inputs and second element of the tuple are outputs associated with inputs.
        self.S.append((empty_word, empty_word))

    def get_row_to_close(self):
        """
        Get row for that need to be closed.

        Returns:

            row that will be moved to S set and closed
        """
        s_rows = set()
        update_S_dot_A = self.get_extended_S()

        for s in self.S.copy():
            s_rows.add(self.row_to_hashable(s))

        for t in update_S_dot_A:
            row_t = self.row_to_hashable(t)
            if row_t not in s_rows:
                self.closing_counter += 1
                self.S.append(t)
                return t

        self.closing_counter = 0
        return None

    def get_extended_S(self, row_prefix = None):
        """
        Helper generator function that returns extended S, or S.A set.
        For all values in the cell, create a new row where inputs is parent input plus element of alphabet, and
        output is parent output plus value in cell.

        Returns:

            extended S set.
        """

        rows = self.S if row_prefix == None else [row_prefix]

        S_dot_A = []
        for row in rows:
            curr_node = self.sul.pta.get_to_node(row[0], row[1])
            for a in self.A:
                trace = self.sul.pta.get_all_traces(curr_node, a)

                for t in trace:
                    new_row = (row[0] + a, row[1] + (t[-1],))
                    if new_row not in self.S:
                        S_dot_A.append(new_row)
        return S_dot_A

    def update_obs_table(self, s_set=None, e_set: list = None):
        """
        Perform the membership queries.
        With  the  all-weather  assumption,  each  output  query  is  tried  a  number  of  times  on  the  system,
        and  the  driver  reports  the  set  of  all  possible  outputs.

        Args:

            s_set: Prefixes of S set on which to preform membership queries (Default value = None)
            e_set: Suffixes of E set on which to perform membership queries
        """

        update_S = s_set if s_set else self.S + self.get_extended_S()
        update_E = e_set if e_set else self.E
        idc_dict = {}

        # update_S, update_E = self.S + self.S_dot_A, self.E


        # TODO: Problem - output is not observed and IDC is caused - the behavior should be checked again!
        def inner_update_obs_table(inner_update_S, inner_update_E, idc_dict):
            for s in inner_update_S:
                for e in inner_update_E:
                    num_s_e_sampled = 0
                    sample_IDC_counter = 0
                    while num_s_e_sampled < self.n_samples:
                        print(Fore.RED + "expected_output:" + str(s[1]) + Fore.WHITE)
                        expected_output = s[1] if type(s[1]) == tuple else tuple(s[1],)

                        # constant.NONDET_QUERY_NUMBER = 5 * 5 if len(s[0] + e) < 4 else (4 ** len(s[0] + e)) * 4
                        # constant.NONDET_QUERY_NUMBER = 5 * 5 if len(s[0] + e) < 4 else math.factorial(len(s[0] + e)) * 3
                        # constant.NONDET_THRESHOLD = 3 if len(expected_output) < 4 else len(expected_output)
                        # constant.NONDET_THRESHOLD = 2

                        expected_prefix = (s[0] + e, expected_output)
                        # idc_found, repeat_flag, observed_output = self.sul.query(expected_prefix)
                        idc_found, repeat_list, observed_output = self.sul.query(expected_prefix)

                        if idc_found:
                            check_S = []
                            for entry in inner_update_S:
                                if entry == ((), ()):
                                    break
                                check_S.append(entry)

                            if len(observed_output) == 0:
                                trace_to_resample = ((), ())
                            else:
                                trace_to_resample = (expected_prefix[0][:len(observed_output[0])], observed_output[0])

                            if trace_to_resample in idc_dict:
                                idc_dict[trace_to_resample] += 1
                            else:
                                idc_dict[trace_to_resample] = 1

                            sample_IDC_counter += 1
                            if sample_IDC_counter >= self.n_samples:
                                # index = None
                                trace_to_return = [] if trace_to_resample == ((), ()) or trace_to_resample in check_S else [trace_to_resample]
                                # if trace_to_resample in check_S:
                                #     index = check_S.index(trace_to_resample)
                                return trace_to_return#, index
                            else:
                                continue

                        # elif repeat_flag:
                            # trace_to_resample = (expected_prefix[0][:len(observed_output[0])], observed_output[0])
                            # check_S = []
                            # for entry in update_S:
                            #     if entry == ((), ()):
                            #         break
                            #     check_S.append(entry)
                            # if trace_to_resample not in check_S:
                            #     return trace_to_resample
                        elif len(repeat_list) != 0:
                            check_S = []
                            for entry in inner_update_S:
                                if entry == ((), ()):
                                    break
                                check_S.append(entry)

                            traces_to_resample = []
                            for out_rep in [rep for rep in repeat_list if rep != ((), ())]:
                                trace_to_resample = (expected_prefix[0][:len(out_rep)], out_rep)
                                index = None
                                if trace_to_resample not in check_S:
                                    traces_to_resample.append(trace_to_resample)
                            return traces_to_resample
                        
                        for out in observed_output:
                            if out[:len(s[1])] == s[1]:
                                print(Fore.GREEN + "matched output")
                                num_s_e_sampled += 1
                                self.sampling_counter[s[0] + e] += 1
                                break

        traces_to_resample = inner_update_obs_table(update_S, update_E, idc_dict)
        # trace_to_resample = inner_update_obs_table(update_S, update_E, idc_dict)
        # while trace_to_resample is not None:
        #     if trace_to_resample != update_S[0]:
        #         update_S.insert(0, trace_to_resample)
        #     # if idc_dict[trace_to_resample] >= 2:
        #     #     input("Please physically reset the device!")
        #     trace_to_resample = inner_update_obs_table(update_S, update_E, idc_dict)
        while traces_to_resample is not None:
            for trace in traces_to_resample:
                update_S.insert(0, trace)
            # if idc_dict[trace_to_resample] >= 2:
            #     input("Please physically reset the device!")
            traces_to_resample = inner_update_obs_table(update_S, update_E, idc_dict)


                    # output_list = self.sul.query(s[0] + e)
                    # # output = tuple(self.sul.query(s[0] + e))
                    # # Here I basically say...
                    # # add just the last element of the output if it e is element of alphabet
                    # # else add last len(e) outputs
                    # for output_entry in output_list:
                    #     output = tuple(output_entry)
                    #     o = output[-1] if len(e) == 1 else tuple(output[-len(e):])
                    #     self.add_to_T((s[0], output[:len(s[1])]), e, o)
                    #
                    #     if output[:len(s[1])] == s[1]:
                    #         num_s_e_sampled += 1

    def clean_obs_table(self):
        """
        Moves duplicates from S to S_dot_A. The entries in S_dot_A which are based on the moved row get deleted.
        The table will be smaller and more efficient.

        """
        # just for testing without cleaning
        # return False

        tmp_S = self.S.copy()
        tmp_both_S = self.S + self.get_extended_S()
        hashed_rows_from_s = set()

        tmp_S.sort(key=lambda t: len(t[0]))

        for s in tmp_S:
            hashed_s_row = self.row_to_hashable(s)
            if hashed_s_row in hashed_rows_from_s:
                if s in self.S:
                    self.S.remove(s)
                size = len(s[0])
                for row_prefix in tmp_both_S:
                    s_both_row = (row_prefix[0][:size], row_prefix[1][:size])
                    if s != row_prefix and s == s_both_row:
                        if row_prefix in self.S:
                            self.S.remove(row_prefix)
            else:
                hashed_rows_from_s.add(hashed_s_row)

    def gen_hypothesis(self) -> Automaton:
        """
        Generate automaton based on the values found in the observation table.

        Returns:

            Current hypothesis

        """
        state_distinguish = dict()
        states_dict = dict()
        initial = None

        stateCounter = 0
        for prefix in self.S:
            state_id = f's{stateCounter}'
            states_dict[prefix] = OnfsmState(state_id)

            states_dict[prefix].prefix = prefix
            state_distinguish[self.row_to_hashable(prefix)] = states_dict[prefix]

            if prefix == self.S[0]:
                initial = states_dict[prefix]
            stateCounter += 1

        for prefix in self.S:
            curr_node = self.sul.pta.get_to_node(prefix[0], prefix[1])
            for a in self.A:
                trace = self.sul.pta.get_all_traces(curr_node, a)
                for t in trace:
                    reached_row = (prefix[0] + a, prefix[1] + (t[-1],))
                    if self.row_to_hashable(reached_row) not in state_distinguish.keys():
                        print('reeee')
                    state_in_S = state_distinguish[self.row_to_hashable(reached_row)]
                    assert state_in_S  # shouldn't be necessary because of the if condition
                    states_dict[prefix].transitions[a[0]].append((t[-1], state_in_S))

        assert initial
        automaton = Onfsm(initial, [s for s in states_dict.values()])
        automaton.characterization_set = self.E

        return automaton

    def row_to_hashable(self, row_prefix):
        """
        Creates the hashable representation of the row. Frozenset is used as the order of element in each cell does not
        matter

        Args:

            row_prefix: prefix of the row in the observation table

        Returns:

            hashable representation of the row

        """
        row_repr = tuple()
        curr_node = self.sul.pta.get_to_node(row_prefix[0], row_prefix[1])

        for e in self.E:
            cell = self.sul.pta.get_all_traces(curr_node, e)
            while not cell:
                self.update_obs_table(s_set=[row_prefix], e_set=[e])
                cell = self.sul.pta.get_all_traces(curr_node, e)

            row_repr += (frozenset(cell),)

        return row_repr

