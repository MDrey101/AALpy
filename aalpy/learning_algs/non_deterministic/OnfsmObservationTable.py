from collections import Counter, defaultdict

import constant
from aalpy.automata import Onfsm, OnfsmState, StochasticMealyState, StochasticMealyMachine
from aalpy.learning_algs.non_deterministic.NonDeterministicSULWrapper import NonDeterministicSULWrapper


class NonDetObservationTable:

    def __init__(self, alphabet: list, sul: NonDeterministicSULWrapper, n_sampling, debug=False):
        """
        Construction of the non-deterministic observation table.

        Args:

            alphabet: input alphabet
            sul: system under learning
            n_sampling: number of samples to be performed for each cell
        """
        assert alphabet is not None and sul is not None

        self.alphabet = alphabet
        self.A = [tuple([a]) for a in alphabet]
        self.S = list()  # prefixes of S

        self.E = [tuple([a]) for a in alphabet]

        self.n_samples = n_sampling
        self.closing_counter = 0

        self.sul = sul

        self.debug = debug
        self.sampling_counter = Counter()

        empty_word = tuple()

        # Elements of S are in form that is presented in 'Learning Finite State Models of Observable Nondeterministic
        # Systems in a Testing Context'. Each element of S is a (inputs, outputs) tuple, where first element of the
        # tuple are inputs and second element of the tuple are outputs associated with inputs.
        self.S.append((empty_word, empty_word))
        self.query_counter = 0

        self.pruned_nodes = set()

    def get_row_to_close(self):
        """
        Get row for that need to be closed.

        Returns:

            row that will be moved to S set and closed
        """
        # pruned = self.sul.cache.prune()
        # self.pruned_nodes.update(pruned)
        # self.remove_reduntant_suffixes()

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

    def get_extended_S(self, row_prefix=None):
        """
        Helper generator function that returns extended S, or S.A set.
        For all values in the cell, create a new row where inputs is parent input plus element of alphabet, and
        output is parent output plus value in cell.

        Returns:

            extended S set.
        """

        rows = self.S if row_prefix is None else [row_prefix]

        S_dot_A = []
        for row in rows:
            for a in self.A:
                trace = self.sul.cache.get_all_traces(row, a)

                for t in trace:
                    new_row = (row[0] + a, row[1] + (t[-1],))
                    if new_row not in self.S:
                        S_dot_A.append(new_row)
        return S_dot_A

    def query_missing_observations(self, s=None, e=None):
        s_set = s if s is not None else self.S + self.get_extended_S()
        e_set = e if e is not None else self.E

        for s in s_set:
            for e in e_set:
                if self.debug:
                    print(f'Sampling: Prefix, Suffix:  {s}, {e}')

                while self.sul.cache.get_s_e_sampling_frequency(s, e) < self.n_samples:
                    # break
                    self.query_counter += 1
                    if self.query_counter >= constant.QERY_THRESHOLD_UNTIL_RESET:
                        self.query_counter = 0
                        input("Query Threshold reached - please reset the device!")
                    self.sampling_counter[s] += 1
                    # if self.sampling_counter[s] >= sampling_failsafe:
                    #     break
                    self.sul.query(s[0] + e)


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

        for e in self.E:
            cell = self.sul.cache.get_all_traces(row_prefix, e)
            while cell is None:
                self.query_missing_observations([row_prefix], [e])
                cell = self.sul.cache.get_all_traces(row_prefix, e)

            row_repr += (frozenset(cell),)

        return row_repr

    def clean_obs_table(self):
        """
        Moves duplicates from S to S_dot_A. The entries in S_dot_A which are based on the moved row get deleted.
        The table will be smaller and more efficient.

        """

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

        # self.remove_reduntant_suffixes()

    def gen_hypothesis(self, stochastic=False):
        """
        Generate automaton based on the values found in the observation table.
        If stochastic is set to True, returns a Stochastic Mealy Machine.

        Returns:

            Current hypothesis
        """

        state_distinguish = dict()
        states_dict = dict()
        initial = None

        stateCounter = 0

        state_class = OnfsmState if not stochastic else StochasticMealyState
        model_class = Onfsm if not stochastic else StochasticMealyMachine

        for prefix in self.S:
            state_id = f's{stateCounter}'
            states_dict[prefix] = state_class(state_id)

            states_dict[prefix].prefix = prefix
            state_distinguish[self.row_to_hashable(prefix)] = states_dict[prefix]

            if prefix == self.S[0]:
                initial = states_dict[prefix]
            stateCounter += 1

        for prefix in self.S:
            for a in self.A:
                observations_in_cell = self.sul.cache.get_all_traces(prefix, a)
                probability_distribution = None
                if stochastic:
                    probability_distribution = self.sul.cache.get_sampling_distributions(prefix, a[0])
                for obs in observations_in_cell:
                    reached_row = (prefix[0] + a, prefix[1] + (obs[-1],))
                    destination = state_distinguish[self.row_to_hashable(reached_row)]
                    assert destination
                    if not stochastic:
                        states_dict[prefix].transitions[a[0]].append((obs[-1], destination))
                    else:
                        states_dict[prefix].transitions[a[0]].append((destination, obs[-1],
                                                                      probability_distribution[obs[-1]]))

        assert initial
        automaton = model_class(initial, [s for s in states_dict.values()])
        automaton.characterization_set = self.E

        return automaton

    def remove_reduntant_suffixes(self):
        sorted_e = sorted(self.E, key=len)
        values = dict()

        to_remove = []
        for e in sorted_e:
            column = ()
            for row in list(self.S + self.get_extended_S()):
                cell = self.sul.cache.get_all_traces(row, e)
                column += (frozenset(cell),)
            values[e] = column

        observed = set()
        for e in sorted_e:
            if values[e] not in observed:
                observed.add(values[e])
            else:
                to_remove.append(e)

        for remove_candidate in to_remove:
            sorted_e.remove(remove_candidate)

        self.E = sorted_e

    def sample_cex(self, cex):
        prefix = cex[0][:-1], cex[1][:-1]
        suffix = (cex[0][-1],)

        sampling_achieved = 0
        sampling_attempts = 0
        while self.sul.cache.get_s_e_sampling_frequency(prefix, suffix) < self.n_samples or sampling_achieved < 10:
            sampling_attempts += 1
            output = self.sul.query(cex[0])
            if output[:-1] == prefix[1]:
                sampling_achieved += 1
        if self.debug:
            print('Sampling', cex)
            print('Sampling attempts', sampling_attempts)
            print(self.sul.cache.get_sampling_distributions(prefix, suffix[0]))

