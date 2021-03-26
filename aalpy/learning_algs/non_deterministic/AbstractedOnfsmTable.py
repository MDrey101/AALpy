from abc import abstractclassmethod
from collections import defaultdict

from aalpy.learning_algs.non_deterministic.OnfsmObservationTable import NonDetObservationTable
from aalpy.base import Automaton, SUL
from aalpy.utils.HelperFunctions import Dict, extend_set
from aalpy.automata import Onfsm, OnfsmState


class AbstractedNonDetObservationTable:
    def __init__(self,alphabet: list, sul: SUL, abstraction_mapping: Dict, n_sampling=100):
        """
        Construction of the non-deterministic observation table.

        Args:

            alphabet: input alphabet
            sul: system under learning
            n_sampling: number of samples to be performed for each cell
            abstraction_mapping: map for translation of outputs
        """

        assert alphabet is not None and sul is not None

        self.observation_table = NonDetObservationTable(alphabet, sul, n_sampling)

        self.S = list()
        self.S_dot_A = []
        self.E = []
        self.T = defaultdict(dict)
        self.A = [tuple([a]) for a in alphabet]

        self.abstraction_mapping = abstraction_mapping

        empty_word = tuple()
        self.S.append((empty_word, empty_word))

    def update_obs_table(self, s_set=None, e_set: list = None):
        """
        Perform the membership queries.
        With  the  all-weather  assumption,  each  output  query  is  tried  a  number  of  times  on  the  system,
        and  the  driver  reports  the  set  of  all  possible  outputs.

        Args:

            s_set: Prefixes of S set on which to preform membership queries (Default value = None)
            e_set: Suffixes of E set on which to perform membership queries


        """

        self.observation_table.update_obs_table(s_set,e_set)
        self.abstract_obs_table()

    def abstract_obs_table(self):
        """

        """

        self.S = self.observation_table.S
        self.S_dot_A = self.observation_table.S_dot_A
        self.E = self.observation_table.E

        update_S = self.S + self.S_dot_A
        update_E = self.E

        for s in update_S:
            for e in update_E:
                observed_outputs = self.observation_table.T[s][e]
                for o_tup in observed_outputs:
                    if(len(e) == 1):
                        o_tup = tuple([o_tup])
                    for o in o_tup:
                        abstract_output = self.abstraction_mapping[o]
                        self.add_to_T(s,e,abstract_output)
    
    def add_to_T(self, s, e, value):
        """
        Add values to the cell at T[s][e].

        Args:

            s: prefix
            e: element of S
            value: value to be added to the cell


        """
        if e not in self.T[s]:
            self.T[s][e] = set()
        self.T[s][e].add(value)

    def update_extended_S(self, row):
        """
        Helper generator function that returns extended S, or S.A set.
        For all values in the cell, create a new row where inputs is parent input plus element of alphabet, and
        output is parent output plus value in cell.

        Returns:

            New rows of extended S set.
        """
        return self.observation_table.update_extended_S(row)
    
    def get_row_to_close(self):
        """
        Get row for that need to be closed.

        Returns:

            row that will be moved to S set and closed
        """
        s_rows = set()
        for s in self.S:
            s_rows.add(self.row_to_hashable(s))

        for t in self.S_dot_A:
            row_t = self.row_to_hashable(t)

            if row_t not in s_rows:
                self.S.append(t)
                self.S_dot_A.remove(t)
                return t

        return None
    
    def get_row_to_complete(self):
        """
        Get row for that need to be completed.

        Returns:
            row that needs to be added to the extended S set
        """

        s_rows = set()
        for s in self.S:
            s_rows.add(tuple((s,self.row_to_hashable(s))))

        for s_row in s_rows:
            similar_s_dot_a_rows = []
            for t in self.S_dot_A:
                row_t = self.row_to_hashable(t)
                if row_t == s_row[1]:
                    similar_s_dot_a_rows.append(t)
            similar_s_dot_a_rows.sort(key=lambda s: len(s[0]))
            for a in self.A: # TODO: check if there is a mistake in the paper
                complete_outputs = self.observation_table.T[s_row[0]][a]
                for similar_s_dot_a_row in similar_s_dot_a_rows:
                    t_row_outputs = self.observation_table.T[similar_s_dot_a_row][a]
                    output_difference = t_row_outputs.difference(complete_outputs)
                    if len(output_difference) > 0:
                        extension = None
                        for o in output_difference:
                            extension = (similar_s_dot_a_row[0] + a, similar_s_dot_a_row[1] + tuple([o]))
                            if extension not in self.S and extension not in self.S_dot_A:
                                return extension
                            else: 
                                complete_outputs = complete_outputs.union(output_difference)

        return None
    
    def get_row_to_make_consistent(self):
        """

        """
        unified_S = self.S + self.S_dot_A
        s_rows = set()
        for s in self.S:
            s_rows.add(tuple((s,self.row_to_hashable(s))))

        for s_row in s_rows:
            similar_s_dot_a_rows = []
            for t in self.S_dot_A:
                row_t = self.row_to_hashable(t)
                if row_t == s_row[1]:
                    similar_s_dot_a_rows.append(t)
            similar_s_dot_a_rows.sort(key=lambda s: len(s[0]))
            for a in self.A: # TODO: check if there is a mistake in the paper
                outputs = self.observation_table.T[s_row[0]][a]
                for o in outputs:
                    extended_s_sequence = (s_row[0][0] + a, s_row[0][1] + tuple([o]))
                    if extended_s_sequence in unified_S:
                        extended_s_sequence_row = self.row_to_hashable(extended_s_sequence)
                        for similar_s_dot_a_row in similar_s_dot_a_rows:
                            extended_s_dot_a_sequence = (similar_s_dot_a_row[0] + a, similar_s_dot_a_row[1] + tuple([o]))
                            if extended_s_dot_a_sequence in unified_S:
                                extended_s_dot_a_sequence_row = self.row_to_hashable(extended_s_dot_a_sequence)
                                if extended_s_sequence_row is not extended_s_dot_a_sequence_row:
                                    return self.get_distinctive_input_sequence(extended_s_sequence, extended_s_dot_a_sequence, a)

        return None


    
    def get_distinctive_input_sequence(self, first_row, second_row, input):
        """

        """
        for e in self.E:
            if self.T[first_row][e] is not self.T[second_row][e]:
                return tuple([input]) + e

        return None



    def complete_extended_S(self, row_prefix):
        """
        """
        extension = [row_prefix]
        self.observation_table.S_dot_A.extend(extension)
        return extension
    
    def update_E(self, seq):
        """
        """
        if seq not in self.E:
            self.E.append(seq)
    
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
            #if e in self.T[row_prefix].keys():
            row_repr += (frozenset(self.T[row_prefix][e]),)
        return row_repr
    
    def gen_hypothesis(self) -> Automaton:
        """
        Generate automaton based on the values found in the observation table.

        Returns:

            Current hypothesis

        """
        state_distinguish = dict()
        states_dict = dict()
        initial = None

        unified_S = self.S + self.S_dot_A

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
            similar_rows = []
            for row in unified_S:
                if self.row_to_hashable(row) == self.row_to_hashable(prefix):
                    similar_rows.append(row)
            for row in similar_rows:
                for a in self.A:
                    for t in self.observation_table.T[row][a]:
                        if (row[0] + a, row[1] + tuple([t])) in unified_S:
                            state_in_S = state_distinguish[self.row_to_hashable((row[0] + a, row[1] + tuple([t])))]
                            states_dict[prefix].transitions[a[0]].append((t, state_in_S))

        assert initial
        automaton = Onfsm(initial, [s for s in states_dict.values()])
        automaton.characterization_set = self.E

        return automaton
    
    def extend_S_dot_A(self, cex_prefixes: list):
        prefixes = self.S + self.S_dot_A
        prefixes_to_extend = []
        for cex_prefix in cex_prefixes:
            if cex_prefix not in prefixes:
                prefixes_to_extend.append(cex_prefix)
        return prefixes_to_extend
    
    def cex_processing(self, cex: tuple, hypothesis: Onfsm):
        """
        
        """

        cex_len = len(cex[0])
        hypothesis.reset_to_initial()

        for step in range(0,cex_len-1):
            hypothesis.step_to(cex[0][step],cex[1][step])

        possible_outputs = hypothesis.outputs_on_input(cex[0][cex_len-1])

        equivalent_output = False
        
        for out in possible_outputs:
            if (self.abstraction_mapping[cex[1][cex_len-1]] == self.abstraction_mapping[out]):
                equivalent_output = True
                break

        if equivalent_output:
            # add prefixes of cex to S_dot_A
            cex_prefixes = [(tuple(cex[0][0:i+1]),tuple(cex[1][0:i+1])) for i in range(0,len(cex[0]))]
            prefixes_to_extend = self.extend_S_dot_A(cex_prefixes)
            self.observation_table.S_dot_A.extend(prefixes_to_extend)
            self.update_obs_table(s_set=prefixes_to_extend)
        else: 
            # add distinguishing suffixes of cex to E
            cex_suffixes = self.observation_table.cex_processing(cex)
            added_suffixes = extend_set(self.observation_table.E, cex_suffixes)
            self.update_obs_table(e_set=added_suffixes)

