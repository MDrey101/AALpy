from collections import defaultdict

from aalpy.learning_algs.non_deterministic.OnfsmObservationTable import NonDetObservationTable
from aalpy.base import SUL
from aalpy.utils.HelperFunctions import Dict

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

        self.abstraction_mapping = abstraction_mapping

    def update_obs_table(self, s_set=None, e_set: list = None):
        """
        Perform the membership queries.
        With  the  all-weather  assumption,  each  output  query  is  tried  a  number  of  times  on  the  system,
        and  the  driver  reports  the  set  of  all  possible  outputs.

        Args:

            s_set: Prefixes of S set on which to preform membership queries (Default value = None)
            e_set: Suffixes of E set on which to perform membership queries


        """

        self.observation_table.update_obs_table()

    def abstract_obs_table(self):
        """

        """

        self.S = self.observation_table.S
        self.S_dot_A = self.observation_table.S_dot_A
        self.E = self.observation_table.E

        update_S = self.S + self.S_dot_A
        update_E = self.E

        #for s in update_S:
            #for e in update_E:
                #if e not in self.T[s].keys():
                    #for _ in range(self.n_samples):
                    #    output = tuple(self.sul.query(s[0] + e))
                        # Here I basically say... add just the last element of the output if it e is element of alphabet
                        # else add last len(e) outputs
                    #    o = output[-1] if len(e) == 1 else tuple(output[-len(e):])
                    #    self.add_to_T((s[0], output[:len(s[1])]), e, o)
    
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
