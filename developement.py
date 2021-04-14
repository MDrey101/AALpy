from random import seed

from aalpy.base import SUL
from aalpy.learning_algs import run_abstracted_Lstar_ONFSM
from aalpy.oracles import UnseenOutputRandomWalkEqOracle
from aalpy.utils import load_automaton_from_file, visualize_automaton
from Examples import multi_client_mqtt_example

class TestSUL(SUL):
    def __init__(self):
        super().__init__()
        self.bc = 0
        self.state = 0

    def pre(self):
        self.bc = 0
        self.state = 0
        pass

    def post(self):
        self.bc = 0
        self.state = 0
        pass

    def step(self, letter):
        if letter == 'a':
            if self.state == 0:
                self.state = 1
                return 2
            elif self.state == 1:
                self.state = 0
                return 2
            else: 
                return 1
        else:
            if self.state == 0:
                return 0
            elif self.state == 1:
                self.bc += 1
                self.state = 2
                return 0
            else:
                if self.bc > 2:
                    self.state = 0
                    self.bc = 0
                    return 'O'
                else:
                    self.state = 1
                    return 0

sul = TestSUL()
alph = ['a', 'b']
eq_oracle = UnseenOutputRandomWalkEqOracle(alph, sul, num_steps=5000, reset_prob=0.5, reset_after_cex=True)

abstraction_mapping = dict()
abstraction_mapping[0] = 0
abstraction_mapping['O'] = 0

#learned_onfsm = run_abstracted_Lstar_ONFSM(alph, sul, eq_oracle=eq_oracle, abstraction_mapping = abstraction_mapping, n_sampling=50, print_level=2)

learned_onfsm = multi_client_mqtt_example()

visualize_automaton(learned_onfsm, path="five_clients_mqtt_onfsm", file_type='dot')
