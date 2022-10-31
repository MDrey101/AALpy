from random import choice, choices, randint

from aalpy.automata import Onfsm
from aalpy.base import SUL
from aalpy.learning_algs import run_non_det_Lstar, run_Alergia
from FailSafeOracle import FailSafeOracle
from aalpy.utils import load_automaton_from_file


class FailSUL(SUL):
    def __init__(self, mdp: Onfsm):
        super().__init__()
        self.onfsm = mdp

    def pre(self):
        self.onfsm.reset_to_initial()

    def post(self):
        pass

    def step(self, letter):
        if "DANGER" in self.onfsm.outputs_on_input(letter):
            possible_states = self.onfsm.current_state.transitions[letter]
            danger_position = ["DANGER" == state[0] for state in possible_states]
            probability_distributions = [0.05 if d else 0.9 for d in danger_position]

            transition = choices(possible_states, probability_distributions, k=1)[0]
        else:
            transition = choice(self.onfsm.current_state.transitions[letter])
        output = transition[0]
        self.onfsm.current_state = transition[1]
        return output


def test_alergia():
    model = load_automaton_from_file("models_with_undesired_transitions/model_1.dot", "onfsm")
    alphabet = model.get_input_alphabet()
    sul = FailSUL(model)

    data_set = []
    for _ in range(10000):
        sul.pre()
        test_seq = ['Init']
        for _ in range(randint(4, 10)):
            i = choice(alphabet)
            o = sul.step(i)
            test_seq.append((i, o))

        data_set.append(test_seq)

    learned_model = run_Alergia(data_set, 'smm')
    learned_model.visualize()


if __name__ == '__main__':
    model = load_automaton_from_file("models_with_undesired_transitions/model_0.dot", "onfsm")
    model1 = load_automaton_from_file("models_with_undesired_transitions/model_1.dot", "onfsm")
    model2 = load_automaton_from_file("models_with_undesired_transitions/model_2.dot", "onfsm")
    model3 = load_automaton_from_file("models_with_undesired_transitions/model_3.dot", "onfsm")
    model4 = load_automaton_from_file("models_with_undesired_transitions/model_3.dot", "onfsm")
    model5 = load_automaton_from_file("models_with_undesired_transitions/model_5.dot", "onfsm")

    # test_alergia()
    # exit()
    model = load_automaton_from_file("models_with_undesired_transitions/model_1.dot", "onfsm")
    alphabet = model.get_input_alphabet()

    # 2, 3, 4
    from random import seed
    # seed(3)
    # 13

    sul = FailSUL(model)

    eq_oracle = FailSafeOracle(alphabet, sul, num_walks=100, min_walk_len=4, max_walk_len=10, reset_after_cex=False)
    # eq_oracle = RandomWordEqOracle(alphabet, sul, num_walks=1000, min_walk_len=4, max_walk_len=10)

    learned_model = run_non_det_Lstar(alphabet, sul, eq_oracle, n_sampling=10,)

    learned_model.visualize()

    # if learned_model.size > 4:
    #     sul2 = OnfsmSUL(learned_model)
    #     eq_oracle = RandomWordEqOracle(alphabet, sul2, num_walks=1000, min_walk_len=4, max_walk_len=10)
    #     minimized_automaton = run_non_det_Lstar(alphabet, sul2, eq_oracle, n_sampling=10,)
    #
    #     minimized_automaton.visualize()
