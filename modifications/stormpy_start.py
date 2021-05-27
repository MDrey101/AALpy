import stormpy.examples
import stormpy.examples.files

path = stormpy.examples.files.prism_dtmc_die
prism_program = stormpy.parse_prism_program(path)

model = stormpy.build_model(prism_program)
print(f"Number of states: {model.nr_states}")
print(f"Number of transitions: {model.nr_transitions}")
print(f"Labels: {model.labeling.get_labels()}\n")

formula_str = "P=? [F s=2]"
properties = stormpy.parse_properties(formula_str, prism_program)
model = stormpy.build_model(prism_program, properties)
print(f"Labels in the model: {sorted(model.labeling.get_labels())}")
print(f"Number of states: {model.nr_states}\n")

formula_str = "P=? [F s=7 & d=2]"
properties = stormpy.parse_properties(formula_str, prism_program)
model = stormpy.build_model(prism_program, properties)
print(f"Labels in the model: {sorted(model.labeling.get_labels())}")
print(f"Number of states: {model.nr_states}\n")

properties = stormpy.parse_properties(formula_str, prism_program)
model = stormpy.build_model(prism_program, properties)
result = stormpy.model_checking(model, properties[0])

assert result.result_for_all_states
for value in result.get_values():
    print(value)

initial_state = model.initial_states[0]
print(result.at(initial_state))
print("\n")

path = stormpy.examples.files.prism_dtmc_die
prism_program = stormpy.parse_prism_program(path)
model = stormpy.build_model(prism_program)
print(model.model_type)
print("\n")

for state in model.states:
    for action in state.actions:
        for transition in action.transitions:
            print(f"From state {state}, with probability {transition.value()}, go to state {transition.column}")