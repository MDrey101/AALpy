from aalpy.base import SUL
from aalpy.learning_algs.non_deterministic.TraceTree import TraceTree


class NonDeterministicSULWrapper(SUL):
    """
    Wrapper for non-deterministic SUL. After every step, input/output pair is added to the tree containing all traces.
    """

    def __init__(self, sul: SUL, pruning_threshold=0.2):
        super().__init__()
        self.sul = sul
        self.cache = TraceTree(threshold=pruning_threshold)
        self.test = []

    def pre(self):
        self.test = []
        self.cache.reset()
        self.sul.pre()

    def post(self):
        # print(self.test)
        self.sul.post()

    def step(self, letter):
        self.test.append(letter)
        out = self.sul.step(letter)
        self.cache.add_to_tree(letter, out)
        return out
