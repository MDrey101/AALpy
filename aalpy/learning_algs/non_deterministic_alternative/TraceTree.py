from collections import defaultdict, Counter

from aalpy.base import SUL


class SULWrapper(SUL):
    """
    Wrapper for non-deterministic SUL. After every step, input/output pair is added to the tree containing all traces.
    """

    def __init__(self, sul: SUL):
        super().__init__()
        self.sul = sul
        self.pta = TraceTree()

    def pre(self):
        self.pta.reset()
        self.sul.pre()

    def post(self):
        self.sul.post()

    def step(self, letter):
        out = self.sul.step(letter)
        self.pta.add_to_tree(letter, out)
        return out


class Node:
    __slots__ = ['output', 'children', 'parent', 'frequency_counter']

    def __init__(self, output):
        self.output = output
        self.children = defaultdict(list)
        self.parent = None

        # frq counter
        self.frequency_counter = 0

    def get_child(self, inp, out):
        """
        Args:
          inp:
          out:

        Returns:

        """
        return next((child for child in self.children[inp] if child.output == out), None)

    def get_prefix(self):
        prefix = ()
        curr_node = self
        while curr_node.parent is not None:
            prefix = (curr_node.output,) + prefix
            curr_node = curr_node.parent
        return prefix


class TraceTree:
    """
    Tree used for keeping track of seen observations.
    """

    def __init__(self):
        self.root_node = Node(None)
        self.curr_node = None

    def reset(self):
        self.curr_node = self.root_node

    def add_to_tree(self, inp, out):
        """
        Adds new element to tree and makes it the current node

        Args:

          inp: Input
          out: Output

        """
        if inp not in self.curr_node.children.keys() or \
                out not in {child.output for child in self.curr_node.children[inp]}:
            node = Node(out)
            self.curr_node.children[inp].append(node)
            node.parent = self.curr_node

        self.curr_node = self.curr_node.get_child(inp, out)
        self.curr_node.frequency_counter += 1

    def add_trace(self, inputs, outputs):
        self.reset()
        for i, o in zip(inputs, outputs):
            self.add_to_tree(i, o)

    def get_to_node(self, inputs, outputs):
        """
        Follows the path described by inp and out and returns the node which is reached

        Args:
          inputs: Inputs
          outputs: Outputs

        Returns:

          Node that is reached when following the given input and output through the tree
        """
        curr_node = self.root_node
        for i, o in zip(inputs, outputs):
            node = curr_node.get_child(i, o)
            if node is None:
                return None
            curr_node = node

        return curr_node

    def sample(self, sampling_num):
        pass

    def get_all_traces(self, curr_node=None, e=None):
        """

        Args:

          curr_node: current node
          e: List of inputs

        Returns:

          Traces of outputs corresponding to the input-sequence given by e
        """

        if not curr_node or not e:
            return []

        queue = [(curr_node, 0)]
        reached_nodes = []
        while queue:
            node, depth = queue.pop(0)
            if depth == len(e):
                reached_nodes.append(node)
            else:
                children_with_same_input = node.children[e[depth]]
                for c in children_with_same_input:
                    queue.append((c, depth + 1))

        cell = [node.get_prefix()[-len(e):] for node in reached_nodes]
        return cell

    def get_table(self, s, e):
        """
        Generates a table from the tree

        Args:
          s: rows from S, S_dot_A, or both which should be presented in the table.
          e: E

        Returns:
          a table in a format that can be used for printing.
        """
        result = {}
        for pair in s:
            curr_node = self.get_to_node(pair[0], pair[1])
            result[pair] = {}

            for inp in e:
                result[pair][inp] = self.get_all_traces(curr_node, inp)

        return result

    def prune(self, threshold=0.1):
        counter = 0
        pruned_nodes = set()

        queue = [(self.root_node, tuple())]
        while queue:
            curr_node, path = queue.pop(0)
            to_delete = []
            for inp in curr_node.children.keys():
                children = curr_node.children[inp]
                total_samples = sum(child.frequency_counter for child in children)
                for child in children:
                    #if child.frequency_counter / total_samples <= threshold:
                    if child.output == 'DANGER':
                        to_delete.append((inp, child.output, path + (inp, child.output)))
                    else:
                        queue.append((child, path + (inp, child.output)))

            for i, o, path_to_delete_node in to_delete:
                delete_candidate = curr_node.get_child(i, o)
                if delete_candidate is not None:
                    # detach from the tree/cache
                    curr_node.children[i].remove(delete_candidate)
                    # get inputs and outputs from path to node
                    inputs, outputs = path_to_delete_node[0::2], path_to_delete_node[1::2]
                    # add to set of nodes that were pruned in this iteration
                    pruned_nodes.add((inputs, outputs))
                    counter += 1

        return pruned_nodes

    def find_cex_in_cache(self, hypothesis):

        queue = [(self.root_node, tuple())]
        while queue:
            curr_node, path = queue.pop(0)

            if path:
                hypothesis.reset_to_initial()
                inputs, outputs = [], []
                for i, o in zip(path[0::2], path[1::2]):
                    inputs.append(i)
                    outputs.append(o)
                    out = hypothesis.step_to(i, o)
                    if out is None:
                        return inputs, outputs
            for inp in curr_node.children.keys():
                children = curr_node.children[inp]
                for child in children:
                    # if curr_node.frequency_counter[(inp, child_out)] >= threshold:
                    queue.append((child, path + (inp, child.output)))

        return None

    def get_s_e_sampling_frequency(self, prefix, suffix):
        sampling_frequency = 0
        curr_node = self.root_node
        for i, o in zip(prefix[0], prefix[1]):
            curr_node = curr_node.get_child(i, o)

        queue = [(curr_node, 0)]
        while queue:
            node, depth = queue.pop(0)
            children_with_same_input = node.children[suffix[depth]]
            if depth == len(suffix) - 1:
                for c in children_with_same_input:
                    sampling_frequency += c.frequency_counter
            else:
                for c in children_with_same_input:
                    queue.append((c, depth + 1))

        return sampling_frequency
