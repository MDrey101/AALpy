from collections import defaultdict

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
        """ """
        self.pta.reset()
        self.sul.pre()

    def post(self):
        """ """
        self.sul.post()

    def step(self, letter):
        """

        Args:
          letter:

        Returns:

        """
        out = self.sul.step(letter)
        self.pta.add_to_tree(letter, out)
        return out


class Node:
    """ """
    def __init__(self, output):
        self.output = output
        self.children = defaultdict(list)

    def get_child(self, inp, out):
        """

        Args:
          inp:
          out:

        Returns:

        """
        return next((child for child in self.children[inp] if child.output == out), None)


class TraceTree:
    """ """
    def __init__(self):
        self.root_node = Node(None)
        self.curr_node = None

    def reset(self):
        """ """
        self.curr_node = self.root_node

    def add_to_tree(self, inp, out):
        """
        Args:
          inp:
          out:

        Returns:
        """
        if inp not in self.curr_node.children.keys() or out not in [child.output for child in self.curr_node.children[inp]]:
            node = Node(out)
            self.curr_node.children[inp].append(node)
        else:
            self.curr_node = self.curr_node.get_child(inp, out)



    def get_to_node(self, inp, out):
        """
        Args:
          inp:
          out:

        Returns: Node that is reached when following the given input and output through the tree
        """
        curr_node = self.root_node
        for i, o in zip(inp, out):
            node = curr_node.get_child(i, o)
            if node is None:
                return None
            curr_node = node

        return curr_node


    def get_single_trace(self, curr_node=None, e=None):
        """
        Args:
          curr_node:
          e:

        Returns: Trace of outputs from one input of e
        """

        if curr_node is None:
            return []

        if not e:
            return []

        e = list(e)
        res = [[]]

        input = e.pop(0)
        num_out = len(curr_node.children[input])

        if num_out == 1:
            res[0].append(curr_node.children[input][0].output)
            tmp = self.get_single_trace(curr_node.children[input][0], e)
            tmp_len = len(tmp)
            if tmp_len == 1:
                res[0] = res[0] + list(tmp[0])
            else:
                for counter in range(0, tmp_len):
                    if counter < tmp_len - 1:
                        res.append(res[0] + list(tmp[counter]))
                    else:
                        res[0] = res[0] + list(tmp[counter])
        else:
            tmp_pos = 0
            for counter in range(0, num_out):
                if counter < num_out - 1:
                    res.append(res[0] + [curr_node.children[input][counter].output])
                    tmp_pos = len(res) - 1

                    tmp = self.get_single_trace(curr_node.children[input][counter], e)
                    tmp_len = len(tmp)
                    if tmp_len == 1:
                        res[tmp_pos] = res[tmp_pos] + list(tmp[0])
                    else:
                        for counter2 in range(0, tmp_len):
                            if counter2 < tmp_len - 1:
                                res.append(res[tmp_pos] + list(tmp[counter2]))
                            else:
                                res[tmp_pos] = res[tmp_pos] + list(tmp[counter2])
                else:
                    res[0].append(curr_node.children[input][counter].output)

                    tmp = self.get_single_trace(curr_node.children[input][counter], e)
                    tmp_len = len(tmp)
                    if tmp_len == 1:
                        res[0] = res[0] + list(tmp[0])
                    else:
                        for counter2 in range(0, tmp_len):
                            if counter2 < tmp_len - 1:
                                res.append(res[0] + list(tmp[counter2]))
                            else:
                                res[0] = res[0] + list(tmp[counter2])

        for i in range(0, len(res)):
            res[i] = tuple(res[i])

        return res


    def get_all_traces(self, inputs, outputs, e):
        """
        Args:
          inputs:
          outputs:
          e:

        Returns:

        """
        res = {}
        curr_node = self.get_to_node(inputs, outputs, e)

        for inp in e:
            res[inp] = self.get_single_trace(curr_node, inp)

        return res


    def get_table(self, s, e):
        """
        Args:
          s:
          e:

        Returns:

        """
        res = {}
        for pair in s:
            curr_node = self.get_to_node(pair[0], pair[1])
            res[pair] = {}

            for inp in e:
                res[pair][inp] = self.get_single_trace(curr_node, inp)

        return res


    def print_trace_tree(self, curr=None, depth=0, curr_str=""):
        """
        Args:
          curr: current node. normally root of tree, but can be any node
          depth: needed for recursion. should not be changed.
          curr_str: needed for recursion. should not be changed.

        Returns: prints trace tree
        """

        if curr is None and depth == 0:
            curr = self.root_node
            print("()")

        curr_str = curr_str + " ├─ "

        # go through all inputs
        for i, node in enumerate(list(curr.children.keys())):

            # go through all outputs of a single input
            for c in range(0, len(curr.children[node])):

                # if it is the last output of the last input on the current level
                if i == len(list(curr.children.keys())) - 1 and c == len(curr.children[node]) - 1:
                    curr_str = list(curr_str)[:-4]
                    curr_str = ''.join(curr_str) + " └─ "
                elif c <= len(curr.children[node]) - 1:
                    curr_str = list(curr_str)[:-4]
                    curr_str = ''.join(curr_str) + " ├─ "

                print(curr_str + node, curr.children[node][c].output)

                # if it is the last output of the last input on the current level
                if i == len(list(curr.children.keys())) - 1 and c == len(curr.children[node]) - 1:
                    curr_str = list(curr_str)[:-4]
                    curr_str = ''.join(curr_str) + "    "

                else:
                    curr_str = list(curr_str)[:-4]
                    curr_str = ''.join(curr_str) + " |  "

                self.print_trace_tree(curr.children[node][c], depth + 1, curr_str)

