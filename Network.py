


def GetOpenIndices (network_dict):
    # Store all the indices
    inds_all = []
    for name, inds in network_dict.items():
        inds_all.extend(inds)

    # Store the open indices which appear only once
    open_inds = []
    for ind in inds_all:
        if inds_all.count(ind) == 1:
            open_inds.append(ind)
    return open_inds

class TensorPattern:
    def __init__ (self, name, labels):
        self.name = name
        self.labels = labels

    def __add__ (self, other):
        if isinstance(other, Network):
            self.name = self.name+"+"+other.name

            order1 = GetOpenIndices(self.dict)
            order2 = GetOpenIndices(other.dict)
            assert order1 == order2

            

    def __imul__(self, scalar):
        assert isinstance(x, (int, float, complex))
        for term in self.terms:
            term[0] *= scalar

    def __add__ (self, network):

    def __sub__ (self, network):
        self.add(pattern, -1.])

