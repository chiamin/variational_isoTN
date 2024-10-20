import copy, sys
sys.path.insert(0,'/home/chiamin/Cytnx/Install/')
import cytnx
import Pattern as pat

class NetworkFunction:
    def __init__ (self):
        self.networks = []
        self.tensors = dict()
        self.terms = []

    def add (self, pattern, coef=1.):
        #assert len(pat.GetOpenIndices(pattern)) == 0    # assert no open index
        self.networks.append([coef, pattern])

    def putTensor (self, name, T, indices):
        self.tensors[name] = [T, indices]

    def __add__(self, other):
        if isinstance(other, NetworkFunction):
            res = NetworkFunction()
            res.networks = self.networks + other.networks
            return res
        else:
            return NotImplemented

    # Compute the value of the function
    # The value is a UniTensor which can be a scalar or a tensor
    def value (self):
        # Compute the value for each term
        self.terms = []
        for coef, pattern in self.networks:
            # Define network
            network = cytnx.Network()
            network.FromString (pat.ToNetworkString(pattern))

            # Put actual tensors into the network
            for name in pattern:
                if name != "TOUT":
                    network.PutUniTensor(name, self.tensors[name][0], self.tensors[name][1])

            each = network.Launch()
            self.terms.append (each * coef)

        # Compute the overall value
        # <all_terms> is a list of UniTensor
        res = self.terms[0]
        for i in range(1,len(self.terms)):
            res = res + self.terms[i]
        return res

    def gradientNetwork (self, name):
        res = NetworkFunction()
        # Compute the derivative for each term
        for coef, network in self.networks:
            order = pat.CountOrder(network, name)
            deriv = network.copy()                      # a network dictionary
            if order == 1:
                rm_name = name
            elif order == 2:
                rm_name = name+"'"
            elif order != 0:
                print("Do not support order:",order)
                raise Exception

            if order != 0.:
                del deriv[rm_name]
                # Set the output order the same as the tensor in the network
                deriv["TOUT"] = network[rm_name]
                res.add(deriv, coef*order)
        return res

    # Return the derivative of the NetworkFunction to a specific tensor
    # The returned derivative is also a NetworkFunction
    def gradient (self, name):
        res = self.gradientNetwork(name)
        # Copy the tensors
        res.tensors = self.tensors
        # Compute the derivative tensor
        T = res.value()
        # Set the same labels as the target tensor
        T.relabels_(self.tensors[name][1])
        return T

