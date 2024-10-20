import copy, sys
sys.path.insert(0,'/home/chiamin/Cytnx/Install/')
import cytnx
import numpy as np
sys.path.append('../')
import Pattern as pat
import NetworkFunction as netf
import Hamiltonian as hamilt
import UniTensorTools as uniten
import Utility as ut

class CostFunction:
    def __init__ (self, network_function):
        self.nets = network_function

    def setGarget (self, tensor_name):
        self.name = tensor_name

    def updateTensor (self, name, tensor):
        # Update the tensor
        labels = self.nets.tensors[name][1]
        self.nets.putTensor(name, tensor, labels)
        # Update the "Prime" tensor if exist
        name = name+"'"
        if name in self.nets.tensors:
            labels = self.nets.tensors[name][1]
            self.nets.putTensor(name, tensor.Dagger(), labels)

    def gradient (self, tensor):
        self.updateTensor(self.name, tensor)
        return self.nets.gradient(self.name)

    # ToDo: Can be optimized by storing the intermediate tensors
    def value (self, tensor):
        self.updateTensor(self.name, tensor)
        val = self.nets.value()
        return val
