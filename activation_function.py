from abc import ABC, abstractclassmethod

class Function(ABC):
    @staticmethod
    @abstractclassmethod
    def g(u):
        ...
class Derivative(ABC):
    @staticmethod
    @abstractclassmethod
    def dg(u):
        ...
        
class BinaryStep(Function):
    def g(u):
        return 0 if u >= -1 else -1
class SignFunction(Function):
    def g(u):
        return 0 if u >= -1 else -1
class tanh(Function, Derivative):
    def g(u):
        ...
    def dg(u):
        ...