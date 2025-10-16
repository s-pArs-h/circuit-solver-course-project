import sympy as sp

s = sp.Symbol('s')

class Element:
    def __init__(self, name, n1,n2, value):
        self.name = name
        self.n1 = n1
        self.n2 = n2
        self.value = value

    def impedence(self,s):
        raise NotImplementedError("Define in subclass")

    def admittance(self):
        return 1 / self.impedence


class Resistor(Element):
    def impedance(self,s):
        return sp.sympify(self.value)

class Capacitor(Element):
    def impedance(self,s):
        return 1 / (s * sp.sympify(self.value))

class Inductor(Element):
    def impedance(self,s):
        return s * sp.sympify(self.value)

class VoltageSource(Element):
    def impedence(self,s):
        return 0

class CurrentSource(Element):
    def impedence(self,s):
        return sp.oo 
