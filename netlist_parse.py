from elements import Resistor, Capacitor, Inductor, VoltageSource, CurrentSource

ELEMENT_CLASSES = {
    "R": Resistor,
    "C": Capacitor,
    "L": Inductor,
    "V": VoltageSource,
    "I": CurrentSource,
}

def readNetList(filename):
    with open(filename) as file:
        lines = [line.rstrip() for line in file]
    element_list = []
    for line in lines:
        element_list.append(parseNetList(line))
    return element_list 


def parseNetList(line):
    items = line.split(" ")
    if len(items)!=4:
        raise ValueError("item size is wrong")
    name = items[0]
    n1 = items[1]
    n2 = items[2]
    value = float(items[3])
    element_type = name[0].upper()
    if element_type not in ELEMENT_CLASSES:
        raise ValueError("item is invalid")
    return ELEMENT_CLASSES[element_type](name,n1,n2,value)

