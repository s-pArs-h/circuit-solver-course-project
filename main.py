from netlist_parse import readNetList
from matrix_maker import extractNodes, solveKcl, solveKcl_matrix
import sympy as sp
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    elements_list = readNetList("netlists/test1.net")
    nodes_list = extractNodes(elements_list)
    s = sp.Symbol('s', complex=True)
    t = sp.Symbol('t', real=True)


    V_nodes = solveKcl_matrix(elements_list, nodes_list)
    v_t = sp.inverse_laplace_transform(V_nodes['n2'], s, t)
    print(v_t)

