
import sympy as sp

def extractNodes(elements):
    nodes = set()
    for e in elements:
        nodes.add(e.n1)
        nodes.add(e.n2)
    nodes.discard("n0")  
    return list(nodes)


def incidenceMatrix(elements, nodes):
    A = []
    for n in nodes:
        row = []
        for e in elements:
            if e.n1 == n:
                row.append(1)
            elif e.n2 == n:
                row.append(-1)
            else:
                row.append(0)
        A.append(row)
    return sp.Matrix(A)


import sympy as sp

def solveKcl(elements, nodes):
    s = sp.Symbol('s', complex=True)

    V_nodes = {n: sp.Symbol(f"V_{n}") for n in nodes}

    I_inj = {n: 0 for n in nodes}

    passive_elems = []
    voltage_sources = []
    current_sources = []

    for e in elements:
        t = e.name[0].upper()
        if t in ["R", "L", "C"]:
            passive_elems.append(e)
        elif t == "V":
            voltage_sources.append(e)
        elif t == "I":
            current_sources.append(e)

    kcl_eqs = []

    for n in nodes:
        eq = 0
        for e in passive_elems:
            Z = e.impedance(s)
            V1 = V_nodes[e.n1] if e.n1 != 'n0' else 0
            V2 = V_nodes[e.n2] if e.n2 != 'n0' else 0
            if e.n1 == n or e.n2 == n:
                eq += (V1 - V2) / Z
        for e in current_sources:
            I_val = sp.sympify(e.value)
            if e.n1 == n:
                eq -= I_val  
            if e.n2 == n:
                eq += I_val
        kcl_eqs.append(sp.Eq(eq, 0))

    for e in voltage_sources:
        V1 = 0 if e.n1 == 'n0' else V_nodes[e.n1]
        V2 = 0 if e.n2 == 'n0' else V_nodes[e.n2]
        kcl_eqs.append(sp.Eq(V1 - V2, sp.sympify(e.value)))

    sol = sp.solve(kcl_eqs, list(V_nodes.values()), dict=True)
    if not sol:
        raise ValueError("Could not solve KCL equations.")
    sol = sol[0]

    kcl_dict = {}
    element_currents = {}

    for n in nodes:
        kcl_dict[n] = sol[V_nodes[n]]

    for e in elements:
        V1 = 0 if e.n1 == 'n0' else sol[V_nodes[e.n1]]
        V2 = 0 if e.n2 == 'n0' else sol[V_nodes[e.n2]]

        t = e.name[0].upper()
        if t in ["R", "L", "C"]:
            I = (V1 - V2) / e.impedance(s)
        elif t == "I":
            I = sp.sympify(e.value)
        elif t == "V":
            I = 0
            for n in nodes:
                if e.n1 == n or e.n2 == n:
                    pass  
            I = sp.Symbo(f"I_{e.name}") 
        element_currents[e.name] = sp.simplify(I)

    return kcl_dict, element_currents


def solveKcl_matrix(elements, nodes):
    s = sp.Symbol('s', complex=True)
    V_syms = [sp.Symbol(f"V_{n}") for n in nodes]

    # Build coefficient matrix and RHS
    A = sp.zeros(len(nodes))
    b = sp.zeros(len(nodes), 1)

    for i, n in enumerate(nodes):
        for e in elements:
            t = e.name[0].upper()
            if t in ["R", "L", "C"]:
                Z = e.impedance(s)
                V1 = 1 if e.n1 == n else -1 if e.n2 == n else 0
                V2 = 1 if e.n2 == n else -1 if e.n1 == n else 0
                j1 = nodes.index(e.n1) if e.n1 in nodes else None
                j2 = nodes.index(e.n2) if e.n2 in nodes else None
                if j1 is not None:
                    A[i, j1] += 1/Z if e.n1 == n else -1/Z
                if j2 is not None:
                    A[i, j2] += 1/Z if e.n2 == n else -1/Z
            elif t == "I":
                I_val = sp.sympify(e.value)
                if e.n1 == n:
                    b[i] -= I_val
                if e.n2 == n:
                    b[i] += I_val
            elif t == "V":
                # Voltage source handled separately
                pass

    # Voltage sources: add as constraints (replace rows)
    for e in elements:
        if e.name[0].upper() == "V":
            idx1 = nodes.index(e.n1) if e.n1 in nodes else None
            idx2 = nodes.index(e.n2) if e.n2 in nodes else None
            # replace first row for this source
            row = sp.zeros(1, len(nodes))
            if idx1 is not None:
                row[0, idx1] = 1
            if idx2 is not None:
                row[0, idx2] = -1
            A[0, :] = row
            b[0] = sp.sympify(e.value)

    V_sol = A.LUsolve(b)
    return {n: V_sol[i] for i, n in enumerate(nodes)}
