import numpy as np

#Simulation Parameters
MAX_ITERATIONS = 100
TOLERANCE = 1e-6
VT = 0.02585


class Node:
    def __init__(self, index, voltage=0.0):
        self.index = index
        self.voltage = voltage


class Element:
    def __init__(self, n1: Node, n2: Node):
        self.n1 = n1
        self.n2 = n2

    def stamp(self, j, f, x_guess, delta_t, current_time):
        pass


class Resistor(Element):
    def __init__(self, value, n1, n2):
        super().__init__(n1, n2)
        self.value = value

    def stamp(self, j, f, x_guess, delta_t, current_time):
        G = 1.0 / self.value
        a, b = self.n1.index, self.n2.index

        v_a = x_guess[a]
        v_b = x_guess[b]

        j[a, a] += G
        j[b, b] += G
        j[a, b] -= G
        j[b, a] -= G

        f[a] += G * (v_a - v_b)
        f[b] -= G * (v_a - v_b)


class Capacitor(Element):
    def __init__(self, value, n1, n2):
        super().__init__(n1, n2)
        self.value = value
        self.prev_current = 0.0
        self.prev_voltage_diff = 0.0

    def stamp(self, j, f, x_guess, delta_t, current_time):
        C = self.value
        a, b = self.n1.index, self.n2.index

        v_a = x_guess[a]
        v_b = x_guess[b]
        v_c_guess = v_a - v_b

        G_eq = 2.0 * C / delta_t
        I_eq = G_eq * self.prev_voltage_diff + self.prev_current

        j[a, a] += G_eq
        j[b, b] += G_eq
        j[a, b] -= G_eq
        j[b, a] -= G_eq

        current_guess = G_eq * v_c_guess - I_eq
        f[a] += current_guess
        f[b] -= current_guess


class Inductor(Element):
    def __init__(self, value, n1, n2):
        super().__init__(n1, n2)
        self.value = value
        self.prev_current = 0.0
        self.prev_voltage_diff = 0.0

    def stamp(self, j, f, x_guess, delta_t, current_time):
        L = self.value
        a, b = self.n1.index, self.n2.index

        v_a = x_guess[a]
        v_b = x_guess[b]
        v_l_guess = v_a - v_b

        G_eq = delta_t / (2.0 * L)
        I_eq = self.prev_current + G_eq * self.prev_voltage_diff

        j[a, a] += G_eq
        j[b, b] += G_eq
        j[a, b] -= G_eq
        j[b, a] -= G_eq

        current_guess = G_eq * v_l_guess + I_eq
        f[a] += current_guess
        f[b] -= current_guess


class VoltageSource(Element):
    def __init__(self, value_func, n1, n2, aux_index):
        super().__init__(n1, n2)
        self.value_func = value_func
        self.aux_index = aux_index

    def stamp(self, j, f, x_guess, delta_t, current_time):
        a, b = self.n1.index, self.n2.index
        m = self.aux_index

        v_a = x_guess[a]
        v_b = x_guess[b]

        j[a, m] += 1
        j[b, m] -= 1

        j[m, a] += 1
        j[m, b] -= 1

        f[m] += v_a - v_b - self.value_func(current_time)


class CurrentSource(Element):
    def __init__(self, value_func, n1, n2, ):
        super().__init__(n1, n2)
        self.value_func = value_func


    def stamp(self, j, f, x_guess, delta_t, current_time):
        a, b = self.n1.index, self.n2.index

        f[a] += self.value_func(current_time)
        f[b] -= self.value_func(current_time)

class Diode(Element):

    def __init__(self, n1, n2, Is=1e-15, n=1.0):
        super().__init__(n1, n2)
        self.Is = Is
        self.n = n

    def stamp(self, j, f, x_guess, delta_t, current_time):
        a, b = self.n1.index, self.n2.index
        v_d = x_guess[a] - x_guess[b]

        V_d_crit_fwd = 0.8
        if v_d > V_d_crit_fwd:
            g_crit = self.Is / (self.n * VT) * np.exp(V_d_crit_fwd / (self.n * VT))
            i_crit = self.Is * (np.exp(V_d_crit_fwd / (self.n * VT)) - 1)
            g_d = g_crit
            i_d = i_crit + g_crit * (v_d - V_d_crit_fwd)
        else:
            exp_val = np.exp(v_d / (self.n * VT))
            i_d = self.Is * (exp_val - 1)
            g_d = (self.Is / (self.n * VT)) * exp_val

        j[a, a] += g_d
        j[b, b] += g_d
        j[a, b] -= g_d
        j[b, a] -= g_d

        f[a] += i_d
        f[b] -= i_d


class BJT_NPN(Element):

    def __init__(self, nc, nb, ne, Is=1e-14, bf=100, br=1):
        self.nc = nc
        self.nb = nb
        self.ne = ne
        self.Is = Is
        self.bf = bf
        self.br = br

    def stamp(self, j, f, x_guess, delta_t, current_time):
        c, b, e = self.nc.index, self.nb.index, self.ne.index

        # Get current voltage guesses
        v_c, v_b, v_e = x_guess[c], x_guess[b], x_guess[e]
        v_be = v_b - v_e
        v_bc = v_b - v_c

        v_be = np.clip(v_be, -1.0, 0.85)
        v_bc = np.clip(v_bc, -1.0, 0.85)

        exp_vbe = np.exp(v_be / VT)
        exp_vbc = np.exp(v_bc / VT)

        i_fwd = self.Is * (exp_vbe - 1)
        i_rev = self.Is * (exp_vbc - 1)

        i_c = (i_fwd / self.Is) * (self.Is * self.bf / self.bf) - i_rev - (
                    i_rev / self.br)
        i_b = (i_fwd / self.bf) + (i_rev / self.br)
        i_c = i_fwd - i_rev * (1 + 1 / self.br)

        g_pi = (self.Is / (self.bf * VT)) * exp_vbe
        g_mu = (self.Is / (self.br * VT)) * exp_vbc
        g_m = (self.Is / VT) * exp_vbe
        g_o = (self.Is / VT) * exp_vbc * (1 + 1 / self.br)

        f[c] += i_c
        f[b] += i_b
        f[e] -= (i_c + i_b)

        j[c, c] += g_o
        j[c, b] += g_m - g_o
        j[c, e] -= g_m

        j[b, c] -= g_mu
        j[b, b] += g_pi + g_mu
        j[b, e] -= g_pi

        j[e, c] -= (g_o - g_mu)
        j[e, b] -= (g_m - g_o + g_pi + g_mu)
        j[e, e] += g_m + g_pi


def simulate(elements, nodes, num_v_sources, t_end, delta_t):
    num_nodes = len(nodes)
    n = num_nodes + num_v_sources
    results = {'time': []}
    for i in range(num_nodes):
        results[f'v{i}'] = []

    x_prev = np.zeros(n)
    for node in nodes:
        x_prev[node.index] = node.voltage

    # --- 1. OUTER LOOP: Time Stepping ---
    current_time = 0.0
    while current_time <= t_end:
        x_guess = np.copy(x_prev)

        # --- 2. INNER LOOP: Newton-Raphson ---
        for iter_count in range(MAX_ITERATIONS):
            j = np.zeros((n, n))
            f = np.zeros(n)

            for elem in elements:
                elem.stamp(j, f, x_guess, delta_t, current_time)

            j[0, :] = 0.0
            j[:, 0] = 0.0
            j[0, 0] = 1.0
            f[0] = x_guess[0] - 0.0

            delta_x = np.linalg.solve(j, -f)

            x_guess += delta_x

            norm = np.linalg.norm(delta_x[:num_nodes])
            if norm < TOLERANCE:
                break
            else:
                print(f"  Warning: Newton-Raphson did not converge at t={current_time}s.")
        x_prev = np.copy(x_guess)

        for elem in elements:
            if isinstance(elem, (Capacitor, Inductor)):
                v_a_n = x_prev[elem.n1.index]
                v_b_n = x_prev[elem.n2.index]
                v_n = v_a_n - v_b_n

                v_prev = elem.prev_voltage_diff
                i_prev = elem.prev_current

                if isinstance(elem, Capacitor):
                    G_eq = 2.0 * elem.value / delta_t
                    elem.prev_current = G_eq * v_n - (G_eq * v_prev + i_prev)
                elif isinstance(elem, Inductor):
                    G_eq = delta_t / (2.0 * elem.value)
                    elem.prev_current = G_eq * v_n + (G_eq * v_prev + i_prev)

                elem.prev_voltage_diff = v_n

        results['time'].append(current_time)
        for i in range(num_nodes):
            results[f'v{i}'].append(x_prev[i])

        current_time += delta_t

    return results


# --- Example: Half-Wave Rectifier with RC Filter ---
# if __name__ == '__main__':
#     # Circuit: Vin --- D1 --- (Node 1) ---+--- R_load --- (Node 0 / GND)
#     #                                      |
#     #                                      +--- C_filter -- (Node 0 / GND)
#     # Source is between Node 2 and GND.
#
#     # Define Nodes
#     gnd = Node(0, 0.0)  # Node 0 is always ground
#     n1 = Node(1, 0.0)  # Output node
#     n2 = Node(2, 0.0)  # Input node from source
#
#     nodes = [gnd, n1, n2]
#     num_v_sources = 1
#
#     # Define Source: 5V peak, 60Hz sine wave
#     v1_func = lambda t: 5.0 * np.sin(2 * np.pi * 60 * t)
#
#     # Define Elements
#     elements = [
#         # The aux_index must be after all node indices. num_nodes-1 is the last node index.
#         VoltageSource(v1_func, n2, gnd, aux_index=len(nodes)),
#         Diode(n2, n1),  # Diode from input to output
#         Resistor(1000, n1, gnd),  # 1k Ohm load resistor
#         Capacitor(10e-6, n1, gnd)  # 10uF filter capacitor
#     ]
#
#     # Run simulation
#     t_end = 5 / 60  # Simulate for 5 cycles
#     delta_t = 1e-5
#     results = simulate(elements, nodes, num_v_sources, t_end, delta_t)
#
#
#
#
