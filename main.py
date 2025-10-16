import numpy as np
from elements import Resistor, Capacitor, Inductor, Diode, BJT_NPN, VoltageSource, CurrentSource

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

