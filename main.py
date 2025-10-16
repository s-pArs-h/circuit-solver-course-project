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


