import numpy as np
import matplotlib.pyplot as plt
import re

# ====================================================================
# --- GLOBAL PARAMETERS AND HELPERS ---
# ====================================================================

MAX_ITERATIONS = 100
TOLERANCE = 1e-6
VT = 0.02585  # Thermal voltage at room temperature (V)
V_d_crit_fwd = 0.85  # Clipping voltage for Diode/BJT stability


# Function to parse SPICE-like time-dependent sources (e.g., SIN, PULSE)
def parse_source_function(definition, V_DC=0.0):
    """Creates a callable function from a string definition."""
    if isinstance(definition, (int, float)):
        return lambda t: float(definition)

    definition = str(definition)
    match_sin = re.match(r'SIN\(([^ ]+) ([^ ]+) ([^ ]+)\)', definition)
    if match_sin:
        V_offset = float(match_sin.group(1))
        V_peak = float(match_sin.group(2))
        freq = float(match_sin.group(3))
        # V(t) = V_offset + V_peak * sin(2 * pi * f * t)
        return lambda t: V_offset + V_peak * np.sin(2 * np.pi * freq * t)

    try:
        dc_val = float(definition)
        return lambda t: dc_val
    except ValueError:
        return lambda t: V_DC


# ====================================================================
# --- MNA ELEMENT CLASSES ---
# (Includes calculate_current for branch reporting)
# ====================================================================

class Node:
    """Represents a circuit node."""

    def __init__(self, index, name, voltage=0.0):
        self.index = index
        self.name = name
        self.voltage = voltage


class Element:
    """Base class for all circuit elements."""

    def __init__(self, n1: Node, n2: Node):
        self.n1 = n1
        self.n2 = n2

    def stamp(self, j, f, x_guess, delta_t, current_time):
        """Stamps element contribution onto Jacobian (J) and Residual (f)."""
        pass

    def calculate_current(self, x_final, delta_t, current_time):
        """Calculates the current leaving n1 and entering n2 (Branch Current)."""
        return 0.0


class Resistor(Element):
    def __init__(self, value, n1, n2):
        super().__init__(n1, n2)
        self.value = value

    def stamp(self, j, f, x_guess, delta_t, current_time):
        G = 1.0 / self.value
        a, b = self.n1.index, self.n2.index
        v_a, v_b = x_guess[a], x_guess[b]

        j[a, a] += G
        j[b, b] += G
        j[a, b] -= G
        j[b, a] -= G

        f[a] += G * (v_a - v_b)
        f[b] -= G * (v_a - v_b)

    def calculate_current(self, x_final, delta_t, current_time):
        v_a, v_b = x_final[self.n1.index], x_final[self.n2.index]
        return (v_a - v_b) / self.value


class Capacitor(Element):
    def __init__(self, value, n1, n2):
        super().__init__(n1, n2)
        self.value = value
        self.prev_current = 0.0
        self.prev_voltage_diff = 0.0

    def stamp(self, j, f, x_guess, delta_t, current_time):
        C = self.value
        a, b = self.n1.index, self.n2.index
        v_a, v_b = x_guess[a], x_guess[b]
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

    def calculate_current(self, x_final, delta_t, current_time):
        C = self.value
        v_a, v_b = x_final[self.n1.index], x_final[self.n2.index]
        v_c_final = v_a - v_b

        G_eq = 2.0 * C / delta_t
        I_eq = G_eq * self.prev_voltage_diff + self.prev_current

        return G_eq * v_c_final - I_eq


class Inductor(Element):
    def __init__(self, value, n1, n2):
        super().__init__(n1, n2)
        self.value = value
        self.prev_current = 0.0
        self.prev_voltage_diff = 0.0

    def stamp(self, j, f, x_guess, delta_t, current_time):
        L = self.value
        a, b = self.n1.index, self.n2.index
        v_a, v_b = x_guess[a], x_guess[b]
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

    def calculate_current(self, x_final, delta_t, current_time):
        L = self.value
        v_a, v_b = x_final[self.n1.index], x_final[self.n2.index]
        v_l_final = v_a - v_b

        G_eq = delta_t / (2.0 * L)
        I_eq = self.prev_current + G_eq * self.prev_voltage_diff

        return G_eq * v_l_final + I_eq


class VoltageSource(Element):
    def __init__(self, value_func, n1, n2, aux_index):
        super().__init__(n1, n2)
        self.value_func = value_func
        self.aux_index = aux_index

    def stamp(self, j, f, x_guess, delta_t, current_time):
        a, b = self.n1.index, self.n2.index
        m = self.aux_index

        v_a, v_b = x_guess[a], x_guess[b]
        V_s = self.value_func(current_time)

        j[a, m] += 1
        j[b, m] -= 1
        j[m, a] += 1
        j[m, b] -= 1
        f[m] += v_a - v_b - V_s

    def calculate_current(self, x_final, delta_t, current_time):
        return x_final[self.aux_index]


class CurrentSource(Element):
    def __init__(self, value_func, n1, n2):
        super().__init__(n1, n2)
        self.value_func = value_func

    def stamp(self, j, f, x_guess, delta_t, current_time):
        a, b = self.n1.index, self.n2.index
        I_s = self.value_func(current_time)
        f[a] += I_s
        f[b] -= I_s

    def calculate_current(self, x_final, delta_t, current_time):
        return self.value_func(current_time)


class Diode(Element):
    def __init__(self, n1, n2, Is=1e-15, n=1.0):
        super().__init__(n1, n2)
        self.Is = Is
        self.n = n

    def stamp(self, j, f, x_guess, delta_t, current_time):
        a, b = self.n1.index, self.n2.index
        v_d = x_guess[a] - x_guess[b]

        if v_d > V_d_crit_fwd:
            g_crit = self.Is / (self.n * VT) * np.exp(V_d_crit_fwd / (self.n * VT))
            i_crit = self.Is * (np.exp(V_d_crit_fwd / (self.n * VT)) - 1)
            g_d = g_crit
            i_d = i_crit + g_crit * (v_d - V_d_crit_fwd)
        else:
            arg = v_d / (self.n * VT)
            exp_val = np.exp(arg)
            i_d = self.Is * (exp_val - 1)
            g_d = (self.Is / (self.n * VT)) * exp_val

        j[a, a] += g_d
        j[b, b] += g_d
        j[a, b] -= g_d
        j[b, a] -= g_d
        f[a] += i_d
        f[b] -= i_d

    def calculate_current(self, x_final, delta_t, current_time):
        v_d = x_final[self.n1.index] - x_final[self.n2.index]

        if v_d > V_d_crit_fwd:
            g_crit = self.Is / (self.n * VT) * np.exp(V_d_crit_fwd / (self.n * VT))
            i_crit = self.Is * (np.exp(V_d_crit_fwd / (self.n * VT)) - 1)
            return i_crit + g_crit * (v_d - V_d_crit_fwd)
        else:
            arg = v_d / (self.n * VT)
            return self.Is * (np.exp(arg) - 1)


class BJT_NPN:  # 3-terminal devices do not inherit from Element base class
    def __init__(self, nc, nb, ne, Is=1e-14, bf=100.0, br=1.0):
        self.nc, self.nb, self.ne = nc, nb, ne
        self.Is, self.bf, self.br = Is, bf, br
        self.alpha_f = bf / (1.0 + bf)
        self.alpha_r = br / (1.0 + br)

    def _calculate_currents(self, v_c, v_b, v_e):
        """Helper to calculate terminal currents based on terminal voltages."""
        v_be = v_b - v_e
        v_bc = v_b - v_c

        v_be = np.clip(v_be, a_min=None, a_max=V_d_crit_fwd)
        v_bc = np.clip(v_bc, a_min=None, a_max=V_d_crit_fwd)

        exp_be = np.exp(v_be / VT)
        exp_bc = np.exp(v_bc / VT)

        I_BE = self.Is * (exp_be - 1.0)
        I_BC = self.Is * (exp_bc - 1.0)
        g_be = self.Is / VT * exp_be
        g_bc = self.Is / VT * exp_bc

        I_C = self.alpha_f * I_BE - I_BC * (1.0 + 1.0 / self.br)
        I_E = -I_BE * (1.0 + 1.0 / self.bf) + self.alpha_r * I_BC
        I_B = -(I_C + I_E)

        return I_C, I_B, I_E, g_be, g_bc

    def stamp(self, j, f, x_guess, delta_t, current_time):
        c, b, e = self.nc.index, self.nb.index, self.ne.index
        v_c, v_b, v_e = x_guess[c], x_guess[b], x_guess[e]

        I_C, I_B, I_E, g_be, g_bc = self._calculate_currents(v_c, v_b, v_e)

        j[c, b] += self.alpha_f * g_be - g_bc * (1.0 + 1.0 / self.br)
        j[c, e] -= self.alpha_f * g_be
        j[c, c] += g_bc * (1.0 + 1.0 / self.br)

        j[e, b] -= g_be * (1.0 + 1.0 / self.bf) + self.alpha_r * g_bc
        j[e, e] += g_be * (1.0 + 1.0 / self.bf)
        j[e, c] -= self.alpha_r * g_bc

        j[b, c] = -(j[c, c] + j[e, c])
        j[b, b] = -(j[c, b] + j[e, b])
        j[b, e] = -(j[c, e] + j[e, e])

        f[c] += I_C
        f[b] += I_B
        f[e] += I_E

    def calculate_current(self, x_final, delta_t, current_time):
        """Returns I_C (Collector Current) as the primary branch current."""
        v_c, v_b, v_e = x_final[self.nc.index], x_final[self.nb.index], x_final[self.ne.index]
        I_C, I_B, I_E, _, _ = self._calculate_currents(v_c, v_b, v_e)
        return I_C


# ====================================================================
# --- MNA/NR CORE ENGINE ---
# ====================================================================

def update_dynamic_history(elements, x_prev, delta_t):
    """Updates the previous voltage and current for C and L for the next time step."""
    for elem in elements:
        if isinstance(elem, (Capacitor, Inductor)):
            v_a = x_prev[elem.n1.index]
            v_b = x_prev[elem.n2.index]
            elem.prev_voltage_diff = v_a - v_b

            if isinstance(elem, Capacitor):
                G_eq = 2.0 * elem.value / delta_t
                I_eq = G_eq * elem.prev_voltage_diff + elem.prev_current
                elem.prev_current = G_eq * elem.prev_voltage_diff - I_eq
            elif isinstance(elem, Inductor):
                G_eq = delta_t / (2.0 * elem.value)
                I_eq = elem.prev_current + G_eq * elem.prev_voltage_diff
                elem.prev_current = G_eq * elem.prev_voltage_diff + I_eq


def simulate(elements_named, nodes_map, n_equations, t_end, delta_t):
    """Runs the full transient simulation and collects results."""

    num_nodes = len(nodes_map)
    n = n_equations

    voltage_steps = []
    current_steps = []

    # Initialize state vector (x_prev)
    x_prev = np.zeros(n)
    for node_obj in nodes_map.values():
        x_prev[node_obj.index] = node_obj.voltage

    current_time = 0.0
    convergence_status = 1
    elements = [e for n, e in elements_named]

    # Calculate the number of steps needed
    num_steps = int(np.floor(t_end / delta_t)) + 1

    for step_count in range(num_steps):
        current_time = step_count * delta_t
        x_guess = np.copy(x_prev)
        converged_in_step = False

        # --- Newton-Raphson Loop ---
        for iter_count in range(MAX_ITERATIONS):
            j = np.zeros((n, n))
            f = np.zeros(n)

            for name, elem in elements_named:
                elem.stamp(j, f, x_guess, delta_t, current_time)

            delta_x, _, _, _ = np.linalg.lstsq(j, -f, rcond=None)
            x_guess += delta_x

            norm = np.linalg.norm(delta_x)
            if norm < TOLERANCE:
                converged_in_step = True
                break

        if not converged_in_step:
            convergence_status = -1
            # print(f"Warning: NR failed at t={current_time:.4e}s. Norm: {norm:.2e}")

        # --- Record Results (Use last guess, even if non-converged) ---
        x_final = x_guess

        # 1. Node Voltages
        node_voltages = x_final[:num_nodes]
        voltage_steps.append(node_voltages)

        # 2. Branch Currents
        branch_currents = []
        for name, elem in elements_named:
            branch_currents.append(elem.calculate_current(x_final, delta_t, current_time))
        current_steps.append(np.array(branch_currents))

        # --- Post-NR Updates ---
        x_prev = x_final
        update_dynamic_history(elements, x_prev, delta_t)

    # Convert lists of arrays into final NumPy matrices
    voltage_matrix = np.array(voltage_steps)
    current_matrix = np.array(current_steps)

    return voltage_matrix, current_matrix, convergence_status


# ====================================================================
# --- SOLVER INTERFACE (Netlist Parser) ---
# ====================================================================

class CircuitSolver:
    """Parses a netlist and runs the simulation."""

    def __init__(self):
        self._reset_solver()

    def _reset_solver(self):
        self.nodes_map = {}  # Maps name to Node object
        self.elements_named = []  # Stores (name, element_obj)
        self.next_node_index = 0
        self.next_aux_index = 0
        self.node_names_ordered = []
        self.branch_names_ordered = []
        self.ground_node_name = None

    def _get_or_create_node(self, node_name):
        """Ensures a node exists and returns its object."""
        node_name = str(node_name)
        if node_name not in self.nodes_map:
            # Check if this node is supposed to be ground
            if node_name == self.ground_node_name:
                index = 0
            else:
                # Find the next non-ground index
                index = self.next_node_index
                if index == 0:
                    self.next_node_index += 1
                    index = self.next_node_index  # Skip index 0 if it's not the ground node being created here

                self.next_node_index += 1

            new_node = Node(index, name=node_name)
            self.nodes_map[node_name] = new_node

            # Reorder node names to match matrix index (if possible, but MNA handles indices)
            if node_name not in self.node_names_ordered:
                self.node_names_ordered.append(node_name)
                self.node_names_ordered.sort(key=lambda name: self.nodes_map[name].index)

        return self.nodes_map[node_name]

    def _get_ground_node(self, ground_node_name):
        """Sets up the ground node mapping and index 0."""
        ground_node_name = str(ground_node_name)
        self.ground_node_name = ground_node_name

        # Ensure the ground node is always created first and assigned index 0
        if ground_node_name not in self.nodes_map:
            self.nodes_map[ground_node_name] = Node(0, name=ground_node_name, voltage=0.0)
            self.node_names_ordered.append(ground_node_name)
            self.next_node_index = 1  # Start next index at 1
        else:
            # If it already existed, ensure its index is 0
            self.nodes_map[ground_node_name].index = 0

    def _parse_element(self, element_def, name_suffix):
        """Parses a single element definition [Type, N1, N2, Value...]."""
        type_char = element_def[0].upper()
        name = type_char + str(name_suffix)

        try:
            elem = None
            n1 = self._get_or_create_node(element_def[1])
            n2 = self._get_or_create_node(element_def[2])
            value = element_def[3]  # Can be a float or a function string

            if type_char == 'R':
                elem = Resistor(float(value), n1, n2)

            elif type_char == 'C':
                elem = Capacitor(float(value), n1, n2)

            elif type_char == 'L':
                elem = Inductor(float(value), n1, n2)

            elif type_char == 'V':
                value_func = parse_source_function(value)
                aux_index = self.next_node_index + self.next_aux_index
                self.next_aux_index += 1
                elem = VoltageSource(value_func, n1, n2, aux_index)

            elif type_char == 'I':
                value_func = parse_source_function(value)
                elem = CurrentSource(value_func, n1, n2)

            elif type_char == 'D':
                elem = Diode(n1, n2)

            elif type_char == 'Q':
                n3 = self._get_or_create_node(element_def[3])  # Emitter node for BJT
                n_c, n_b, n_e = n1, n2, n3  # Q N1 N2 N3 -> Collector Base Emitter
                elem = BJT_NPN(n_c, n_b, n_e)

            if elem:
                self.elements_named.append((name, elem))
                self.branch_names_ordered.append(name)
                return True

        except Exception as e:
            print(f"Error parsing element {name}: {e}")
            return False
        return False

    def _format_final_output(self, V_matrix, I_matrix, status):
        """
        Formats the final time step's results into the requested two 2D matrices.
        """
        if V_matrix.size == 0:
            return np.array([]), np.array([]), status

        # 1. Node Voltage Matrix (2 x N)
        final_voltages = V_matrix[-1, :]

        # Sort node indices by MNA index (0, 1, 2...)
        sorted_nodes = sorted(self.nodes_map.values(), key=lambda node: node.index)

        node_indices = np.array([node.name for node in sorted_nodes])
        node_values = final_voltages[[node.index for node in sorted_nodes]]

        node_voltage_matrix = np.vstack((node_indices, node_values))

        # 2. Branch Current Matrix (B x 3)
        final_currents = I_matrix[-1, :]
        branch_rows = []

        for i, (name, elem) in enumerate(self.elements_named):
            if isinstance(elem, BJT_NPN):
                # For BJT (Q N1 N2 N3 -> C B E), we report C to E current (I_C)
                n1_name = elem.nc.name
                n2_name = elem.ne.name
            else:
                # For 2-terminal (R, L, C, V, I, D) -> N1 to N2 current
                n1_name = elem.n1.name
                n2_name = elem.n2.name

            branch_rows.append([n1_name, n2_name, final_currents[i]])

        branch_current_matrix = np.array(branch_rows, dtype=object)

        return node_voltage_matrix, branch_current_matrix, status

    def solve_circuit(self, netlist_def: list, ground_node_name: int, t_end=0.0, delta_t=1e-6):
        """
        Main entry point to solve the circuit based on the required I/O contract.

        Input:
          - netlist_def: List of element definitions [Type, N1, N2, Value...]
          - ground_node_name: The name/index of the node that is ground (e.g., '0' or 0)
          - t_end: Final simulation time.
          - delta_t: Time step.

        Output:
          - node_voltage_matrix: 2D array (2 x N) with [Node Name/Index, Final Voltage]
          - branch_current_matrix: 2D array (B x 3) with [N1 Name/Index, N2 Name/Index, Final Current]
          - status: Integer (1 for converged, -1 for non-converged)
        """
        self._reset_solver()
        self._get_ground_node(ground_node_name)

        for i, element_def in enumerate(netlist_def):
            self._parse_element(element_def, i + 1)  # Use 1-based index for naming

        n_equations = self.next_node_index + self.next_aux_index
        if n_equations <= 1:  # Only ground node, or no elements
            print("Error: Circuit not properly defined.")
            return np.array([]), np.array([]), -1

        print(
            f"Solving circuit with {len(self.nodes_map)} nodes and {self.next_aux_index} auxiliary variables (total equations: {n_equations}).")

        # 1. Run full transient simulation
        V_matrix_full, I_matrix_full, status = simulate(
            self.elements_named,
            self.nodes_map,
            n_equations,
            t_end,
            delta_t
        )

        # 2. Format final output as requested (only last time step)
        node_voltage_matrix, branch_current_matrix, status = self._format_final_output(
            V_matrix_full, I_matrix_full, status
        )

        return node_voltage_matrix, branch_current_matrix, status


# ====================================================================
# --- USAGE EXAMPLE: DC BJT Bias Point ---
# The example usage demonstrates the exact input and output structure.
# ====================================================================

if __name__ == '__main__':
    # 1. Define the Netlist and parameters

    # Q1 N_C N_B N_E
    # V1 N_IN 0 5 (DC 5V Source)
    # R1 N_C N_IN 1k (Collector Resistor)
    # R2 N_B 0 10k (Base Resistor)

    netlist_definition = [
        ['V', 'N_IN', 0, 5],
        ['R', 'N_C', 'N_IN', 1000],
        ['R', 'N_B', 0, 10000],
        ['Q', 'N_C', 'N_B', 0]  # Collector, Base, Emitter=GND
    ]

    GROUND_NODE = 0
    T_END = 1e-3  # Simulate until DC steady state
    DELTA_T = 1e-6

    # 2. Initialize and Solve
    solver = CircuitSolver()
    V_final, I_final, status = solver.solve_circuit(
        netlist_definition,
        GROUND_NODE,
        t_end=T_END,
        delta_t=DELTA_T
    )

    # 3. Display Results
    print("\n--- Final Output Matrices ---")
    print(f"Convergence Status: {status} (1=Converged, -1=Failed)")

    print("\n[Output 1] Node Voltage Matrix (2 x N)")
    print("Row 1: Node Name | Row 2: Voltage (V)")
    print(V_final)

    print("\n[Output 2] Branch Current Matrix (B x 3)")
    print("Col 1: N1 | Col 2: N2 | Col 3: Current (A)")
    print(I_final)

    # Example for plotting (optional, runs in environment)
    # The solver also stores the full time-series data internally if needed for plotting

    # Rerun the transient sim to get full data for plotting the transient
    V_matrix_full, _, _ = simulate(
        solver.elements_named, solver.nodes_map, solver.next_node_index + solver.next_aux_index, T_END, DELTA_T
    )
    time = np.linspace(0, T_END, V_matrix_full.shape[0])

    if time.size > 0 and V_final.size > 0:
        plt.figure(figsize=(10, 6))

        # Plot V_C (Collector Voltage)
        vc_index = solver.nodes_map['N_C'].index

        plt.plot(time, V_matrix_full[:, vc_index], label='V_C (Collector Voltage)')

        plt.title('BJT Bias Point Transient Analysis')
        plt.xlabel('Time (s)')
        plt.ylabel('Voltage (V)')
        plt.grid(True)
        plt.legend()
        plt.show()
