import tkinter as tk
from tkinter import ttk, messagebox, font
import math
import logging
from datetime import datetime
from collections import defaultdict

# --- OPTIMIZATION ENGINE ---
# Make sure you have pulp installed: pip install pulp
import pulp

# --- Configuration ---
# Configure logging to output to console and a file
LOG_FILE = f"simulador_acero_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(LOG_FILE),
                        logging.StreamHandler()
                    ])


# --- Data Structures & Classes ---

class Column:
    """
    Represents a specific type of reinforced concrete column.
    This class holds the design parameters. In a real application, this would
    be loaded from a database or a design file.
    """

    def __init__(self, name, width, length, height, cover,
                 long_bar_type, long_bar_qty,
                 stirrup_bar_type, stirrup_spacing_confined,
                 stirrup_spacing_central, confined_length_factor=1 / 6):
        self.name = name
        self.width = width  # cm
        self.length = length  # cm
        self.height = height  # m
        self.cover = cover  # cm
        self.long_bar_type = long_bar_type
        self.long_bar_qty = long_bar_qty
        self.stirrup_bar_type = stirrup_bar_type
        self.stirrup_spacing_confined = stirrup_spacing_confined  # cm
        self.stirrup_spacing_central = stirrup_spacing_central  # cm
        # As per E.060, confined zone is often L/6 at ends
        self.confined_length = self.height * confined_length_factor * 100  # cm

    def calculate_requirements(self):
        """
        Calculates a DETAILED breakdown of required rebar pieces.
        This structured output is crucial for clear UI representation.
        """
        # 1. Longitudinal Bars
        long_bar_length = self.height  # meters
        longitudinal_req = {
            'bar_type': self.long_bar_type,
            'pieces': [round(long_bar_length, 2)] * self.long_bar_qty,
            'description': f"{self.long_bar_qty} barras de {long_bar_length}m c/u"
        }

        # 2. Stirrups (Estribos)
        stirrup_width = self.width - 2 * self.cover
        stirrup_length = self.length - 2 * self.cover
        hook_length = 10 * 2  # two hooks, simplified
        total_stirrup_perimeter = (2 * stirrup_width + 2 * stirrup_length + hook_length) / 100  # meters

        # Calculate stirrup quantity based on spacing
        qty_confined_zone = math.ceil(self.confined_length / self.stirrup_spacing_confined)
        qty_confined_total = qty_confined_zone * 2

        central_length = (self.height * 100) - (2 * self.confined_length)
        qty_central = math.ceil(central_length / self.stirrup_spacing_central) if central_length > 0 else 0
        total_stirrups = qty_confined_total + qty_central

        dist_string = (f"1 @ 5cm, {qty_confined_zone - 1} @ {self.stirrup_spacing_confined}cm (inf), "
                       f"{qty_central} @ {self.stirrup_spacing_central}cm (cen), "
                       f"{qty_confined_zone - 1} @ {self.stirrup_spacing_confined}cm, 1 @ 5cm (sup)")

        stirrups_req = {
            'bar_type': self.stirrup_bar_type,
            'pieces': [round(total_stirrup_perimeter, 2)] * total_stirrups,
            'distribution': dist_string,
            'total_quantity': total_stirrups
        }

        requirements = {
            'longitudinal': longitudinal_req,
            'stirrups': stirrups_req
        }

        logging.info(
            f"Calculated requirements for column '{self.name}': {longitudinal_req['description']}, {stirrups_req['total_quantity']} stirrups.")
        return requirements


class StockManager:
    """
    Manages the on-site inventory of rebar.
    """

    def __init__(self, initial_stock, standard_length=9.0):
        # initial_stock is a dict like {'1/2"': 10, '3/8"': 20}
        self.stock = initial_stock.copy()
        self.standard_length = standard_length  # meters

    def check_stock(self, bar_type):
        return self.stock.get(bar_type, 0)

    def use_stock(self, bar_type, quantity):
        if self.check_stock(bar_type) >= quantity:
            self.stock[bar_type] -= quantity
            return True
        return False


# --- Main Application ---

class ProjectSimulator(tk.Tk):
    """
    The main application class, handling the GUI and simulation logic.
    """

    def __init__(self):
        super().__init__()
        self.title("Simulador y Optimizador de Acero para Columnas")
        self.geometry("1200x900")

        # --- Styling ---
        self.style = ttk.Style(self)
        self.style.theme_use("clam")
        self.style.configure("TFrame", background="#f0f0f0")
        self.style.configure("TLabel", background="#f0f0f0", font=("Segoe UI", 10))
        self.style.configure("Header.TLabel", font=("Segoe UI", 14, "bold"))
        self.style.configure("TButton", font=("Segoe UI", 10, "bold"), padding=5)
        self.style.configure("Accent.TButton", foreground="white", background="#16a34a", relief="flat")
        self.style.map("Accent.TButton", background=[('active', '#15803d')])
        self.style.configure("TEntry", padding=5)

        # --- Data Initialization ---
        self.init_data()

        # --- UI Creation ---
        self.create_widgets()

    def init_data(self):
        """Initialize simulation data: column types and initial stock."""
        logging.info("Initializing simulation data.")
        self.column_types = {
            "C-1 (25x40)": Column("C-1", 25, 40, 3.0, 4, '1/2"', 6, '3/8"', 10, 20),
            "C-2 (30x50)": Column("C-2", 30, 50, 3.0, 4, '5/8"', 8, '3/8"', 10, 25),
            "C-3 (25x25)": Column("C-3", 25, 25, 3.0, 4, '1/2"', 4, '3/8"', 15, 25),
        }

        initial_stock = {'1/2"': 10, '5/8"': 5, '3/8"': 30}
        self.stock_manager = StockManager(initial_stock)

    def create_widgets(self):
        """Create and layout all the GUI elements."""
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill="both", expand=True)

        input_frame = ttk.LabelFrame(main_frame, text="Partida a Construir", padding="10")
        input_frame.pack(fill="x", pady=5)

        self.column_inputs = {}
        for i, (name, col) in enumerate(self.column_types.items()):
            ttk.Label(input_frame, text=f"Cantidad de Columnas '{name}':").grid(row=i, column=0, padx=5, pady=5,
                                                                                sticky="w")
            entry = ttk.Entry(input_frame, width=10)
            entry.grid(row=i, column=1, padx=5, pady=5)
            entry.insert(0, "0")
            self.column_inputs[name] = entry

        simulate_button = ttk.Button(input_frame, text="Optimizar Pedido y Corte", command=self.run_optimization,
                                     style="Accent.TButton")
        simulate_button.grid(row=len(self.column_types), columnspan=2, pady=10)

        output_notebook = ttk.Notebook(main_frame)
        output_notebook.pack(fill="both", expand=True, pady=10)

        self.results_texts = {}
        tabs = ["Resumen de Necesidades", "Plan de Corte Óptimo", "Pedido de Material (Cotización)",
                "Stock Actualizado"]
        for tab_name in tabs:
            tab = ttk.Frame(output_notebook, padding="5")
            output_notebook.add(tab, text=tab_name)
            text_area = tk.Text(tab, wrap="word", font=("Consolas", 10), height=10, relief="solid", borderwidth=1)
            text_area.pack(fill="both", expand=True)
            self.results_texts[tab_name] = text_area

        vis_tab = ttk.Frame(output_notebook, padding="10")
        output_notebook.add(vis_tab, text="Visualización Gráfica")

        vis_control_frame = ttk.Frame(vis_tab)
        vis_control_frame.pack(fill="x", pady=5)
        ttk.Label(vis_control_frame, text="Seleccionar Columna para Visualizar:").pack(side="left", padx=5)
        self.vis_selector = ttk.Combobox(vis_control_frame, values=list(self.column_types.keys()), state="readonly")
        self.vis_selector.pack(side="left", padx=5)
        if self.column_types:
            self.vis_selector.current(0)
        self.vis_selector.bind("<<ComboboxSelected>>", self.on_vis_selection_change)

        self.canvas_cross_section = tk.Canvas(vis_tab, bg="white", height=300)
        self.canvas_cross_section.pack(fill="x", pady=5, expand=False)

        self.canvas_elevation = tk.Canvas(vis_tab, bg="white", height=300)
        self.canvas_elevation.pack(fill="x", pady=5, expand=False)

        self.canvas_cutting_plan = tk.Canvas(vis_tab, bg="white")
        self.canvas_cutting_plan.pack(fill="both", pady=5, expand=True)

        self.last_cutting_plan = None

    def on_vis_selection_change(self, event=None):
        """Redraws visuals when the user selects a different column from the combobox."""
        selected_col_name = self.vis_selector.get()
        if selected_col_name and self.last_cutting_plan is not None:
            selected_column = self.column_types[selected_col_name]
            self.draw_cross_section(selected_column)
            self.draw_stirrup_distribution(selected_column)

    def run_optimization(self):
        """Main logic triggered by the button click."""
        logging.info("=" * 50)
        logging.info("Starting new optimization run.")

        for text_area in self.results_texts.values():
            text_area.delete("1.0", tk.END)
        self.canvas_cross_section.delete("all")
        self.canvas_elevation.delete("all")
        self.canvas_cutting_plan.delete("all")

        try:
            partida = {name: int(entry.get()) for name, entry in self.column_inputs.items()}
            if not any(partida.values()):
                messagebox.showwarning("Entrada Vacía",
                                       "Por favor, ingrese una cantidad para al menos un tipo de columna.")
                return
        except ValueError:
            messagebox.showerror("Error de Entrada", "Por favor, ingrese solo números enteros en las cantidades.")
            logging.error("Invalid input provided by user.")
            return

        detailed_requirements = {}
        flat_requirements = {}
        for name, quantity in partida.items():
            if quantity > 0:
                col_reqs = self.column_types[name].calculate_requirements()
                detailed_requirements[name] = (quantity, col_reqs)

                for part, details in col_reqs.items():
                    bar_type = details['bar_type']
                    if bar_type not in flat_requirements:
                        flat_requirements[bar_type] = []
                    flat_requirements[bar_type].extend(details['pieces'] * quantity)

        self.display_requirements(detailed_requirements)

        order_request, cutting_plan, updated_stock = self.generate_optimal_plans(flat_requirements)
        self.last_cutting_plan = cutting_plan

        self.display_cutting_plan(cutting_plan)
        self.display_order_request(order_request)
        self.display_updated_stock(updated_stock)

        self.on_vis_selection_change()
        self.draw_cutting_plan_visual(cutting_plan)

        logging.info("Optimization run completed successfully.")
        messagebox.showinfo("Optimización Completa",
                            "El análisis ha finalizado. Revise los resultados en las pestañas.")

    def generate_optimal_plans(self, requirements):
        """
        Generates cutting plan and order list using the PuLP optimizer.
        """
        cutting_plan = {}
        order_request = {}
        temp_stock_manager = StockManager(self.stock_manager.stock)
        stock_length_m = temp_stock_manager.standard_length

        for bar_type, pieces in requirements.items():
            if not pieces: continue

            # Convert demand list to demand dictionary {length: count}
            demand_dict = defaultdict(int)
            for p in pieces:
                demand_dict[p] += 1

            # --- PuLP Optimization ---
            status, total_bars, plan = self.solve_csp_with_ilp(stock_length_m, demand_dict)

            if status != 'Optimal':
                messagebox.showerror("Error de Optimización",
                                     f"No se encontró una solución óptima para barras {bar_type}. Estado: {status}")
                continue

            # Process the optimal plan
            order_request[bar_type] = int(total_bars)

            # Format the cutting plan for display
            formatted_plan = []
            bar_counter = 0
            for info in plan.values():
                for _ in range(info['cantidad']):
                    bar_counter += 1
                    waste = stock_length_m - sum(info['patron'])
                    formatted_plan.append({
                        'bar_num': bar_counter,
                        'cuts': info['patron'],
                        'waste_m': round(waste, 2)
                    })
            cutting_plan[bar_type] = formatted_plan

        # For this simulation, we assume all stock is consumed by the optimal plan
        # A more complex simulation would integrate stock into the ILP constraints
        final_stock = {bt: 0 for bt in self.stock_manager.stock.keys()}

        return order_request, cutting_plan, final_stock

    def solve_csp_with_ilp(self, stock_length, demand):
        """Solves the Cutting Stock Problem using Integer Linear Programming."""

        # 1. Generate viable cutting patterns
        unique_piece_lengths = list(demand.keys())
        patterns = self._generate_patterns(stock_length, unique_piece_lengths)
        if not patterns:
            return "No Patterns", None, None

        # 2. Set up the ILP model
        model = pulp.LpProblem("Optimizacion_Corte_Acero", pulp.LpMinimize)

        # Variables: x_i is the number of times pattern i is used
        x = pulp.LpVariable.dicts("Patron", range(len(patterns)), lowBound=0, cat='Integer')

        # Objective function: Minimize the total number of stock bars used
        model += pulp.lpSum(x[i] for i in range(len(patterns))), "Total_Barras_Utilizadas"

        # Constraints: Ensure demand for each piece length is met
        piece_counts_in_pattern = [defaultdict(int) for _ in patterns]
        for i, p in enumerate(patterns):
            for piece_len in p:
                piece_counts_in_pattern[i][piece_len] += 1

        for piece_len, required_qty in demand.items():
            model += pulp.lpSum(piece_counts_in_pattern[i][piece_len] * x[i] for i in
                                range(len(patterns))) >= required_qty, f"Demanda_Pieza_{piece_len}"

        # 3. Solve the model
        solver = pulp.PULP_CBC_CMD(msg=False)
        model.solve(solver)

        # 4. Extract results
        status = pulp.LpStatus[model.status]
        if status == 'Optimal':
            cutting_plan = {}
            for i in range(len(patterns)):
                num_times_used = pulp.value(x[i])
                if num_times_used > 0:
                    cutting_plan[i] = {
                        'patron': patterns[i],
                        'cantidad': int(num_times_used)
                    }
            total_bars = pulp.value(model.objective)
            return status, total_bars, cutting_plan
        else:
            return status, None, None

    def _generate_patterns(self, stock_length, piece_lengths):
        """Generates a set of possible cutting patterns."""
        # This is a simplified pattern generator. More sophisticated methods exist.
        patterns = []
        sorted_pieces = sorted(list(set(piece_lengths)), reverse=True)

        for piece in sorted_pieces:
            if piece > stock_length: continue
            max_qty = int(stock_length / piece)
            patterns.append([piece] * max_qty)

        for i in range(len(sorted_pieces)):
            current_pattern = []
            remaining_space = stock_length
            for j in range(i, len(sorted_pieces)):
                piece = sorted_pieces[j]
                while remaining_space >= piece:
                    current_pattern.append(piece)
                    remaining_space -= piece
            if current_pattern:
                patterns.append(current_pattern)

        unique_patterns_as_tuples = set(tuple(sorted(p)) for p in patterns)
        return [list(p) for p in unique_patterns_as_tuples]

    # --- Display Functions (Text) ---
    def display_requirements(self, detailed_requirements):
        text_area = self.results_texts["Resumen de Necesidades"]
        text_area.delete("1.0", tk.END)
        text_area.insert(tk.END, "RESUMEN DE NECESIDADES POR TIPO DE ELEMENTO\n")
        text_area.insert(tk.END, "=" * 60 + "\n\n")

        for name, (quantity, reqs) in detailed_requirements.items():
            text_area.insert(tk.END, f"ELEMENTO: {name} (Cantidad: {quantity})\n")
            text_area.insert(tk.END, "-" * 40 + "\n")

            long_req = reqs['longitudinal']
            text_area.insert(tk.END, f"  ACERO LONGITUDINAL ({long_req['bar_type']}):\n")
            text_area.insert(tk.END, f"    - {long_req['description']}\n")
            total_long_len = sum(long_req['pieces']) * quantity
            text_area.insert(tk.END, f"    - Longitud Total (para {quantity} col.): {total_long_len:.2f} m\n\n")

            stirrup_req = reqs['stirrups']
            text_area.insert(tk.END, f"  ACERO TRANSVERSAL ({stirrup_req['bar_type']}):\n")
            text_area.insert(tk.END, f"    - Cantidad por columna: {stirrup_req['total_quantity']} estribos\n")
            piece_len = stirrup_req['pieces'][0] if stirrup_req['pieces'] else 0
            text_area.insert(tk.END, f"    - Longitud por estribo: {piece_len:.2f} m\n")
            text_area.insert(tk.END, f"    - Distribución: {stirrup_req['distribution']}\n")
            total_stirrup_len = sum(stirrup_req['pieces']) * quantity
            text_area.insert(tk.END, f"    - Longitud Total (para {quantity} col.): {total_stirrup_len:.2f} m\n")
            text_area.insert(tk.END, "\n" + "=" * 60 + "\n\n")

    def display_cutting_plan(self, cutting_plan):
        text_area = self.results_texts["Plan de Corte Óptimo"]
        text_area.delete("1.0", tk.END)
        text_area.insert(tk.END, "PLAN DE CORTE ÓPTIMO (GENERADO CON ILP)\n")
        text_area.insert(tk.END, "=" * 50 + "\n\n")
        total_waste = 0
        if not cutting_plan:
            text_area.insert(tk.END, "No se requieren cortes.")
            return

        for bar_type, cuts in cutting_plan.items():
            text_area.insert(tk.END, f"Para barras tipo '{bar_type}' (de {self.stock_manager.standard_length}m):\n")
            # Consolidate identical patterns
            pattern_summary = defaultdict(int)
            for bar_cut_info in cuts:
                pattern_tuple = tuple(sorted(bar_cut_info['cuts']))
                pattern_summary[pattern_tuple] += 1
                total_waste += bar_cut_info['waste_m']

            for pattern, count in pattern_summary.items():
                waste_per_bar = self.stock_manager.standard_length - sum(pattern)
                text_area.insert(tk.END,
                                 f"  - Cortar {count} barras con el patrón: {list(pattern)} m. (Merma p/barra: {waste_per_bar:.2f} m)\n")

            text_area.insert(tk.END, "\n")
        text_area.insert(tk.END, f"\nMERMA TOTAL ÓPTIMA: {total_waste:.2f} m\n")

    def display_order_request(self, order_request):
        text_area = self.results_texts["Pedido de Material (Cotización)"]
        text_area.delete("1.0", tk.END)
        text_area.insert(tk.END, "PEDIDO DE MATERIAL NUEVO REQUERIDO (ÓPTIMO)\n")
        text_area.insert(tk.END, "=" * 50 + "\n\n")
        if not order_request:
            text_area.insert(tk.END, "No se necesita material nuevo.")
        else:
            for bar_type, quantity in order_request.items():
                text_area.insert(tk.END, f"- Tipo de Barra: {bar_type}\n")
                text_area.insert(tk.END,
                                 f"  - Cantidad a pedir (barras de {self.stock_manager.standard_length}m): {quantity}\n\n")

    def display_updated_stock(self, updated_stock):
        text_area = self.results_texts["Stock Actualizado"]
        text_area.delete("1.0", tk.END)
        text_area.insert(tk.END, "ESTADO DEL STOCK EN OBRA (POST-SIMULACIÓN)\n")
        text_area.insert(tk.END, "=" * 50 + "\n\n")
        text_area.insert(tk.END, "NOTA: El planificador óptimo asume que se compra todo el material nuevo.\n\n")
        for bar_type, quantity in updated_stock.items():
            text_area.insert(tk.END, f"- Tipo de Barra: {bar_type}: {quantity} barras restantes\n")

    # --- Display Functions (Visual) ---
    def draw_cross_section(self, column):
        canvas = self.canvas_cross_section
        canvas.delete("all")

        padding = 50
        canvas_w = canvas.winfo_width()
        canvas_h = canvas.winfo_height()
        if canvas_w < 50 or canvas_h < 50: return

        scale = min((canvas_w - 2 * padding) / column.length, (canvas_h - 2 * padding) / column.width)

        offset_x = (canvas_w - column.length * scale) / 2
        offset_y = (canvas_h - column.width * scale) / 2

        x1, y1 = offset_x, offset_y
        x2, y2 = offset_x + column.length * scale, offset_y + column.width * scale
        canvas.create_rectangle(x1, y1, x2, y2, fill="#d3d3d3", outline="black")

        stirrup_x1, stirrup_y1 = x1 + column.cover * scale, y1 + column.cover * scale
        stirrup_x2, stirrup_y2 = x2 - column.cover * scale, y2 - column.cover * scale
        canvas.create_rectangle(stirrup_x1, stirrup_y1, stirrup_x2, stirrup_y2, outline="black", width=2)

        bar_radius = 5
        if column.long_bar_qty > 0:
            # A more realistic bar placement
            nx = max(2, round(column.long_bar_qty / 2 / (column.length / (column.length + column.width))))
            ny = math.ceil(column.long_bar_qty / 2) - nx + 2

            x_coords = [stirrup_x1, stirrup_x2]
            if nx > 2:
                x_spacing = (stirrup_x2 - stirrup_x1) / (nx - 1)
                x_coords.extend([stirrup_x1 + i * x_spacing for i in range(1, nx - 1)])

            y_coords = [stirrup_y1, stirrup_y2]
            if ny > 2:
                y_spacing = (stirrup_y2 - stirrup_y1) / (ny - 1)
                y_coords.extend([stirrup_y1 + i * y_spacing for i in range(1, ny - 1)])

            bar_positions = []
            for x_ in sorted(list(set(x_coords))): bar_positions.append((x_, stirrup_y1)); bar_positions.append(
                (x_, stirrup_y2))
            for y_ in sorted(list(set(y_coords)))[1:-1]: bar_positions.append((stirrup_x1, y_)); bar_positions.append(
                (stirrup_x2, y_))

            for (bar_x, bar_y) in bar_positions[:column.long_bar_qty]:
                canvas.create_oval(bar_x - bar_radius, bar_y - bar_radius, bar_x + bar_radius, bar_y + bar_radius,
                                   fill="black")

        # --- Add Dimensions ---
        canvas.create_line(x1, y1 - 20, x2, y1 - 20, arrow=tk.BOTH)
        canvas.create_text((x1 + x2) / 2, y1 - 30, text=f"{column.length} cm", anchor="s")
        canvas.create_line(x1 - 20, y1, x1 - 20, y2, arrow=tk.BOTH)
        canvas.create_text(x1 - 30, (y1 + y2) / 2, text=f"{column.width} cm", angle=90, anchor="s")

        canvas.create_text(canvas_w / 2, 10, anchor="n", text=f"Corte Transversal: {column.name}",
                           font=("Segoe UI", 10, "bold"))

    def draw_stirrup_distribution(self, column):
        canvas = self.canvas_elevation
        canvas.delete("all")
        padding = 40
        canvas_w = canvas.winfo_width()
        canvas_h = canvas.winfo_height()
        if canvas_w < 50 or canvas_h < 50: return

        col_draw_w = 100
        scale_h = (canvas_h - 2 * padding) / (column.height * 100)
        offset_x = (canvas_w - col_draw_w) / 2

        x1, y1 = offset_x, padding
        x2, y2 = offset_x + col_draw_w, padding + column.height * 100 * scale_h
        canvas.create_rectangle(x1, y1, x2, y2, fill="#d3d3d3", outline="black")

        y_bottom_limit = y1 + column.confined_length * scale_h
        y_top_limit = y2 - column.confined_length * scale_h

        # Draw stirrups
        # First stirrup at 5cm
        canvas.create_line(x1, y1 + 5 * scale_h, x2, y1 + 5 * scale_h, width=2)
        # Bottom confined
        current_y = y1 + 5 * scale_h + column.stirrup_spacing_confined * scale_h
        while current_y < y_bottom_limit:
            canvas.create_line(x1, current_y, x2, current_y, width=2)
            current_y += column.stirrup_spacing_confined * scale_h

        # Central zone
        current_y = y_bottom_limit + (column.stirrup_spacing_central * scale_h) / 2
        while current_y < y_top_limit:
            canvas.create_line(x1, current_y, x2, current_y, width=2)
            current_y += column.stirrup_spacing_central * scale_h

        # Top confined
        current_y = y2 - 5 * scale_h - column.stirrup_spacing_confined * scale_h
        while current_y > y_top_limit:
            canvas.create_line(x1, current_y, x2, current_y, width=2)
            current_y -= column.stirrup_spacing_confined * scale_h
        # Last stirrup at 5cm from top
        canvas.create_line(x1, y2 - 5 * scale_h, x2, y2 - 5 * scale_h, width=2)

        # --- Add Dimensions ---
        dim_x = x2 + 30
        # Total Height
        canvas.create_line(dim_x, y1, dim_x, y2, arrow=tk.BOTH)
        canvas.create_text(dim_x + 10, (y1 + y2) / 2, text=f"H = {column.height}m", angle=90, anchor="s")
        # Confined zones
        canvas.create_line(dim_x - 15, y1, dim_x - 15, y_bottom_limit, arrow=tk.BOTH)
        canvas.create_text(dim_x - 20, (y1 + y_bottom_limit) / 2,
                           text=f"{column.confined_length:.0f}cm\n@{column.stirrup_spacing_confined}cm", anchor="e",
                           justify="right")
        canvas.create_line(dim_x - 15, y2, dim_x - 15, y_top_limit, arrow=tk.BOTH)
        canvas.create_text(dim_x - 20, (y2 + y_top_limit) / 2,
                           text=f"{column.confined_length:.0f}cm\n@{column.stirrup_spacing_confined}cm", anchor="e",
                           justify="right")
        # Central zone
        canvas.create_line(dim_x, y_bottom_limit, dim_x, y_top_limit, arrow=tk.BOTH)
        canvas.create_text(dim_x + 10, (y_bottom_limit + y_top_limit) / 2, text=f"@{column.stirrup_spacing_central}cm",
                           angle=90, anchor="s")

        canvas.create_text(canvas_w / 2, 10, anchor="n", text=f"Elevación: Distribución de Estribos ({column.name})",
                           font=("Segoe UI", 10, "bold"))

    def draw_cutting_plan_visual(self, cutting_plan):
        canvas = self.canvas_cutting_plan
        canvas.delete("all")

        y_pos = 20
        padding = 20
        bar_height = 30

        canvas_w = canvas.winfo_width()
        if canvas_w < 50: return

        scale = (canvas_w - 2 * padding) / self.stock_manager.standard_length

        canvas.create_text(canvas_w / 2, y_pos, anchor="n",
                           text=f"Plan de Corte Visual Óptimo (Barras de {self.stock_manager.standard_length}m)",
                           font=("Segoe UI", 10, "bold"))
        y_pos += 30

        colors = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f", "#edc948", "#b07aa1", "#ff9da7", "#9c755f",
                  "#bab0ac"]

        if not cutting_plan:
            canvas.create_text(canvas_w / 2, y_pos + 50, text="No se requieren cortes.", anchor="n",
                               font=("Segoe UI", 12, "italic"))
            return

        # Consolidate for visual representation
        visual_plan = defaultdict(lambda: defaultdict(int))
        for bar_type, cuts_per_type in cutting_plan.items():
            for bar_info in cuts_per_type:
                pattern_tuple = tuple(sorted(bar_info['cuts']))
                visual_plan[bar_type][pattern_tuple] += 1

        for bar_type, pattern_summary in visual_plan.items():
            canvas.create_text(padding, y_pos, anchor="nw", text=f"Tipo de Barra: {bar_type}",
                               font=("Segoe UI", 10, "bold"))
            y_pos += 25

            for pattern_tuple, count in pattern_summary.items():

                waste = self.stock_manager.standard_length - sum(pattern_tuple)
                label = f"Cortar {count}x con patrón:"
                canvas.create_text(padding, y_pos + bar_height / 2, text=label, anchor="w")

                bar_start_x = padding + 150
                canvas_scale = (canvas_w - bar_start_x - padding) / self.stock_manager.standard_length

                # Draw stock bar
                canvas.create_rectangle(bar_start_x, y_pos,
                                        bar_start_x + self.stock_manager.standard_length * canvas_scale,
                                        y_pos + bar_height, outline="black", fill="#eeeeee")

                current_x = bar_start_x
                for i, piece_len in enumerate(pattern_tuple):
                    color = colors[i % len(colors)]
                    piece_width = piece_len * canvas_scale
                    canvas.create_rectangle(current_x, y_pos, current_x + piece_width, y_pos + bar_height, fill=color,
                                            outline="white")
                    if piece_width > 25:
                        canvas.create_text(current_x + piece_width / 2, y_pos + bar_height / 2, text=f"{piece_len}m",
                                           fill="white", font=("Segoe UI", 8, "bold"))
                    current_x += piece_width

                if waste > 0.01:
                    waste_width = waste * canvas_scale
                    canvas.create_rectangle(current_x, y_pos, current_x + waste_width, y_pos + bar_height,
                                            fill="#333333", outline="white", stipple="gray50")
                    if waste_width > 30:
                        canvas.create_text(current_x + waste_width / 2, y_pos + bar_height / 2,
                                           text=f"Merma\n{waste:.2f}m", fill="white", justify="center",
                                           font=("Segoe UI", 8))

                y_pos += bar_height + 10
            y_pos += 20


if __name__ == "__main__":
    app = ProjectSimulator()
    app.mainloop()

