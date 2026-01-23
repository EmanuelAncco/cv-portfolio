# ==================================================================================================
# == SIMULADOR Y OPTIMIZADOR AVANZADO DE ACERO PARA EDIFICACIONES
# == Enfoque: Lógica de Construcción Detallada + Optimización ILP Global + GUI Avanzada
# == Versión: 5.0 (Visualización de Traslapes y Acotación Profesional)
# == Autor: Dr. Consultor en Robótica para Construcción
# == Descripción:
# == Esta versión mejora drásticamente la visualización de ingeniería. Se añade la
# == representación gráfica de los empalmes por traslape en la elevación de las columnas
# == y se rediseña el acotado de estribos para emular un plano estructural, mejorando
# == la claridad y el realismo del simulador.
# ==================================================================================================

import tkinter as tk
from tkinter import ttk, messagebox, font, Canvas
import math
import logging
from datetime import datetime
from collections import defaultdict

# --- Motor de Optimización ---
import pulp

# --- Configuración de Logging ---
LOG_FILE = f"simulador_acero_avanzado_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(LOG_FILE),
                        logging.StreamHandler()
                    ])

# --- Constantes de Ingeniería (Simplificadas) ---
SPLICE_LENGTHS = {  # Longitudes de traslape típicas en metros
    '1/2"': 1.10,
    '5/8"': 1.30,
    '3/4"': 1.60,
    '1"': 2.10,
}


# --- Clases de Dominio (Ingeniería Civil) ---

class Column:
    """
    Representa un tipo de columna de concreto armado, conteniendo sus parámetros de diseño.
    """

    def __init__(self, name, width, length, height, cover,
                 long_bar_type, long_bar_qty_x, long_bar_qty_y,
                 stirrup_bar_type, stirrup_spacing_confined,
                 stirrup_spacing_central, supplementary_stirrups_qty=0, confined_length_factor=1 / 6):
        self.name = name
        self.width = float(width)
        self.length = float(length)
        self.height = float(height)
        self.cover = float(cover)
        self.long_bar_type = long_bar_type
        self.long_bar_qty_x = int(long_bar_qty_x)
        self.long_bar_qty_y = int(long_bar_qty_y)
        self.long_bar_qty = self.long_bar_qty_x * 2 + (
                    self.long_bar_qty_y - 2) * 2 if self.long_bar_qty_y > 1 else self.long_bar_qty_x * 2
        self.stirrup_bar_type = stirrup_bar_type
        self.stirrup_spacing_confined = float(stirrup_spacing_confined)
        self.stirrup_spacing_central = float(stirrup_spacing_central)
        self.supplementary_stirrups_qty = int(supplementary_stirrups_qty)
        self.confined_length = self.height * confined_length_factor * 100

    def calculate_requirements(self):
        requirements = {}
        footing_anchor_length = 0.8
        splice_length = SPLICE_LENGTHS.get(self.long_bar_type, 1.20)
        has_splices = self.height > (9.0 - footing_anchor_length)

        if has_splices:
            main_piece_len = 9.0 - splice_length / 2
            splice_piece_len = self.height - main_piece_len + footing_anchor_length + splice_length / 2
            pieces = [round(main_piece_len, 2), round(splice_piece_len, 2)] * self.long_bar_qty
            description = f"{self.long_bar_qty * 2} piezas para empalme (long. aprox. {main_piece_len:.2f}m y {splice_piece_len:.2f}m)"
        else:
            long_bar_length = self.height + footing_anchor_length
            pieces = [round(long_bar_length, 2)] * self.long_bar_qty
            description = f"{self.long_bar_qty} barras de {long_bar_length:.2f}m c/u"

        requirements['longitudinal'] = {
            'bar_type': self.long_bar_type, 'pieces': pieces, 'description': description,
            'has_splices': has_splices, 'splice_length': splice_length
        }

        stirrup_width = self.width - 2 * self.cover
        stirrup_length = self.length - 2 * self.cover
        hook_length = 10 * 2
        total_stirrup_perimeter = (2 * stirrup_width + 2 * stirrup_length + hook_length) / 100
        qty_confined_zone = math.ceil(self.confined_length / self.stirrup_spacing_confined)
        qty_confined_total = qty_confined_zone * 2
        central_length = (self.height * 100) - (2 * self.confined_length)
        qty_central = math.ceil(central_length / self.stirrup_spacing_central) if central_length > 0 else 0
        total_stirrups = qty_confined_total + qty_central
        requirements['stirrups'] = {
            'bar_type': self.stirrup_bar_type, 'pieces': [round(total_stirrup_perimeter, 2)] * total_stirrups,
            'distribution': f"2 zonas confinadas ({qty_confined_total} est. @ {self.stirrup_spacing_confined}cm) y 1 zona central ({qty_central} est. @ {self.stirrup_spacing_central}cm)",
            'total_quantity': total_stirrups, 'qty_confined_zone': qty_confined_zone, 'qty_central': qty_central
        }

        if self.supplementary_stirrups_qty > 0:
            hook_sup_len = (self.width - 2 * self.cover) + 2 * (hook_length / 2)
            total_hooks = total_stirrups * self.supplementary_stirrups_qty
            requirements['supplementary_hooks'] = {
                'bar_type': self.stirrup_bar_type, 'pieces': [round(hook_sup_len / 100, 2)] * total_hooks,
                'description': f"{self.supplementary_stirrups_qty} gancho(s) por nivel de estribo.",
                'total_quantity': total_hooks
            }

        return requirements


# --- Aplicación Principal de la GUI ---

class AdvancedSteelSimulator(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Simulador y Optimizador Avanzado de Acero para Proyectos v5.0")
        self.geometry("1600x950")
        self.style = ttk.Style(self)
        self.style.theme_use("clam")
        self.configure_styles()
        self.piece_colors = ['#60a5fa', '#facc15', '#4ade80', '#f87171', '#fb923c', '#818cf8', '#a78bfa', '#e879f9',
                             '#22d3ee']
        self.color_map = {}
        self.last_cutting_plan = None
        self.init_project_data()
        self.create_widgets()

    def configure_styles(self):
        self.style.configure("TFrame", background="#f1f5f9")
        self.style.configure("TLabel", background="#f1f5f9", font=("Segoe UI", 10))
        self.style.configure("Title.TLabel", font=("Segoe UI", 12, "bold"), foreground="#0f172a")
        self.style.configure("TButton", font=("Segoe UI", 10, "bold"), padding=6, relief="flat")
        self.style.configure("Accent.TButton", foreground="white", background="#16a34a")
        self.style.map("Accent.TButton", background=[('active', '#15803d')])
        self.style.configure("TEntry", padding=5, fieldbackground="#ffffff")
        self.style.configure("TLabelframe", background="#f1f5f9", bordercolor="#cbd5e1", relief="solid")
        self.style.configure("TLabelframe.Label", background="#f1f5f9", font=("Segoe UI", 11, "bold"),
                             foreground="#334155")
        self.style.configure("TNotebook.Tab", font=("Segoe UI", 10, "bold"), padding=[10, 5])
        self.style.map("TNotebook.Tab", background=[("selected", "#e2e8f0")], foreground=[("selected", "#0f172a")])

    def init_project_data(self):
        logging.info("Inicializando datos de simulación del proyecto.")
        self.column_types = {
            "C-1 (25x40)": Column("C-1", 25, 40, 3.0, 4, '1/2"', 3, 2, '3/8"', 10, 20),
            "C-2 (30x50)": Column("C-2", 30, 50, 3.0, 4, '5/8"', 4, 2, '3/8"', 10, 25, supplementary_stirrups_qty=1),
            "C-3 (25x25)": Column("C-3", 25, 25, 3.0, 4, '1/2"', 2, 2, '3/8"', 15, 25),
            "C-4 (60x30)": Column("C-4", 60, 30, 3.2, 5, '5/8"', 5, 2, '3/8"', 10, 20, supplementary_stirrups_qty=2),
            "C-5 (40x40)": Column("C-5", 40, 40, 3.2, 5, '1/2"', 3, 3, '3/8"', 10, 15, supplementary_stirrups_qty=1),
            "C-6 (30x70)": Column("C-6", 30, 70, 10.5, 5, '5/8"', 6, 2, '3/8"', 10, 15, supplementary_stirrups_qty=3)
        }
        self.initial_stock = {'1/2"': 15, '5/8"': 10, '3/8"': 40}

    def create_widgets(self):
        main_frame = ttk.Frame(self, padding="10");
        main_frame.pack(fill="both", expand=True)
        input_panel = ttk.LabelFrame(main_frame, text="Datos del Proyecto", padding="15");
        input_panel.pack(side="left", fill="y", padx=(0, 10))
        ttk.Label(input_panel, text="Metrado de Columnas:", style="Title.TLabel").pack(anchor="w", pady=(0, 5))
        self.column_inputs = {}
        for name in self.column_types.keys():
            frame = ttk.Frame(input_panel);
            frame.pack(fill="x", pady=2)
            ttk.Label(frame, text=name, width=15).pack(side="left")
            entry = ttk.Entry(frame, width=8);
            entry.pack(side="left", padx=5);
            entry.insert(0, "0")
            self.column_inputs[name] = entry
        ttk.Separator(input_panel, orient="horizontal").pack(fill="x", pady=15)
        ttk.Label(input_panel, text="Stock Actual en Obra:", style="Title.TLabel").pack(anchor="w", pady=(0, 5))
        self.stock_inputs = {}
        for bar_type, qty in self.initial_stock.items():
            frame = ttk.Frame(input_panel);
            frame.pack(fill="x", pady=2)
            ttk.Label(frame, text=f"Barras {bar_type}:", width=15).pack(side="left")
            entry = ttk.Entry(frame, width=8);
            entry.pack(side="left", padx=5);
            entry.insert(0, str(qty))
            self.stock_inputs[bar_type] = entry
        run_button = ttk.Button(input_panel, text="Optimizar Proyecto", command=self.run_optimization,
                                style="Accent.TButton")
        run_button.pack(fill="x", pady=(20, 0), ipady=5)
        output_panel = ttk.Frame(main_frame);
        output_panel.pack(side="left", fill="both", expand=True)
        self.summary_frame = ttk.Frame(output_panel, padding=5);
        self.summary_frame.pack(fill="x")
        notebook = ttk.Notebook(output_panel);
        notebook.pack(fill="both", expand=True)
        vis_tab = ttk.Frame(notebook);
        notebook.add(vis_tab, text="Plan de Corte Visual")
        self.canvas_cutting_plan = Canvas(vis_tab, bg="white", highlightthickness=0)
        v_scroll = ttk.Scrollbar(vis_tab, orient="vertical", command=self.canvas_cutting_plan.yview)
        self.canvas_cutting_plan.config(yscrollcommand=v_scroll.set);
        v_scroll.pack(side="right", fill="y")
        self.canvas_cutting_plan.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        self.cutting_plan_container = ttk.Frame(self.canvas_cutting_plan)
        self.canvas_cutting_plan.create_window((0, 0), window=self.cutting_plan_container, anchor="nw")
        self.cutting_plan_container.bind("<Configure>", lambda e: self.canvas_cutting_plan.configure(
            scrollregion=self.canvas_cutting_plan.bbox("all")))
        bom_tab = ttk.Frame(notebook, padding=5);
        notebook.add(bom_tab, text="Despiece por Elemento")
        self.bom_text = tk.Text(bom_tab, wrap="word", font=("Consolas", 10), relief="solid", borderwidth=1)
        self.bom_text.pack(fill="both", expand=True)
        detail_tab = ttk.Frame(notebook, padding=10);
        notebook.add(detail_tab, text="Detalle de Columnas")
        vis_control_frame = ttk.Frame(detail_tab);
        vis_control_frame.pack(fill="x", pady=5)
        ttk.Label(vis_control_frame, text="Seleccionar Columna:").pack(side="left")
        self.vis_selector = ttk.Combobox(vis_control_frame, values=list(self.column_types.keys()), state="readonly")
        self.vis_selector.pack(side="left", padx=5)
        if self.column_types: self.vis_selector.current(0)
        self.vis_selector.bind("<<ComboboxSelected>>", self.update_detail_visuals)
        self.canvas_cross_section = Canvas(detail_tab, bg="white", height=350);
        self.canvas_cross_section.pack(fill="x", pady=10)
        self.canvas_elevation = Canvas(detail_tab, bg="white", height=400);
        self.canvas_elevation.pack(fill="x", pady=10)
        log_tab = ttk.Frame(notebook, padding=5);
        notebook.add(log_tab, text="Consola de Simulación")
        self.log_text = tk.Text(log_tab, wrap="word", font=("Consolas", 10), relief="solid", borderwidth=1,
                                state="disabled")
        self.log_text.pack(fill="both", expand=True)

    def log_message(self, message):
        logging.info(message)
        self.log_text.config(state="normal");
        self.log_text.insert(tk.END, f"[{datetime.now().strftime('%H:%M:%S')}] {message}\n")
        self.log_text.see(tk.END);
        self.log_text.config(state="disabled");
        self.update_idletasks()

    def run_optimization(self):
        self.log_text.config(state="normal");
        self.log_text.delete("1.0", tk.END);
        self.log_text.config(state="disabled")
        self.log_message("=" * 50);
        self.log_message("INICIANDO NUEVA SIMULACIÓN DE PROYECTO.")
        try:
            partida = {name: int(entry.get()) for name, entry in self.column_inputs.items()}
            current_stock = {name: int(entry.get()) for name, entry in self.stock_inputs.items()}
            if not any(partida.values()):
                messagebox.showwarning("Entrada Vacía", "Ingrese la cantidad para al menos un tipo de columna.");
                self.log_message("ERROR: No se ingresaron cantidades.");
                return
        except ValueError:
            messagebox.showerror("Error de Entrada", "Ingrese solo números enteros.");
            self.log_message("ERROR: Entrada inválida.");
            return
        self.log_message("Calculando despiece detallado para cada tipo de elemento...")
        detailed_reqs_per_type = {name: col.calculate_requirements() for name, col in self.column_types.items()}
        self.display_bill_of_materials(partida, detailed_reqs_per_type)
        self.log_message("Consolidando requerimientos totales de acero para el proyecto...")
        total_project_requirements = defaultdict(list)
        for name, quantity in partida.items():
            if quantity > 0:
                for req_type, details in detailed_reqs_per_type[name].items():
                    total_project_requirements[details['bar_type']].extend(details['pieces'] * quantity)
        self.log_message("Requerimientos consolidados por diámetro de barra.")
        self.log_message("Iniciando motor de optimización ILP para generar planes de corte...")
        order_request, cutting_plan, final_stock = self.generate_optimal_plans(total_project_requirements,
                                                                               current_stock)
        self.last_cutting_plan = cutting_plan
        self.log_message("Optimización completada.")
        self.log_message("Generando visualizaciones y resúmenes...")
        self.display_summary(order_request, cutting_plan)
        self.draw_cutting_plan_visual(cutting_plan, total_project_requirements)
        self.update_detail_visuals()
        self.log_message("Simulación finalizada con éxito.")
        messagebox.showinfo("Optimización Completa", "El análisis ha finalizado. Revise los resultados.")

    def display_bill_of_materials(self, partida, detailed_reqs):
        self.bom_text.delete("1.0", tk.END)
        self.bom_text.tag_configure("header", font=("Consolas", 12, "bold", "underline"))
        self.bom_text.tag_configure("subheader", font=("Consolas", 10, "bold"))
        for name, quantity in partida.items():
            if quantity == 0: continue
            self.bom_text.insert(tk.END, f"ELEMENTO: {name} (CANTIDAD: {quantity})\n", "header")
            reqs = detailed_reqs[name]
            long_req = reqs['longitudinal']
            self.bom_text.insert(tk.END, f"\n  ACERO LONGITUDINAL ({long_req['bar_type']}):\n", "subheader")
            self.bom_text.insert(tk.END, f"    - Detalle: {long_req['description']}\n")
            if long_req['has_splices']: self.bom_text.insert(tk.END,
                                                             f"    - NOTA: Requiere empalme por traslape de {long_req['splice_length']:.2f} m.\n")
            stirrup_req = reqs['stirrups']
            self.bom_text.insert(tk.END, f"\n  ESTRIBOS ({stirrup_req['bar_type']}):\n", "subheader")
            piece_len = stirrup_req['pieces'][0] if stirrup_req['pieces'] else 0
            self.bom_text.insert(tk.END,
                                 f"    - Cant. por Columna: {stirrup_req['total_quantity']} estribos de {piece_len:.2f} m\n")
            self.bom_text.insert(tk.END, f"    - Distribución: {stirrup_req['distribution']}\n")
            if 'supplementary_hooks' in reqs:
                hook_req = reqs['supplementary_hooks']
                self.bom_text.insert(tk.END, f"\n  GANCHOS SUPLEMENTARIOS ({hook_req['bar_type']}):\n", "subheader")
                piece_len_h = hook_req['pieces'][0] if hook_req['pieces'] else 0
                self.bom_text.insert(tk.END,
                                     f"    - Cant. por Columna: {hook_req['total_quantity']} ganchos de {piece_len_h:.2f} m\n")
                self.bom_text.insert(tk.END, f"    - Detalle: {hook_req['description']}\n")
            self.bom_text.insert(tk.END, "=" * 70 + "\n\n")

    def generate_optimal_plans(self, requirements, current_stock):
        cutting_plan, order_request, stock_length_m = {}, {}, 9.0
        for bar_type, pieces in requirements.items():
            if not pieces: continue
            self.log_message(f"Optimizando cortes para barras de {bar_type}...")
            demand_dict = defaultdict(int)
            for p in pieces: demand_dict[p] += 1
            status, total_bars_needed, plan = self.solve_csp_with_ilp(stock_length_m, demand_dict)
            self.log_message(f"  -> Estado del Solver para {bar_type}: {status}")
            if status != 'Optimal': messagebox.showerror("Error de Optimización",
                                                         f"No se encontró solución para {bar_type}. Estado: {status}"); continue
            order_request[bar_type] = max(0, math.ceil(total_bars_needed) - current_stock.get(bar_type, 0))
            formatted_plan = []
            for info in plan.values():
                for _ in range(info['cantidad']):
                    formatted_plan.append(
                        {'cuts': info['patron'], 'waste_m': round(stock_length_m - sum(info['patron']), 2)})
            cutting_plan[bar_type] = formatted_plan
        final_stock = current_stock.copy()
        for bar_type, plan_cuts in cutting_plan.items():
            final_stock[bar_type] = current_stock.get(bar_type, 0) - min(current_stock.get(bar_type, 0), len(plan_cuts))
        return order_request, cutting_plan, final_stock

    def solve_csp_with_ilp(self, stock_length, demand):
        unique_piece_lengths = list(demand.keys())
        patterns = self._generate_patterns(stock_length, unique_piece_lengths)
        if not patterns: return "No Patterns", None, None
        model = pulp.LpProblem("Optimizacion_Corte_Acero", pulp.LpMinimize)
        x = pulp.LpVariable.dicts("Patron", range(len(patterns)), lowBound=0, cat='Integer')
        model += pulp.lpSum(x[i] for i in range(len(patterns))), "Total_Barras_Utilizadas"
        piece_counts_in_pattern = [defaultdict(int) for _ in patterns]
        for i, p in enumerate(patterns):
            for piece_len in p: piece_counts_in_pattern[i][piece_len] += 1
        for piece_len, required_qty in demand.items():
            model += pulp.lpSum(piece_counts_in_pattern[i][piece_len] * x[i] for i in
                                range(len(patterns))) >= required_qty, f"Demanda_Pieza_{piece_len}"
        model.solve(pulp.PULP_CBC_CMD(msg=False))
        status = pulp.LpStatus[model.status]
        if status == 'Optimal':
            plan = {i: {'patron': patterns[i], 'cantidad': int(round(pulp.value(x[i])))} for i in range(len(patterns))
                    if pulp.value(x[i]) > 0.1}
            return status, pulp.value(model.objective), plan
        return status, None, None

    def _generate_patterns(self, stock_length, piece_lengths):
        patterns = []
        sorted_pieces = sorted(list(set(p for p in piece_lengths if p <= stock_length)), reverse=True)
        if not sorted_pieces: return []
        for piece in sorted_pieces: patterns.append([piece] * int(stock_length / piece))
        for i in range(len(sorted_pieces)):
            current_pattern, remaining_space = [], stock_length
            for j in range(i, len(sorted_pieces)):
                piece = sorted_pieces[j]
                while remaining_space >= piece: current_pattern.append(piece); remaining_space -= piece
            if current_pattern: patterns.append(current_pattern)
        return [list(p) for p in set(tuple(sorted(p)) for p in patterns)]

    def get_color_for_piece(self, length):
        if length not in self.color_map: self.color_map[length] = self.piece_colors[
            len(self.color_map) % len(self.piece_colors)]
        return self.color_map[length]

    def display_summary(self, order_request, cutting_plan):
        for widget in self.summary_frame.winfo_children(): widget.destroy()
        total_waste = sum(bar['waste_m'] for plans in cutting_plan.values() for bar in plans)
        total_bars_used = sum(len(plans) for plans in cutting_plan.values())
        efficiency = (1 - total_waste / (total_bars_used * 9.0)) * 100 if total_bars_used > 0 else 100
        ttk.Label(self.summary_frame, text=f"Barras a Utilizar: {total_bars_used}", font=("Segoe UI", 12, "bold"),
                  foreground="#3b82f6").pack(side="left", padx=10)
        ttk.Label(self.summary_frame, text=f"Merma Total Óptima: {total_waste:.2f} m", font=("Segoe UI", 12, "bold"),
                  foreground="#ef4444").pack(side="left", padx=10)
        ttk.Label(self.summary_frame, text=f"Eficiencia del Plan: {efficiency:.2f}%", font=("Segoe UI", 12, "bold"),
                  foreground="#16a34a").pack(side="left", padx=10)

    def draw_cutting_plan_visual(self, cutting_plan, total_project_requirements):
        for widget in self.cutting_plan_container.winfo_children(): widget.destroy()
        self.color_map.clear()
        if not cutting_plan: return
        for bar_type, pattern_summary_list in cutting_plan.items():
            ttk.Label(self.cutting_plan_container, text=f"PLAN DE CORTE PARA BARRAS DE {bar_type}",
                      style="Title.TLabel").pack(anchor="w", pady=(15, 2))
            demand_summary_frame = ttk.Frame(self.cutting_plan_container, padding=5);
            demand_summary_frame.pack(fill="x", pady=(0, 10))
            ttk.Label(demand_summary_frame, text="Demanda a Cubrir:", font=("Segoe UI", 10, "bold")).pack(anchor="w")
            demand_for_type = defaultdict(int)
            for piece in total_project_requirements.get(bar_type, []): demand_for_type[round(piece, 2)] += 1
            row_frame = ttk.Frame(demand_summary_frame);
            row_frame.pack(fill="x")
            for length, qty in sorted(demand_for_type.items()):
                color = self.get_color_for_piece(length)
                item_frame = ttk.Frame(row_frame);
                item_frame.pack(side="left", padx=10, pady=2)
                tk.Label(item_frame, bg=color, width=2, height=1).pack(side="left")
                ttk.Label(item_frame, text=f"{qty} x {length:.2f}m").pack(side="left", padx=5)
            visual_plan = defaultdict(int)
            for bar_info in pattern_summary_list: visual_plan[tuple(sorted(bar_info['cuts']))] += 1
            for pattern, count in visual_plan.items(): self.draw_single_pattern_bar(self.cutting_plan_container,
                                                                                    pattern, count)

    def draw_single_pattern_bar(self, parent, pattern, count):
        bar_height, stock_length = 35, 9.0
        frame = ttk.Frame(parent);
        frame.pack(fill="x", pady=4, padx=5)
        ttk.Label(frame, text=f"Cortar {count}x:", font=("Segoe UI", 10, "bold")).pack(side="left", padx=(0, 10),
                                                                                       anchor="w")
        bar_canvas = Canvas(frame, height=bar_height, bg="#e5e7eb", highlightthickness=0)
        bar_canvas.pack(side="left", expand=True, fill="x")
        self.after(100, lambda c=bar_canvas, p=pattern: self._draw_bar_pieces(c, p, stock_length))

    def _draw_bar_pieces(self, canvas, pattern, stock_length):
        canvas_width = canvas.winfo_width()
        if canvas_width < 20: return
        current_x = 0
        for piece in pattern:
            color = self.get_color_for_piece(piece)
            piece_width = (piece / stock_length) * canvas_width
            canvas.create_rectangle(current_x, 0, current_x + piece_width, 35, fill=color, outline="#f1f5f9", width=2)
            if piece_width > 40: canvas.create_text(current_x + piece_width / 2, 17.5, text=f"{piece:.2f}",
                                                    fill="white", font=("Segoe UI", 8, "bold"))
            current_x += piece_width
        waste = stock_length - sum(pattern)
        if waste > 0.01: canvas.create_rectangle(current_x, 0, canvas_width, 35, fill="#94a3b8", outline="#f1f5f9",
                                                 stipple="gray25")

    def update_detail_visuals(self, event=None):
        selected_col_name = self.vis_selector.get()
        if not selected_col_name: return
        column = self.column_types[selected_col_name]
        self.draw_cross_section(column)
        self.draw_stirrup_distribution(column)

    def draw_cross_section(self, column):
        canvas = self.canvas_cross_section;
        canvas.delete("all")
        padding, bar_radius = 50, 7
        canvas_w, canvas_h = canvas.winfo_width(), canvas.winfo_height()
        if canvas_w < 50: self.after(100, lambda: self.draw_cross_section(column)); return
        scale = min((canvas_w - 2 * padding) / column.length, (canvas_h - 2 * padding) / column.width)
        offset_x = (canvas_w - column.length * scale) / 2
        offset_y = (canvas_h - column.width * scale) / 2
        x1, y1 = offset_x, offset_y
        x2, y2 = offset_x + column.length * scale, offset_y + column.width * scale
        canvas.create_rectangle(x1, y1, x2, y2, fill="#e2e8f0", outline="#475569")
        sx1, sy1 = x1 + column.cover * scale, y1 + column.cover * scale
        sx2, sy2 = x2 - column.cover * scale, y2 - column.cover * scale
        canvas.create_rectangle(sx1, sy1, sx2, sy2, outline="#0f172a", width=2.5)
        bar_positions = []
        if column.long_bar_qty_x > 1:
            x_spacing = (sx2 - sx1) / (column.long_bar_qty_x - 1)
            for i in range(column.long_bar_qty_x): bar_positions.append(
                (sx1 + i * x_spacing, sy1)); bar_positions.append((sx1 + i * x_spacing, sy2))
        if column.long_bar_qty_y > 2:
            y_spacing = (sy2 - sy1) / (column.long_bar_qty_y - 1)
            for i in range(1, column.long_bar_qty_y - 1): bar_positions.append(
                (sx1, sy1 + i * y_spacing)); bar_positions.append((sx2, sy1 + i * y_spacing))
        for (bar_x, bar_y) in bar_positions: canvas.create_oval(bar_x - bar_radius, bar_y - bar_radius,
                                                                bar_x + bar_radius, bar_y + bar_radius, fill="#334155",
                                                                outline="")
        if column.supplementary_stirrups_qty > 0 and column.long_bar_qty_x > 2:
            hook_spacing = (sx2 - sx1) / (column.long_bar_qty_x - 1)
            for i in range(column.supplementary_stirrups_qty):
                hook_x = sx1 + hook_spacing * (i + 1)
                if hook_x < sx2 - hook_spacing:
                    canvas.create_line(hook_x, sy1, hook_x, sy2, fill="#0f172a", width=2);
                    canvas.create_arc(hook_x - bar_radius, sy1 - bar_radius, hook_x + bar_radius, sy1 + bar_radius,
                                      start=180, extent=180, style=tk.ARC, outline="#0f172a", width=2);
                    canvas.create_arc(hook_x - bar_radius, sy2 - bar_radius, hook_x + bar_radius, sy2 + bar_radius,
                                      start=0, extent=180, style=tk.ARC, outline="#0f172a", width=2)
        canvas.create_line(x1, y1 - 20, x2, y1 - 20, arrow=tk.BOTH);
        canvas.create_text((x1 + x2) / 2, y1 - 25, text=f"{column.length} cm", anchor="s")
        canvas.create_line(x1 - 20, y1, x1 - 20, y2, arrow=tk.BOTH);
        canvas.create_text(x1 - 25, (y1 + y2) / 2, text=f"{column.width} cm", angle=90, anchor="s")
        canvas.create_text(canvas_w / 2, 20, text=f"Corte Transversal: {column.name}", font=("Segoe UI", 11, "bold"))

    def draw_stirrup_distribution(self, column):
        canvas = self.canvas_elevation;
        canvas.delete("all")
        padding, canvas_w, canvas_h = 40, canvas.winfo_width(), canvas.winfo_height()
        if canvas_w < 50: self.after(100, lambda: self.draw_stirrup_distribution(column)); return

        col_draw_w = 120
        scale_h = (canvas_h - 2 * padding) / (column.height * 100)
        offset_x, bar_offset = (canvas_w - col_draw_w) / 2, 10
        x1, y1 = offset_x, padding
        x2, y2 = offset_x + col_draw_w, padding + column.height * 100 * scale_h
        canvas.create_rectangle(x1, y1, x2, y2, fill="#e2e8f0", outline="")
        reqs = column.calculate_requirements()

        # --- NUEVO: Visualización de Traslapes ---
        if reqs['longitudinal']['has_splices']:
            splice_len_px = reqs['longitudinal']['splice_length'] * 100 * scale_h
            splice_mid_y = y2 / 2
            # Barra principal (viene de abajo)
            canvas.create_line(x1 + bar_offset, y2, x1 + bar_offset, splice_mid_y - splice_len_px / 2, width=3,
                               fill="#334155")
            # Barra de empalme (continúa hacia arriba)
            canvas.create_line(x2 - bar_offset, splice_mid_y + splice_len_px / 2, x2 - bar_offset, y1, width=3,
                               fill="#334155")
            # Acotación del traslape
            dim_x_splice = x1 - 20
            canvas.create_line(dim_x_splice, splice_mid_y - splice_len_px / 2, dim_x_splice,
                               splice_mid_y + splice_len_px / 2, arrow=tk.BOTH)
            canvas.create_text(dim_x_splice - 5, splice_mid_y,
                               text=f"Traslape\n{reqs['longitudinal']['splice_length']:.2f}m", anchor="e")
        else:
            canvas.create_line(x1 + bar_offset, y2, x1 + bar_offset, y1, width=3, fill="#334155")
            canvas.create_line(x2 - bar_offset, y2, x2 - bar_offset, y1, width=3, fill="#334155")

        # --- MEJORADO: Visualización de Estribos ---
        y_bottom_limit = y1 + column.confined_length * scale_h
        y_top_limit = y2 - column.confined_length * scale_h
        # Dibujar solo algunas líneas representativas
        for i in range(4): canvas.create_line(x1, y1 + 5 * scale_h + i * column.stirrup_spacing_confined * scale_h, x2,
                                              y1 + 5 * scale_h + i * column.stirrup_spacing_confined * scale_h, width=2,
                                              fill="#0f172a")
        for i in range(4): canvas.create_line(x1, y2 - 5 * scale_h - i * column.stirrup_spacing_confined * scale_h, x2,
                                              y2 - 5 * scale_h - i * column.stirrup_spacing_confined * scale_h, width=2,
                                              fill="#0f172a")
        for i in range(3): canvas.create_line(x1, (y_bottom_limit + y_top_limit) / 2 + (
                    i - 1) * column.stirrup_spacing_central * scale_h, x2, (y_bottom_limit + y_top_limit) / 2 + (
                                                          i - 1) * column.stirrup_spacing_central * scale_h, width=2,
                                              fill="#0f172a")

        # Acotaciones de distribución
        dim_x_dist = x2 + 50
        # Zona confinada inferior
        canvas.create_line(dim_x_dist, y1, dim_x_dist, y_bottom_limit, arrow=tk.BOTH)
        canvas.create_text(dim_x_dist + 5, (y1 + y_bottom_limit) / 2, anchor="w",
                           text=f"{reqs['stirrups']['qty_confined_zone']} Est.\n@ {column.stirrup_spacing_confined} cm")
        # Zona central
        canvas.create_line(dim_x_dist, y_bottom_limit, dim_x_dist, y_top_limit, arrow=tk.BOTH)
        canvas.create_text(dim_x_dist + 5, (y_bottom_limit + y_top_limit) / 2, anchor="w",
                           text=f"{reqs['stirrups']['qty_central']} Est.\n@ {column.stirrup_spacing_central} cm")
        # Zona confinada superior
        canvas.create_line(dim_x_dist, y_top_limit, dim_x_dist, y2, arrow=tk.BOTH)
        canvas.create_text(dim_x_dist + 5, (y_top_limit + y2) / 2, anchor="w",
                           text=f"{reqs['stirrups']['qty_confined_zone']} Est.\n@ {column.stirrup_spacing_confined} cm")

        canvas.create_text(canvas_w / 2, 15, text=f"Elevación: Detalle de Acero ({column.name})",
                           font=("Segoe UI", 11, "bold"))


if __name__ == "__main__":
    app = AdvancedSteelSimulator()
    app.mainloop()

