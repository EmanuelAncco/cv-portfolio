# ==================================================================================================
# == OPTIMIZADOR VISUAL PARA EL PROBLEMA DE CORTE DE ACERO (1D CUTTING STOCK PROBLEM)
# == Enfoque: Programación Lineal Entera (ILP) con GUI en Tkinter
# == Autor: Dr. Asesor de Robótica en Construcción
# == Versión: 2.0
# == Descripción:
# == Este script integra el solver ILP con una interfaz gráfica de usuario (GUI)
# == construida con Tkinter. Permite al usuario introducir datos de forma interactiva y
# == visualizar los resultados de la optimización, facilitando la comunicación
# == de la solución óptima.
# ==================================================================================================

import tkinter as tk
from tkinter import ttk, messagebox, Canvas
import pulp
from collections import defaultdict


# --------------------------------------------------------------------------------------------------
# FASE 1: GENERACIÓN DE PATRONES DE CORTE VIABLES (Lógica del script original)
# --------------------------------------------------------------------------------------------------
def generar_patrones(longitud_stock, piezas_demandadas):
    patrones = []
    piezas_ordenadas = sorted(piezas_demandadas, reverse=True)

    for pieza in piezas_ordenadas:
        if pieza > longitud_stock:
            continue
        cantidad_max = longitud_stock // pieza
        patrones.append([pieza] * cantidad_max)

    for i in range(len(piezas_ordenadas)):
        patron_actual = []
        espacio_restante = longitud_stock
        for j in range(i, len(piezas_ordenadas)):
            pieza = piezas_ordenadas[j]
            while espacio_restante >= pieza:
                patron_actual.append(pieza)
                espacio_restante -= pieza
        if patron_actual:
            patrones.append(patron_actual)

    patrones_unicos_tuplas = set(tuple(sorted(p)) for p in patrones)
    patrones_unicos = [list(p) for p in patrones_unicos_tuplas]

    return patrones_unicos


# --------------------------------------------------------------------------------------------------
# FASE 2: FORMULACIÓN Y RESOLUCIÓN DEL MODELO ILP (Lógica del script original)
# --------------------------------------------------------------------------------------------------
def resolver_csp_con_ilp(longitud_stock, demanda, patrones):
    piezas_requeridas = list(demanda.keys())
    conteo_piezas_patron = defaultdict(lambda: defaultdict(int))
    for i, patron in enumerate(patrones):
        for pieza in patron:
            if pieza in piezas_requeridas:
                conteo_piezas_patron[i][pieza] += 1

    modelo = pulp.LpProblem("Optimizacion_Corte_Acero", pulp.LpMinimize)
    x = pulp.LpVariable.dicts("Patron", range(len(patrones)), lowBound=0, cat='Integer')
    modelo += pulp.lpSum(x[i] for i in range(len(patrones))), "Total_Barras_Utilizadas"

    for pieza, cantidad in demanda.items():
        modelo += pulp.lpSum(
            conteo_piezas_patron[i][pieza] * x[i] for i in range(len(patrones))) >= cantidad, f"Demanda_Pieza_{pieza}"

    # Usar el solver CBC que viene con PuLP, suprimiendo la salida a consola
    solver = pulp.PULP_CBC_CMD(msg=False)
    modelo.solve(solver)

    estado = pulp.LpStatus[modelo.status]
    if estado == 'Optimal':
        plan_de_corte = {}
        for i in range(len(patrones)):
            num_cortes = pulp.value(x[i])
            if num_cortes > 0:
                plan_de_corte[i] = {
                    'patron': patrones[i],
                    'cantidad': int(num_cortes)
                }
        total_barras = pulp.value(modelo.objective)
        return estado, total_barras, plan_de_corte
    else:
        return estado, None, None


# --------------------------------------------------------------------------------------------------
# FASE 3: APLICACIÓN DE INTERFAZ GRÁFICA (GUI con Tkinter)
# --------------------------------------------------------------------------------------------------
class CuttingStockApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Optimizador Visual de Corte de Acero")
        self.master.geometry("1200x800")

        # Estilo
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TLabel", font=("Inter", 10))
        style.configure("TButton", font=("Inter", 10, "bold"))
        style.configure("Header.TLabel", font=("Inter", 14, "bold"))
        style.configure("Title.TLabel", font=("Inter", 20, "bold"))

        # Colores para las piezas
        self.piece_colors = ['#60a5fa', '#facc15', '#4ade80', '#f87171', '#fb923c', '#818cf8', '#a78bfa', '#e879f9',
                             '#22d3ee']
        self.color_map = {}
        self.color_index = 0

        # Layout principal
        main_frame = ttk.Frame(self.master, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # -- Columna de Entradas --
        input_frame = ttk.LabelFrame(main_frame, text="Parámetros de Entrada", padding="10")
        input_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        ttk.Label(input_frame, text="Longitud Barra Madre (mm):").pack(anchor="w")
        self.stock_length_var = tk.StringVar(value="9000")
        ttk.Entry(input_frame, textvariable=self.stock_length_var).pack(fill=tk.X, pady=(0, 10))

        ttk.Label(input_frame, text="Demanda de Piezas:").pack(anchor="w")
        self.demand_listbox = tk.Listbox(input_frame, height=15)
        self.demand_listbox.pack(fill=tk.X, pady=(0, 5))

        add_frame = ttk.Frame(input_frame)
        add_frame.pack(fill=tk.X)
        self.piece_length_var = tk.StringVar()
        self.piece_qty_var = tk.StringVar()
        ttk.Entry(add_frame, textvariable=self.piece_length_var, width=10).pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Entry(add_frame, textvariable=self.piece_qty_var, width=10).pack(side=tk.LEFT, expand=True, fill=tk.X,
                                                                             padx=5)

        ttk.Button(input_frame, text="Agregar Pieza", command=self.add_demand).pack(fill=tk.X, pady=5)
        ttk.Button(input_frame, text="Quitar Seleccionada", command=self.remove_demand).pack(fill=tk.X)

        ttk.Button(input_frame, text="Optimizar y Visualizar", command=self.run_optimization,
                   style="Accent.TButton").pack(fill=tk.X, pady=(20, 0))
        style.configure("Accent.TButton", foreground="white", background="#22c55e")

        # -- Columna de Resultados --
        results_frame = ttk.Frame(main_frame)
        results_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.summary_frame = ttk.Frame(results_frame)
        self.summary_frame.pack(fill=tk.X, pady=5)

        self.canvas_frame = ttk.Frame(results_frame)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)
        self.canvas = Canvas(self.canvas_frame, bg='white')
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        v_scroll = ttk.Scrollbar(self.canvas_frame, orient="vertical", command=self.canvas.yview)
        v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.configure(yscrollcommand=v_scroll.set)

        self.results_container = ttk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.results_container, anchor="nw")
        self.results_container.bind("<Configure>",
                                    lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))

        # Cargar demanda inicial
        self.initial_demand = {2500: 50, 3100: 42, 1800: 35, 4200: 20, 950: 60}
        for length, qty in self.initial_demand.items():
            self.demand_listbox.insert(tk.END, f"{qty}x de {length} mm")

    def add_demand(self):
        try:
            length = int(self.piece_length_var.get())
            qty = int(self.piece_qty_var.get())
            if length <= 0 or qty <= 0:
                raise ValueError
            self.demand_listbox.insert(tk.END, f"{qty}x de {length} mm")
            self.piece_length_var.set("")
            self.piece_qty_var.set("")
        except ValueError:
            messagebox.showerror("Error", "Por favor ingrese longitud y cantidad válidas (números enteros positivos).")

    def remove_demand(self):
        selected_indices = self.demand_listbox.curselection()
        for i in reversed(selected_indices):
            self.demand_listbox.delete(i)

    def get_color_for_piece(self, length):
        if length not in self.color_map:
            self.color_map[length] = self.piece_colors[self.color_index % len(self.piece_colors)]
            self.color_index += 1
        return self.color_map[length]

    def run_optimization(self):
        # 1. Recolectar datos de la UI
        try:
            stock_length = int(self.stock_length_var.get())
            demand_items = self.demand_listbox.get(0, tk.END)
            if not demand_items:
                messagebox.showwarning("Advertencia", "La lista de demanda está vacía.")
                return

            demanda = {}
            for item in demand_items:
                parts = item.split('x de ')
                qty = int(parts[0])
                length = int(parts[1].replace(' mm', ''))
                demanda[length] = demanda.get(length, 0) + qty
        except (ValueError, IndexError):
            messagebox.showerror("Error",
                                 "Datos de entrada inválidos. Verifique la longitud de stock y la lista de demanda.")
            return

        # 2. Ejecutar el solver (el cerebro de Python)
        piezas_demandadas = list(demanda.keys())
        patrones = generar_patrones(stock_length, piezas_demandadas)
        if not patrones:
            messagebox.showerror("Error",
                                 "No se pudieron generar patrones de corte. Verifique que las piezas no sean más largas que el stock.")
            return

        estado, total_barras, plan_corte = resolver_csp_con_ilp(stock_length, demanda, patrones)

        # 3. Mostrar resultados en la GUI
        if estado == 'Optimal':
            self.display_results(stock_length, demanda, total_barras, plan_corte)
        else:
            messagebox.showerror("Error de Optimización", f"No se encontró una solución óptima. Estado: {estado}")

    def display_results(self, stock_length, demanda, total_barras, plan_corte):
        # Limpiar resultados anteriores
        for widget in self.summary_frame.winfo_children():
            widget.destroy()
        for widget in self.results_container.winfo_children():
            widget.destroy()
        self.color_map.clear()
        self.color_index = 0

        # Calcular estadísticas de resumen
        total_demand_len = sum(k * v for k, v in demanda.items())
        total_stock_len = total_barras * stock_length
        total_waste = total_stock_len - total_demand_len
        waste_percentage = (total_waste / total_stock_len) * 100

        # Mostrar resumen
        ttk.Label(self.summary_frame, text=f"Barras Totales: {int(total_barras)}", font=("Inter", 12, "bold"),
                  foreground="#3b82f6").pack(side=tk.LEFT, padx=10)
        ttk.Label(self.summary_frame, text=f"Desperdicio Total: {(total_waste / 1000):.2f} m",
                  font=("Inter", 12, "bold"), foreground="#ef4444").pack(side=tk.LEFT, padx=10)
        ttk.Label(self.summary_frame, text=f"Eficiencia: {(100 - waste_percentage):.2f}%", font=("Inter", 12, "bold"),
                  foreground="#22c55e").pack(side=tk.LEFT, padx=10)

        # Mostrar plan visual
        canvas_width = 800  # Ancho fijo para las barras visuales
        for i, info in plan_corte.items():
            patron = info['patron']
            cantidad = info['cantidad']

            frame = ttk.Frame(self.results_container, padding=5)
            frame.pack(fill=tk.X, pady=5)

            long_cortada = sum(patron)
            desperdicio = stock_length - long_cortada

            label_text = f"Cortar {cantidad} veces el Patrón (Desperdicio por barra: {desperdicio}mm)"
            ttk.Label(frame, text=label_text, font=("Inter", 10, "bold")).pack(anchor="w")

            bar_canvas = Canvas(frame, width=canvas_width, height=40, bg="#e5e7eb", highlightthickness=0)
            bar_canvas.pack(fill=tk.X)

            current_x = 0
            for piece in patron:
                piece_width = (piece / stock_length) * canvas_width
                color = self.get_color_for_piece(piece)
                bar_canvas.create_rectangle(current_x, 0, current_x + piece_width, 40, fill=color, outline="#9ca3af")
                bar_canvas.create_text(current_x + piece_width / 2, 20, text=str(piece), font=("Inter", 10, "bold"))
                current_x += piece_width

            if desperdicio > 0:
                waste_width = (desperdicio / stock_length) * canvas_width
                bar_canvas.create_rectangle(current_x, 0, current_x + waste_width, 40, fill="#fca5a5",
                                            outline="#9ca3af", stipple="gray25")


# --- EJECUCIÓN PRINCIPAL DE LA APLICACIÓN ---
if __name__ == "__main__":
    root = tk.Tk()
    app = CuttingStockApp(root)
    root.mainloop()
