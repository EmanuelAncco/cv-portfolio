import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np

# --- Tu función de cálculo tal cual ---
def calculate_pipeline_stress(
    D, t, E, nu, alpha_T, Sy,
    H, p, delta_T,
    W_traffic, A_contacto, If,
    PGV, C, alpha_seismic
):
    sigma_h = (p * D) / (2 * t)
    sigma_a_p = nu * sigma_h
    sigma_a_T = E * alpha_T * delta_T
    Hc = H - D / 2.0
    Qd = 0.0 if Hc <= 0 else (3 * W_traffic) / (2 * np.pi * Hc**2)
    Wt = If * Qd * D
    X = 2 * Hc
    km = 10
    Zpipe = (np.pi * D**2 * t) / 4.0
    sigma_L_traf = (Wt * X**2 / km) / Zpipe if km and Zpipe else 0.0
    sigma_a_w = E * alpha_seismic * (PGV / C) if C else 0.0
    sigma_L = sigma_a_p + sigma_a_T + sigma_L_traf + sigma_a_w
    sigma_h_total = sigma_h
    sigma_VM = np.sqrt(sigma_L**2 - sigma_L*sigma_h_total + sigma_h_total**2)
    ratio = sigma_VM / Sy if Sy else float('inf')
    return {
        "σh (MPa)": sigma_h/1e6,
        "σa,p (MPa)": sigma_a_p/1e6,
        "σa,T (MPa)": sigma_a_T/1e6,
        "σL,traf (MPa)": sigma_L_traf/1e6,
        "σa,w (MPa)": sigma_a_w/1e6,
        "σL total (MPa)": sigma_L/1e6,
        "σVM (MPa)": sigma_VM/1e6,
        "Ratio σVM/Sy": ratio
    }

# --- Valores por defecto ---
defaults = {
    "D": 0.61, "t": 0.0095, "E": 2.07e11, "nu": 0.3, "alpha_T": 1.2e-5, "Sy": 4.48e8,
    "H": 1.5, "p": 7e6, "delta_T": -15,
    "If": 1.5, "PGV": 0.40, "C": 800, "alpha_seismic": 1.0
}
# Definimos algunos vehículos de ejemplo
vehicles = {
    "Camión ligero (35.5 kN/rueda)": {"W_traffic": 35500, "A_contacto": 0.1},
    "Camión medio (50 kN/rueda)":       {"W_traffic": 50000, "A_contacto": 0.12},
    "Camión pesado (75 kN/rueda)":      {"W_traffic": 75000, "A_contacto": 0.15},
}

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Cálculo de Esfuerzos en Gasoducto")
        self.geometry("600x700")
        self._build_ui()

    def _build_ui(self):
        frm = ttk.Frame(self, padding=10)
        frm.pack(fill="both", expand=True)

        # Entradas para parámetros generales
        self.entries = {}
        row = 0
        for key, val in defaults.items():
            ttk.Label(frm, text=key).grid(row=row, column=0, sticky="e", pady=2)
            ent = ttk.Entry(frm); ent.grid(row=row, column=1, pady=2, sticky="we")
            ent.insert(0, str(val))
            self.entries[key] = ent
            row += 1

        # Selector de vehículo
        ttk.Label(frm, text="Vehículo").grid(row=row, column=0, sticky="e", pady=5)
        self.vehicle_var = tk.StringVar(value=list(vehicles.keys())[0])
        veh_menu = ttk.OptionMenu(frm, self.vehicle_var, self.vehicle_var.get(),
                                  *vehicles.keys(), command=self._on_vehicle_change)
        veh_menu.grid(row=row, column=1, sticky="we", pady=5)
        row += 1

        # Entradas para carga de tráfico (W_traffic y A_contacto)
        for key in ("W_traffic", "A_contacto"):
            ttk.Label(frm, text=key).grid(row=row, column=0, sticky="e", pady=2)
            ent = ttk.Entry(frm); ent.grid(row=row, column=1, pady=2, sticky="we")
            self.entries[key] = ent
            row += 1

        # Botón de cálculo
        calc_btn = ttk.Button(frm, text="Calcular", command=self._calculate)
        calc_btn.grid(row=row, column=0, columnspan=2, pady=10, sticky="we")
        row += 1

        # Text widget para mostrar resultados
        self.result_txt = tk.Text(frm, height=10, width=50)
        self.result_txt.grid(row=row, column=0, columnspan=2, pady=10)

        # Inicializar campos de tráfico según vehículo por defecto
        self._on_vehicle_change(self.vehicle_var.get())

    def _on_vehicle_change(self, vehicle_name):
        params = vehicles[vehicle_name]
        self.entries["W_traffic"].delete(0, tk.END)
        self.entries["W_traffic"].insert(0, str(params["W_traffic"]))
        self.entries["A_contacto"].delete(0, tk.END)
        self.entries["A_contacto"].insert(0, str(params["A_contacto"]))

    def _calculate(self):
        try:
            # Recopilar datos
            kwargs = {k: float(e.get()) for k, e in self.entries.items()}
            # Ejecutar cálculo
            res = calculate_pipeline_stress(**kwargs)
            # Mostrar resultados
            self.result_txt.delete("1.0", tk.END)
            for k, v in res.items():
                self.result_txt.insert(tk.END, f"{k}: {v:.3f}\n")
        except Exception as e:
            messagebox.showerror("Error", f"Revisa tus entradas:\n{e}")

if __name__ == "__main__":
    App().mainloop()
