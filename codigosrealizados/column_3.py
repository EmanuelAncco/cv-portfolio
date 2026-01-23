import tkinter as tk
from tkinter import ttk, messagebox
import logging

# Configuración del logging (buena práctica de ingeniería)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class AppPredimensionViga:
    """
    Una aplicación con interfaz gráfica (Tkinter) para verificar la esbeltez
    de un muro de albañilería y predimensionar una viga de confinamiento
    horizontal (viga collar) si es necesaria.
    """

    def __init__(self, root):
        self.root = root
        self.root.title("Predimensionamiento de Viga de Confinamiento")
        self.root.geometry("700x650")

        self.style = ttk.Style()
        self.style.theme_use('clam')

        # --- Frame de Entradas ---
        input_frame = ttk.LabelFrame(root, text="Datos del Muro", padding=(10, 10))
        input_frame.pack(pady=10, padx=10, fill="x")

        # Altura total
        ttk.Label(input_frame, text="Altura libre total (H) (m):").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.h_total_var = tk.StringVar(value="4.8")
        ttk.Entry(input_frame, textvariable=self.h_total_var, width=10).grid(row=0, column=1, padx=5, pady=5)

        # Espesor del muro
        ttk.Label(input_frame, text="Espesor del muro (t) (cm):").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.t_muro_var = tk.StringVar(value="13")
        ttk.Entry(input_frame, textvariable=self.t_muro_var, width=10).grid(row=1, column=1, padx=5, pady=5)

        # Divisiones (para la viga)
        ttk.Label(input_frame, text="N° de paños verticales (N):").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        self.n_div_var = tk.StringVar(value="2")
        ttk.Entry(input_frame, textvariable=self.n_div_var, width=10).grid(row=2, column=1, padx=5, pady=5)

        # --- Frame de Parámetros (Norma) ---
        params_frame = ttk.LabelFrame(root, text="Parámetros Normativos (E.070)", padding=(10, 10))
        params_frame.pack(pady=10, padx=10, fill="x")

        # Esbeltez máxima
        ttk.Label(params_frame, text="Esbeltez máx. permitida (h_ef / t):").grid(row=0, column=0, sticky="w", padx=5,
                                                                                 pady=5)
        self.lambda_max_var = tk.StringVar(value="20")
        ttk.Entry(params_frame, textvariable=self.lambda_max_var, width=10).grid(row=0, column=1, padx=5, pady=5)

        # Peralte mínimo viga
        ttk.Label(params_frame, text="Peralte/Altura mín. viga (h) (cm):").grid(row=1, column=0, sticky="w", padx=5,
                                                                                pady=5)
        self.h_viga_min_var = tk.StringVar(value="20")
        ttk.Entry(params_frame, textvariable=self.h_viga_min_var, width=10).grid(row=1, column=1, padx=5, pady=5)

        # Botón de cálculo
        self.calc_button = ttk.Button(root, text="Verificar y Predimensionar", command=self.calcular_dimensiones)
        self.calc_button.pack(pady=10, padx=10, fill="x")

        # --- Frame de Resultados (Esquema y Memoria) ---
        results_frame = ttk.Frame(root, padding=(10, 10))
        results_frame.pack(pady=10, padx=10, fill="both", expand=True)

        # Esquema
        self.canvas = tk.Canvas(results_frame, bg="white", width=250, height=350)
        self.canvas.pack(side="left", fill="y", padx=(0, 10))

        # Memoria de Cálculo
        self.memoria_text = tk.Text(results_frame, wrap="word", height=20, width=50, font=("Consolas", 10))
        self.memoria_text.pack(side="right", fill="both", expand=True)

        # Inicializar el dibujo
        self.dibujar_esquema(4.8, 2.4, 2.4, 2)

    def dibujar_esquema(self, H_total, h_ef_1, h_ef_2, n_div):
        """Dibuja un esquema del muro, sus divisiones y la viga de confinamiento."""
        self.canvas.delete("all")

        # Dimensiones del canvas
        cw = 250
        ch = 350

        # Márgenes
        mx = 40
        my = 30

        # Factor de escala
        escala = (ch - 2 * my) / H_total

        # Posiciones
        y_suelo = ch - my
        y_techo = my
        ancho_muro = cw - 2 * mx
        x1 = mx
        x2 = cw - mx

        # Dibujar columnas (límites)
        self.canvas.create_rectangle(x1 - 10, y_techo, x1, y_suelo, fill="gray", outline="black")
        self.canvas.create_rectangle(x2, y_techo, x2 + 10, y_suelo, fill="gray", outline="black")
        self.canvas.create_text(x1 - 5, y_suelo + 10, text="Columna", font=("Arial", 8))
        self.canvas.create_text(x2 + 5, y_suelo + 10, text="Columna", font=("Arial", 8))

        # Dibujar muro (paños)
        self.canvas.create_rectangle(x1, y_techo, x2, y_suelo, fill="#f0f0f0", outline="black")

        # Dibujar viga(s) intermedia(s)
        if n_div > 1:
            # Asumimos divisiones iguales
            h_ef_m = h_ef_1
            y_viga = y_suelo - (h_ef_m * escala)
            h_viga_px = 0.20 * escala  # Dibujar viga de 20cm a escala

            self.canvas.create_rectangle(x1, y_viga - (h_viga_px / 2), x2, y_viga + (h_viga_px / 2),
                                         fill="darkgray", outline="black", dash=(4, 2))
            self.canvas.create_text(x2 + 25, y_viga, text="Viga", font=("Arial", 8, "bold"), anchor="w")
            self.canvas.create_text(x2 + 25, y_viga + 10, text="h=20cm", font=("Arial", 8, "bold"), anchor="w")

        # --- Acotaciones (Dimensiones) ---

        # Cota Total (H)
        cx = 15  # Coordenada X de la cota
        self.canvas.create_line(cx, y_techo, cx, y_suelo, arrow=tk.BOTH)
        self.canvas.create_line(cx - 3, y_techo, x1 - 10, y_techo, dash=(2, 2))
        self.canvas.create_line(cx - 3, y_suelo, x1, y_suelo, dash=(2, 2))
        self.canvas.create_text(cx, ch / 2, text=f"H = {H_total:.2f} m", angle=90, font=("Arial", 9))

        # Cota Efectiva (h_ef)
        cx_ef = cw - 15  # Coordenada X de la cota
        if n_div > 1:
            y_viga_centro = y_suelo - (h_ef_1 * escala)  # Asumiendo divisiones iguales

            # Cota paño inferior
            self.canvas.create_line(cx_ef, y_suelo, cx_ef, y_viga_centro, arrow=tk.BOTH)
            self.canvas.create_line(x2 + 10, y_suelo, cx_ef + 3, y_suelo, dash=(2, 2))
            self.canvas.create_line(x2, y_viga_centro, cx_ef + 3, y_viga_centro, dash=(2, 2))
            self.canvas.create_text(cx_ef, (y_suelo + y_viga_centro) / 2, text=f"h_ef = {h_ef_1:.2f} m", angle=270,
                                    font=("Arial", 9))

            # Cota paño superior
            self.canvas.create_line(cx_ef, y_viga_centro, cx_ef, y_techo, arrow=tk.BOTH)
            self.canvas.create_line(x2, y_viga_centro, cx_ef + 3, y_viga_centro, dash=(2, 2))
            self.canvas.create_line(x2 + 10, y_techo, cx_ef + 3, y_techo, dash=(2, 2))
            self.canvas.create_text(cx_ef, (y_viga_centro + y_techo) / 2, text=f"h_ef = {h_ef_2:.2f} m", angle=270,
                                    font=("Arial", 9))

        self.canvas.create_text(cw / 2, 10, text="Esquema del Muro Confinado", font=("Arial", 10, "bold"))

    def calcular_dimensiones(self):
        """
        Realiza la verificación de esbeltez y genera la memoria de cálculo
        para la viga de confinamiento.
        """
        try:
            # 1. Obtener datos de entrada
            H_total = float(self.h_total_var.get())
            t_muro_cm = float(self.t_muro_var.get())
            n_div = int(self.n_div_var.get())
            lambda_max = float(self.lambda_max_var.get())
            h_viga_min_cm = float(self.h_viga_min_var.get())

            # Validaciones de ingeniería
            if H_total <= 0 or t_muro_cm <= 0 or n_div <= 0 or lambda_max <= 0 or h_viga_min_cm <= 0:
                raise ValueError("Los valores deben ser positivos.")

            # 2. Cálculos
            t_muro_m = t_muro_cm / 100.0

            # Esbeltez sin viga
            lambda_sin_viga = H_total / t_muro_m

            # Esbeltez con viga
            h_ef = H_total / n_div
            lambda_con_viga = h_ef / t_muro_m

            # 3. Predimensionamiento de Viga
            # El ancho de la viga es SIEMPRE igual al espesor del muro
            b_viga_rec = t_muro_cm
            # La altura (peralte) es el mínimo normativo
            h_viga_rec = h_viga_min_cm

            # 4. Generar Memoria de Cálculo
            memoria = "--- MEMORIA DE CÁLCULO (PREDIMENSIONAMIENTO) ---\n\n"
            memoria += "1. DATOS INICIALES:\n"
            memoria += f" - Altura libre total (H):   {H_total:.2f} m\n"
            memoria += f" - Espesor del muro (t):     {t_muro_cm:.0f} cm\n"
            memoria += f" - N° de paños verticales (N): {n_div}\n"
            memoria += f" - Esbeltez Máx. (Norma):    {lambda_max:.1f}\n\n"

            memoria += "2. VERIFICACIÓN DE ESBELTEZ (SIN VIGA):\n"
            memoria += f" - Esbeltez (H/t) = {H_total:.2f} m / {t_muro_m:.2f} m = {lambda_sin_viga:.2f}\n"

            if lambda_sin_viga > lambda_max:
                memoria += f" - !ALERTA!: {lambda_sin_viga:.2f} > {lambda_max:.1f}. El muro es DEMASIADO ESBELTO.\n"
                memoria += " - Se requiere una viga de confinamiento horizontal (viga collar).\n\n"
            else:
                memoria += f" - OK: {lambda_sin_viga:.2f} <= {lambda_max:.1f}. El muro cumple.\n"
                memoria += " - No se requeriría viga intermedia por esbeltez.\n\n"

            memoria += "3. VERIFICACIÓN DE ESBELTEZ (CON VIGA INTERMEDIA):\n"
            memoria += f" - Altura efectiva paño (h_ef) = {H_total:.2f} m / {n_div} = {h_ef:.2f} m\n"
            memoria += f" - Esbeltez paño (h_ef/t) = {h_ef:.2f} m / {t_muro_m:.2f} m = {lambda_con_viga:.2f}\n"

            if lambda_con_viga > lambda_max:
                memoria += f" - !FALLA!: {lambda_con_viga:.2f} > {lambda_max:.1f}.\n"
                memoria += " - La esbeltez del paño sigue siendo muy alta.\n"
                memoria += " - SOLUCIÓN: Aumentar el N° de paños (ej. 3) o el espesor del muro.\n\n"
            else:
                memoria += f" - OK: {lambda_con_viga:.2f} <= {lambda_max:.1f}.\n"
                memoria += " - La división es adecuada y los paños cumplen con la esbeltez.\n\n"

            memoria += "4. PREDIMENSIONAMIENTO VIGA DE CONFINAMIENTO:\n"
            memoria += "   (Basado en Norma E.070 - Mínimos)\n\n"
            memoria += f" - ANCHO (b): Debe ser igual al espesor del muro.\n"
            memoria += f"   b = {b_viga_rec:.0f} cm\n\n"
            memoria += f" - PERALTE/ALTURA (h): Mínimo normativo.\n"
            memoria += f"   h = {h_viga_rec:.0f} cm\n\n"
            memoria += " - DIMENSIÓN RECOMENDADA: "
            memoria += f"Viga de {b_viga_rec:.0f} cm x {h_viga_rec:.0f} cm\n\n"
            memoria += " - REFUERZO MÍNIMO (Referencial):\n"
            memoria += "   - 4 Varillas 3/8\" (o 2 Varillas 1/2\")\n"
            memoria += "   - Estribos 1/4\" @ 10-20 cm (según diseño)\n"

            # 5. Actualizar UI
            self.memoria_text.delete("1.0", tk.END)
            self.memoria_text.insert("1.0", memoria)

            # 6. Actualizar Dibujo
            # Asumimos paños iguales por simplicidad
            self.dibujar_esquema(H_total, h_ef, h_ef, n_div)

            logging.info(f"Cálculo de viga realizado con H={H_total}, t={t_muro_cm}, N={n_div}")

        except ValueError as e:
            messagebox.showerror("Error de Entrada",
                                 f"Por favor, ingrese valores numéricos válidos y positivos. \nError: {e}")
            logging.warning(f"Error en la entrada de datos: {e}")
        except Exception as e:
            messagebox.showerror("Error Inesperado", f"Ha ocurrido un error: {e}")
            logging.error(f"Error inesperado en cálculo: {e}", exc_info=True)


if __name__ == "__main__":
    logging.info("Iniciando la aplicación de predimensionamiento de vigas.")
    app_root = tk.Tk()
    app = AppPredimensionViga(app_root)
    app_root.mainloop()
    logging.info("Aplicación cerrada.")