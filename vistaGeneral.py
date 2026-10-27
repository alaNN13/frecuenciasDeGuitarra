"""
Módulo Frontend: Interfaz Gráfica de Usuario (UI/UX)
Maneja exclusivamente la vista (Tkinter), eventos de usuario y actualización de gráficas.
"""
import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.image as mpimg

# Importación del núcleo modular de backend
import nucleo_backend as backend  

try:
    import sounddevice as sd
except ImportError:
    sd = None

try:
    import music21
except ImportError:
    music21 = None

# Constantes de diseño (Temática de estudio oscuro)
BG = "#121212"
PANEL = "#1E1E1E"
PANEL2 = "#2A2A2A"
WHITE = "#E0E0E0"
ORANGE = "#F57C00"
GREEN = "#66BB6A"
RED = "#E53935"
MUTED = "#888888"

class MusicNotesAppUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistema Inteligente de Detección de Notas Musicales")
        self.root.geometry("1280x850")
        self.root.minsize(1020, 700)
        self.root.configure(bg=BG)

        # Variables de estado de la UI
        self.file_path = None
        self.audio_data = None
        self.sample_rate = None
        self.dsp_results = None
        self.predictions = []
        self.dsp_done = False
        self.predictions_done = False

        self._configure_styles()
        self._build_ui()
        self.log("Sistema iniciado. Interfaz modular cargada correctamente.")
        
        self.is_latin = True

    def _configure_styles(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TProgressbar", troughcolor=PANEL2, background=ORANGE)
        style.configure("Treeview", background=PANEL, foreground=WHITE, fieldbackground=PANEL, rowheight=27)
        style.configure("Treeview.Heading", background=PANEL2, foreground=ORANGE, font=("Segoe UI", 10, "bold"))
        style.map("Treeview", background=[("selected", "#5A3516")], foreground=[("selected", WHITE)])

    def _build_ui(self):
        # 1. Header
        header = tk.Frame(self.root, bg=BG, padx=22, pady=16)
        header.pack(fill="x")
        tk.Label(header, text="DETECCIÓN DE NOTAS MUSICALES", bg=BG, fg=WHITE, font=("Segoe UI", 21, "bold")).pack(anchor="w")
        tk.Label(header, text="Arquitectura Modular · Procesamiento Digital de Señales", bg=BG, fg=ORANGE, font=("Segoe UI", 10)).pack(anchor="w", pady=(4, 0))

        # 2. Panel de Ingesta (Top)
        top = tk.Frame(self.root, bg=PANEL, padx=14, pady=12)
        top.pack(fill="x", padx=20, pady=(0, 12))
        
        self.file_label = tk.Label(top, text="Ningún archivo seleccionado", bg=PANEL, fg=MUTED, anchor="w", font=("Segoe UI", 10), width=45)
        self.file_label.pack(side="left", fill="x", expand=True, padx=(0, 10))

        self._button(top, "Cargar WAV", self.handler_load_wav).pack(side="left", padx=4)
        self._button(top, "Simular 440 Hz", self.handler_simulate, color=GREEN, fg=BG).pack(side="left", padx=4)
        self.play_button = self._button(top, "Reproducir", self.handler_play)
        self.play_button.pack(side="left", padx=4)

        # 3. Cuerpo Principal
        body = tk.Frame(self.root, bg=BG)
        body.pack(fill="both", expand=True, padx=20, pady=(0, 12))
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=2)
        body.rowconfigure(0, weight=1)
        body.rowconfigure(1, weight=1)

        # 3a. Panel de Gráficas (Matplotlib embed)
        self.plot_panel = tk.Frame(body, bg=PANEL, padx=10, pady=10)
        self.plot_panel.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=(0, 10))
        tk.Label(self.plot_panel, text="VISUALIZACIÓN DE LA SEÑAL Y PARTITURA", bg=PANEL, fg=ORANGE, font=("Segoe UI", 11, "bold")).pack(anchor="w")

        self.figure = Figure(figsize=(7.2, 7.0), dpi=100, facecolor=PANEL)
        self.ax_wave = self.figure.add_subplot(311)
        self.ax_spec = self.figure.add_subplot(312)
        self.ax_staff = self.figure.add_subplot(313)
        
        self._style_axes(self.ax_wave, "Forma de onda", "Tiempo (s)", "Amplitud")
        self._style_axes(self.ax_spec, "Espectrograma", "Tiempo (s)", "Frecuencia (Hz)")
        self._style_staff_axes(self.ax_staff, "Pentagrama (Notación Formal)")
        
        self.figure.tight_layout(pad=2.0)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_panel)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, pady=(8, 0))

        # 3b. Panel de Controles
        controls = tk.Frame(body, bg=PANEL, padx=14, pady=12)
        controls.grid(row=0, column=1, sticky="nsew", pady=(0, 10))
        tk.Label(controls, text="PROCESAMIENTO", bg=PANEL, fg=ORANGE, font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(0, 10))
        
        self.dsp_button = self._button(controls, "1. Analizar señal (DSP)", self.handler_run_dsp, state="disabled")
        self.dsp_button.pack(fill="x", pady=4)
        self.predict_button = self._button(controls, "2. Detectar notas", self.handler_run_prediction, state="disabled", color=GREEN, fg=BG)
        self.predict_button.pack(fill="x", pady=4)
        self.report_button = self._button(controls, "Exportar reporte PDF", self.handler_export_pdf, state="disabled")
        self.report_button.pack(fill="x", pady=4)
        
        self.progress = ttk.Progressbar(controls, mode="indeterminate")
        self.progress.pack(fill="x", pady=(12, 5))
        self.status_label = tk.Label(controls, text="Estado: esperando audio", bg=PANEL, fg=MUTED, anchor="w")
        self.status_label.pack(fill="x", pady=(3, 0))
        
        self.metrics_label = tk.Label(controls, text="Frecuencia: -- Hz\nNota: --", bg=PANEL, fg=WHITE, justify="left", anchor="w", font=("Segoe UI", 12, "bold"))
        self.metrics_label.pack(fill="x", pady=(14, 0))

        # 3c. Tabla de Resultados
        results_frame = tk.Frame(body, bg=PANEL, padx=10, pady=10)
        results_frame.grid(row=1, column=1, sticky="nsew")
        tk.Label(results_frame, text="NOTAS DETECTADAS", bg=PANEL, fg=ORANGE, font=("Segoe UI", 11, "bold")).pack(anchor="w")
        
        self.table = ttk.Treeview(results_frame, columns=("time", "note", "freq"), show="headings", height=6)
        self.table.heading("time", text="Tiempo (ns)")
        self.table.heading("note", text="Nota")
        self.table.heading("freq", text="Frecuencia (Hz)")
        self.table.column("time", width=90, anchor="center")
        self.table.column("note", width=90, anchor="center")
        self.table.column("freq", width=130, anchor="center")
        self.table.pack(fill="both", expand=True, pady=(8, 0))
        self.table.bind("<Button-1>", self.on_table_click)

        # 4. Consola del Sistema
        log_frame = tk.Frame(self.root, bg=PANEL, padx=12, pady=8)
        log_frame.pack(fill="x", padx=20, pady=(0, 18))
        tk.Label(log_frame, text="MONITOR DEL SISTEMA", bg=PANEL, fg=ORANGE, font=("Segoe UI", 9, "bold")).pack(anchor="w")
        self.log_text = tk.Text(log_frame, height=4, bg="#0C0C0C", fg=GREEN, relief="flat", font=("Consolas", 9), state="disabled")
        self.log_text.pack(fill="x", pady=(5, 0))

    def _button(self, parent, text, command, color=ORANGE, fg=WHITE, state="normal"):
        return tk.Button(parent, text=text, command=command, bg=color, fg=fg, state=state, relief="flat", bd=0, padx=12, pady=9, font=("Segoe UI", 9, "bold"), cursor="hand2")

    def log(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.configure(state="normal")
        self.log_text.insert("end", f"[{timestamp}] {message}\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _set_status(self, text, color=MUTED):
        self.status_label.configure(text=f"Estado: {text}", fg=color)

    def _style_axes(self, ax, title, xlabel, ylabel):
        ax.set_facecolor("#171717")
        ax.set_title(title, color=ORANGE, fontsize=10, loc="left")
        ax.set_xlabel(xlabel, color=WHITE, fontsize=8)
        ax.set_ylabel(ylabel, color=WHITE, fontsize=8)
        ax.tick_params(colors=MUTED, labelsize=8)

    def _style_staff_axes(self, ax, title):
        ax.set_facecolor("#171717")
        ax.set_title(title, color=ORANGE, fontsize=10, loc="left")
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_visible(False)

    def _reset_results(self):
        self.dsp_done = False
        self.predictions_done = False
        self.predictions = []
        for item in self.table.get_children():
            self.table.delete(item)
        self.dsp_button.configure(state="normal" if self.audio_data is not None else "disabled")
        self.predict_button.configure(state="disabled")
        self.report_button.configure(state="disabled")
        self.metrics_label.configure(text="Frecuencia: -- Hz\nNota: --")
        self.ax_staff.clear()
        self._style_staff_axes(self.ax_staff, "Pentagrama (Notación Formal)")
        self.canvas.draw_idle()

    # ==========================================
    # MANEJADORES DE EVENTOS Y CONEXIÓN AL BACKEND
    # ==========================================
    def handler_load_wav(self):
        path = filedialog.askopenfilename(filetypes=[("Archivos WAV", "*.wav")])
        if not path: return
        
        try:
            self.sample_rate, self.audio_data = backend.load_wav(path)
            self.file_path = path
            self.file_label.configure(text=os.path.basename(path), fg=WHITE)
            self._reset_results()
            self._set_status("Audio cargado correctamente", GREEN)
            self.log(f"Archivo cargado: {os.path.basename(path)}")
            self._plot_waveform()
        except Exception as exc:
            messagebox.showerror("Error", f"Error en la ingesta: {str(exc)}")

    def handler_simulate(self):
        try:
            self.sample_rate, self.audio_data = backend.simulate_tone()
            self.file_path = None
            self.file_label.configure(text="Tono simulado 440Hz (La4)", fg=WHITE)
            self._reset_results()
            self._set_status("Simulación generada", GREEN)
            self.log("Tono puro de prueba de 440 Hz generado en memoria.")
            self._plot_waveform()
        except Exception as exc:
            messagebox.showerror("Error", str(exc))

    def _plot_waveform(self):
        if self.audio_data is None: return
        self.ax_wave.clear()
        self._style_axes(self.ax_wave, "Forma de onda", "Tiempo (s)", "Amplitud")
        duration = len(self.audio_data) / self.sample_rate
        step = max(1, len(self.audio_data) // 12000)
        times = np.arange(0, len(self.audio_data), step) / self.sample_rate
        self.ax_wave.plot(times, self.audio_data[::step], color=ORANGE, linewidth=0.8)
        self.ax_wave.set_xlim(0, max(duration, 0.01))
        self.figure.tight_layout(pad=2.0)
        self.canvas.draw_idle()

    def handler_play(self):
        if self.audio_data is None:
            messagebox.showinfo("Sin audio", "Primero carga un archivo WAV o genera la simulación.")
            return
        if sd is None:
            messagebox.showwarning("Aviso", "Instala sounddevice para reproducir audio:\npip install sounddevice")
            return
        try:
            sd.stop()
            sd.play(self.audio_data.astype('float32'), self.sample_rate)
            self.log("Reproducción de audio iniciada.")
        except Exception as exc:
            messagebox.showerror("Error de reproducción", str(exc))

    def handler_run_dsp(self):
        if self.audio_data is None: return
        self.dsp_button.configure(state="disabled")
        self.progress.start(10)
        self._set_status("Analizando señal (DSP)...", ORANGE)
        
        def work():
            try:
                self.dsp_results = backend.run_dsp(self.audio_data, self.sample_rate)
                self.root.after(0, self._finish_dsp_ui)
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Fallo DSP: {str(e)}"))
                self.root.after(0, self.progress.stop)
            
        threading.Thread(target=work, daemon=True).start()

    def _finish_dsp_ui(self):
        self.progress.stop()
        self.dsp_done = True
        self.predict_button.configure(state="normal")
        self._set_status("DSP completado", GREEN)
        
        if self.dsp_results:
            freqs, times, magnitude = self.dsp_results
            self.ax_spec.clear()
            self._style_axes(self.ax_spec, "Espectrograma", "Tiempo (s)", "Frecuencia (Hz)")
            
            mask = freqs <= min(5000, self.sample_rate / 2)
            db = 20 * np.log10(np.maximum(magnitude[mask], 1e-7))
            self.ax_spec.pcolormesh(times, freqs[mask], db, shading="auto", cmap="inferno", vmin=-80, vmax=-10)
            self.ax_spec.set_ylim(0, min(5000, self.sample_rate / 2))
            self.figure.tight_layout(pad=2.0)
            self.canvas.draw_idle()
            
            self.log("Análisis DSP completado y espectrograma renderizado con éxito.")

    def handler_run_prediction(self):
        if self.audio_data is None or not self.dsp_done: return
        self.predict_button.configure(state="disabled")
        self.progress.start(10)
        self._set_status("Detectando notas...", ORANGE)
        
        def work():
            try:
                self.predictions = backend.run_ml_prediction(self.audio_data, self.sample_rate)
                self.root.after(0, self._finish_prediction_ui)
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Fallo de predicción: {str(e)}"))
                self.root.after(0, self.progress.stop)
            
        threading.Thread(target=work, daemon=True).start()

    def _finish_prediction_ui(self):
        self.progress.stop()
        self.predictions_done = True
        self.predict_button.configure(state="normal")
        self.report_button.configure(state="normal")
        self._set_status(f"Detección completada ({len(self.predictions)} notas)", GREEN)
        self.log(f"Se detectaron {len(self.predictions)} cambios de nota.")
        
        for item in self.table.get_children():
            self.table.delete(item)
            
        for item in self.predictions:
            self.table.insert("", "end", values=(
                str(item["time"]), item["note"], f"{item['freq']:.2f}"
            ))
            
        if self.predictions:
            first = self.predictions[0]
            self.metrics_label.configure(text=f"Primera nota: {first['note']}\nFrecuencia: {first['freq']:.2f} Hz")
            self._draw_matplotlib_staff(self.predictions)
        else:
            self.metrics_label.configure(text="Frecuencia: -- Hz\nNota: --")
            self.ax_staff.clear()
            self._style_staff_axes(self.ax_staff, "Pentagrama (Notación Formal)")
            self.canvas.draw_idle()

    def _draw_matplotlib_staff(self, results):
        ax = self.ax_staff
        ax.clear()
        self._style_staff_axes(ax, "Pentagrama (Notación Formal)")
        ax.set_xlim(0, 10)
        ax.set_ylim(-6, 10)

        staff_lines = [2, 3, 4, 5, 6]
        for y in staff_lines:
            ax.plot([0.5, 9.5], [y, y], color=WHITE, lw=1.2, zorder=1)

        ax.plot([0.5, 0.5], [2, 6], color=WHITE, lw=2, zorder=1)
        ax.plot([9.5, 9.5], [2, 6], color=WHITE, lw=2, zorder=1)
        ax.text(0.7, 3.2, "𝄞", color=ORANGE, fontsize=36, va="center", ha="center", zorder=2)

        if not results: return

        max_notes = min(len(results), 12)
        x_positions = np.linspace(2.0, 9.0, max_notes)

        for i, item in enumerate(results[:max_notes]):
            midi = item.get("midi")
            if not midi: continue

            steps_from_e4 = (midi - 64) / 2.0
            y_pos = 2.0 + steps_from_e4
            x = x_positions[i]

            ax.plot(x, y_pos, "o", color=GREEN, markersize=8, zorder=4)
            stem_dir = 1.8 if y_pos < 4.0 else -1.8
            ax.plot([x + 0.08, x + 0.08], [y_pos, y_pos + stem_dir], color=GREEN, lw=1.5, zorder=3)
            
            if "♯" in item["note"]:
                ax.text(x - 0.25, y_pos, "♯", color=GREEN, fontsize=11, fontweight="bold", va="center", ha="right", zorder=5)

            ax.text(x, -2.5, item["note"], color=WHITE, fontsize=8, fontweight="bold",
                    ha="right", va="top", rotation=35,
                    bbox=dict(boxstyle="round,pad=0.15", facecolor=PANEL, edgecolor="none", alpha=0.8), zorder=6)
        
        self.figure.tight_layout(pad=2.0)
        self.canvas.draw_idle()

    def handler_export_pdf(self):
        if not self.predictions_done: return
        path = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF", "*.pdf")], initialfile="reporte_notas_musicales.pdf")
        if not path: return
        try:
            backend.generate_pdf_report(path, self.predictions, self.file_path, self.sample_rate)
            self.log(f"Reporte PDF exportado en: {path}")
            self._set_status("PDF generado", GREEN)
            messagebox.showinfo("Reporte generado", f"Se guardó el reporte en:\n{path}")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo generar PDF: {str(e)}")

    def on_table_click(self, event):
        """Detecta si el usuario hizo clic en la cabecera de la columna 'Nota'."""
        region = self.table.identify("region", event.x, event.y)
        if region == "heading":
            col = self.table.identify_column(event.x)  # #2 corresponde a la columna "note"
            if col == "#2":
                self.toggle_notation()

    def _get_note_name(self, midi):
        """Convierte un valor MIDI a texto según la nomenclatura activa de forma general."""
        if midi is None or not isinstance(midi, (int, np.integer)):
            return "Indeterminada"
        
        note_idx = midi % 12
        octave = midi // 12 - 1
        
        if self.is_latin:
            notes = ["Do", "Do♯", "Re", "Re♯", "Mi", "Fa", "Fa♯", "Sol", "Sol♯", "La", "La♯", "Si"]
        else:
            notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
            
        return f"{notes[note_idx]}{octave}"

    def toggle_notation(self):
        """Alterna el estado de la nomenclatura e instantáneamente actualiza toda la interfaz."""
        # Conmutar estado
        self.is_latin = not self.is_latin
        
        if not self.predictions:
            self.log("Nomenclatura alternada. (Sin notas detectadas actualmente)")
            return

        # Actualizar las notas guardadas en el modelo con base en su identificador universal MIDI
        for item in self.predictions:
            if "midi" in item and item["midi"] is not None:
                item["note"] = self._get_note_name(item["midi"])

        # Refrescar los elementos visuales de la tabla
        for child in self.table.get_children():
            self.table.delete(child)
            
        for item in self.predictions:
            self.table.insert("", "end", values=(
                str(item["time"]), item["note"], f"{item['freq']:.2f}"
            ))

        # Actualizar la métrica principal lateral
        if self.predictions:
            first = self.predictions[0]
            self.metrics_label.configure(
                text=f"Primera nota: {first['note']}\nFrecuencia: {first['freq']:.2f} Hz"
            )
            # Redibujar el pentagrama vectorial en Matplotlib para reflejar el cambio en las notas
            self._draw_matplotlib_staff(self.predictions)

        self.log("Nomenclatura de notas visualizada en el esquema alternativo.")
if __name__ == "__main__":
    root = tk.Tk()
    app = MusicNotesAppUI(root)
    root.mainloop()