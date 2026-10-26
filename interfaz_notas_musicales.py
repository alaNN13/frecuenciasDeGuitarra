"""
Sistema Inteligente de Detección de Notas Musicales
Interfaz gráfica de escritorio con Tkinter.

Instalación:
    python -m pip install numpy scipy matplotlib sounddevice reportlab music21

Ejecución:
    python interfaz_notas_musicales.py

Nota:
- La carga, visualización y estimación básica de frecuencia funcionan con WAV.
- La clasificación mostrada es una estimación preliminar por autocorrelación;
  todavía no utiliza un modelo SVM entrenado.
- Soporta notación de partitura formal usando music21 (o renderizador alternativo
  en Matplotlib si no hay motor de partitura de escritorio configurado).
"""

import os
import tempfile
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from datetime import datetime

import numpy as np
from scipy.io import wavfile
from scipy import signal

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.image as mpimg

try:
    import sounddevice as sd
except ImportError:
    sd = None

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet
except ImportError:
    SimpleDocTemplate = None

try:
    import music21
except ImportError:
    music21 = None


BG = "#121212"
PANEL = "#1E1E1E"
PANEL2 = "#252525"
ORANGE = "#F57C00"
GREEN = "#66BB6A"
WHITE = "#F2F2F2"
MUTED = "#B0B0B0"
RED = "#EF5350"

NOTE_NAMES = ["Do", "Do♯", "Re", "Re♯", "Mi", "Fa",
              "Fa♯", "Sol", "Sol♯", "La", "La♯", "Si"]


def frequency_to_note(freq):
    """Convierte una frecuencia positiva a la nota temperada más cercana."""
    if freq is None or not np.isfinite(freq) or freq <= 0:
        return "Indeterminada", None
    midi = int(round(69 + 12 * np.log2(freq / 440.0)))
    name = NOTE_NAMES[midi % 12]
    octave = midi // 12 - 1
    return f"{name}{octave}", midi


def estimate_f0_autocorrelation(frame, sample_rate):
    """Estimación sencilla de frecuencia fundamental mediante autocorrelación."""
    frame = np.asarray(frame, dtype=np.float64)
    if frame.size < 16:
        return None
    frame = frame - np.mean(frame)
    peak = np.max(np.abs(frame))
    if peak < 1e-5:
        return None
    frame = frame / peak
    corr = signal.correlate(frame, frame, mode="full", method="fft")
    corr = corr[len(corr) // 2:]
    if corr.size < 3 or corr[0] <= 0:
        return None
    corr = corr / corr[0]

    min_freq, max_freq = 55.0, min(1500.0, sample_rate / 2 - 1)
    min_lag = max(1, int(sample_rate / max_freq))
    max_lag = min(len(corr) - 1, int(sample_rate / min_freq))
    if max_lag <= min_lag:
        return None

    section = corr[min_lag:max_lag + 1]
    peaks, _ = signal.find_peaks(section)
    if len(peaks):
        valid = [p for p in peaks if section[p] > 0.15]
        chosen = valid[0] if valid else int(np.argmax(section))
        lag = min_lag + chosen
    else:
        lag = min_lag + int(np.argmax(section))

    if lag <= 0:
        return None
    freq = sample_rate / lag
    if min_freq <= freq <= max_freq:
        return float(freq)
    return None


class MusicNotesApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistema Inteligente de Detección de Notas Musicales")
        self.root.geometry("1280x850")
        self.root.minsize(1020, 700)
        self.root.configure(bg=BG)

        self.file_path = None
        self.sample_rate = None
        self.audio = None
        self.original_audio = None
        self.dsp_done = False
        self.predictions_done = False
        self.results = []
        self.playing = False

        self._configure_styles()
        self._build_ui()
        self.log("Sistema iniciado. Carga un archivo WAV o usa la simulación de 440 Hz.")

    def _configure_styles(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TProgressbar", troughcolor=PANEL2,
                        background=ORANGE, bordercolor=PANEL2,
                        lightcolor=ORANGE, darkcolor=ORANGE)
        style.configure("Treeview", background=PANEL, foreground=WHITE,
                        fieldbackground=PANEL, rowheight=27, borderwidth=0)
        style.configure("Treeview.Heading", background=PANEL2,
                        foreground=ORANGE, font=("Segoe UI", 10, "bold"))
        style.map("Treeview", background=[("selected", "#5A3516")],
                  foreground=[("selected", WHITE)])

    def _build_ui(self):
        header = tk.Frame(self.root, bg=BG, padx=22, pady=16)
        header.pack(fill="x")
        tk.Label(header, text="DETECCIÓN DE NOTAS MUSICALES",
                 bg=BG, fg=WHITE, font=("Segoe UI", 21, "bold")).pack(anchor="w")
        tk.Label(header, text="Procesamiento Digital de Señales · Análisis de audio WAV & Notación Formal",
                 bg=BG, fg=ORANGE, font=("Segoe UI", 10)).pack(anchor="w", pady=(4, 0))

        top = tk.Frame(self.root, bg=PANEL, padx=14, pady=12)
        top.pack(fill="x", padx=20, pady=(0, 12))

        self.file_label = tk.Label(top, text="Ningún archivo seleccionado",
                                   bg=PANEL, fg=MUTED, anchor="w",
                                   font=("Segoe UI", 10), width=45)
        self.file_label.pack(side="left", fill="x", expand=True, padx=(0, 10))

        self._button(top, "Cargar WAV", self.load_wav).pack(side="left", padx=4)
        self._button(top, "Simular 440 Hz", self.simulate_tone,
                     color=GREEN, fg=BG).pack(side="left", padx=4)
        self.play_button = self._button(top, "Reproducir", self.play_audio)
        self.play_button.pack(side="left", padx=4)

        body = tk.Frame(self.root, bg=BG)
        body.pack(fill="both", expand=True, padx=20, pady=(0, 12))
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=2)
        body.rowconfigure(0, weight=1)
        body.rowconfigure(1, weight=1)

        self.plot_panel = tk.Frame(body, bg=PANEL, padx=10, pady=10)
        self.plot_panel.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=(0, 10))
        tk.Label(self.plot_panel, text="VISUALIZACIÓN DE LA SEÑAL Y PARTITURA",
                 bg=PANEL, fg=ORANGE, font=("Segoe UI", 11, "bold")).pack(anchor="w")

        # Se configura Figure con 3 subplots: Forma de Onda, Espectrograma y Pentagrama
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

        controls = tk.Frame(body, bg=PANEL, padx=14, pady=12)
        controls.grid(row=0, column=1, sticky="nsew", pady=(0, 10))
        tk.Label(controls, text="PROCESAMIENTO", bg=PANEL, fg=ORANGE,
                 font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(0, 10))
        self.dsp_button = self._button(controls, "1. Analizar señal (DSP)",
                                       self.run_dsp, state="disabled")
        self.dsp_button.pack(fill="x", pady=4)
        self.predict_button = self._button(controls, "2. Detectar notas",
                                           self.run_prediction, state="disabled",
                                           color=GREEN, fg=BG)
        self.predict_button.pack(fill="x", pady=4)
        self.report_button = self._button(controls, "Exportar reporte PDF",
                                          self.export_pdf, state="disabled")
        self.report_button.pack(fill="x", pady=4)
        self.progress = ttk.Progressbar(controls, mode="indeterminate")
        self.progress.pack(fill="x", pady=(12, 5))
        self.status_label = tk.Label(controls, text="Estado: esperando audio",
                                     bg=PANEL, fg=MUTED, anchor="w",
                                     font=("Segoe UI", 9))
        self.status_label.pack(fill="x", pady=(3, 0))
        self.metrics_label = tk.Label(controls, text="Frecuencia: -- Hz\nNota: --",
                                      bg=PANEL, fg=WHITE, justify="left",
                                      anchor="w", font=("Segoe UI", 12, "bold"))
        self.metrics_label.pack(fill="x", pady=(14, 0))

        results_frame = tk.Frame(body, bg=PANEL, padx=10, pady=10)
        results_frame.grid(row=1, column=1, sticky="nsew")
        tk.Label(results_frame, text="NOTAS DETECTADAS",
                 bg=PANEL, fg=ORANGE, font=("Segoe UI", 11, "bold")).pack(anchor="w")
        columns = ("time", "note", "freq")
        self.table = ttk.Treeview(results_frame, columns=columns,
                                  show="headings", height=6)
        self.table.heading("note", text="Nota")
        self.table.heading("freq", text="Frecuencia (Hz)")
        self.table.column("time", width=90, anchor="center")
        self.table.column("note", width=90, anchor="center")
        self.table.column("freq", width=130, anchor="center")
        self.table.pack(fill="both", expand=True, pady=(8, 0))

        log_frame = tk.Frame(self.root, bg=PANEL, padx=12, pady=8)
        log_frame.pack(fill="x", padx=20, pady=(0, 18))
        tk.Label(log_frame, text="MONITOR DEL SISTEMA",
                 bg=PANEL, fg=ORANGE, font=("Segoe UI", 9, "bold")).pack(anchor="w")
        self.log_text = tk.Text(log_frame, height=4, bg="#0C0C0C", fg=GREEN,
                                insertbackground=WHITE, relief="flat",
                                font=("Consolas", 9), wrap="word")
        self.log_text.pack(fill="x", pady=(5, 0))
        self.log_text.configure(state="disabled")

    def _button(self, parent, text, command, color=ORANGE, fg=WHITE, state="normal"):
        return tk.Button(parent, text=text, command=command, bg=color, fg=fg,
                         activebackground=color, activeforeground=fg,
                         disabledforeground="#777777", state=state,
                         relief="flat", bd=0, padx=12, pady=9,
                         font=("Segoe UI", 9, "bold"), cursor="hand2")

    def _style_axes(self, ax, title, xlabel, ylabel):
        ax.set_facecolor("#171717")
        ax.set_title(title, color=ORANGE, fontsize=10, loc="left")
        ax.set_xlabel(xlabel, color=WHITE, fontsize=8)
        ax.set_ylabel(ylabel, color=WHITE, fontsize=8)
        ax.tick_params(colors=MUTED, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#444444")
        ax.grid(True, color="#333333", linewidth=0.5, alpha=0.7)

    def _style_staff_axes(self, ax, title):
        ax.set_facecolor("#171717")
        ax.set_title(title, color=ORANGE, fontsize=10, loc="left")
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_visible(False)

    def log(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        if hasattr(self, "log_text"):
            self.log_text.configure(state="normal")
            self.log_text.insert("end", f"[{timestamp}] {message}\n")
            self.log_text.see("end")
            self.log_text.configure(state="disabled")

    def _set_status(self, text, color=MUTED):
        self.status_label.configure(text=f"Estado: {text}", fg=color)

    def _reset_results(self):
        self.dsp_done = False
        self.predictions_done = False
        self.results = []
        for item in self.table.get_children():
            self.table.delete(item)
        self.dsp_button.configure(state="normal" if self.audio is not None else "disabled")
        self.predict_button.configure(state="disabled")
        self.report_button.configure(state="disabled")
        self.metrics_label.configure(text="Frecuencia: -- Hz\nNota: --")
        
        self.ax_staff.clear()
        self._style_staff_axes(self.ax_staff, "Pentagrama (Notación Formal)")
        self.canvas.draw_idle()

    def load_wav(self):
        path = filedialog.askopenfilename(
            title="Seleccionar archivo WAV",
            filetypes=[("Archivos WAV", "*.wav"), ("Todos los archivos", "*.*")]
        )
        if not path:
            return
        if os.path.splitext(path)[1].lower() != ".wav":
            messagebox.showerror("Formato no válido", "Selecciona un archivo con extensión .wav.")
            return
        try:
            sample_rate, data = wavfile.read(path)
            if data.size == 0:
                raise ValueError("El archivo no contiene muestras.")
            if data.ndim == 2:
                data = data.astype(np.float64).mean(axis=1)
            else:
                data = data.astype(np.float64)

            raw_rate, raw_data = wavfile.read(path)
            if np.issubdtype(raw_data.dtype, np.integer):
                info = np.iinfo(raw_data.dtype)
                scale = max(abs(info.min), info.max)
                if raw_data.ndim == 2:
                    raw_data = raw_data.astype(np.float64).mean(axis=1)
                data = raw_data.astype(np.float64) / float(scale)
            else:
                if raw_data.ndim == 2:
                    raw_data = raw_data.astype(np.float64).mean(axis=1)
                data = raw_data.astype(np.float64)
                max_abs = np.max(np.abs(data))
                if max_abs > 1.0:
                    data = data / max_abs

            if not np.all(np.isfinite(data)):
                raise ValueError("El archivo contiene valores no válidos.")
            if sample_rate <= 0:
                raise ValueError("Frecuencia de muestreo no válida.")

            self.file_path = path
            self.sample_rate = int(sample_rate)
            self.audio = data
            self.original_audio = data.copy()
            self.file_label.configure(text=os.path.basename(path), fg=WHITE)
            self._reset_results()
            self.dsp_button.configure(state="normal")
            self._set_status("audio cargado correctamente", GREEN)
            self.log(f"Archivo cargado: {path}")
            self.log(f"Frecuencia de muestreo: {self.sample_rate} Hz | Duración: {len(data)/self.sample_rate:.2f} s")
            self._plot_waveform()
        except Exception as exc:
            messagebox.showerror("No se pudo cargar el audio", str(exc))
            self.log(f"Error al cargar WAV: {exc}")

    def simulate_tone(self):
        self.sample_rate = 44100
        duration = 2.0
        t = np.arange(int(self.sample_rate * duration)) / self.sample_rate
        self.audio = 0.7 * np.sin(2 * np.pi * 440.0 * t)
        self.original_audio = self.audio.copy()
        self.file_path = None
        self.file_label.configure(text="Simulación: tono puro de 440 Hz (La4)", fg=GREEN)
        self._reset_results()
        self.dsp_button.configure(state="normal")
        self._set_status("tono de prueba generado", GREEN)
        self.log("Se generó un tono de prueba de 440 Hz (La4), duración 2 segundos.")
        self._plot_waveform()

    def _plot_waveform(self):
        if self.audio is None:
            return
        self.ax_wave.clear()
        self._style_axes(self.ax_wave, "Forma de onda", "Tiempo (s)", "Amplitud")
        duration = len(self.audio) / self.sample_rate
        step = max(1, len(self.audio) // 12000)
        times = np.arange(0, len(self.audio), step) / self.sample_rate
        self.ax_wave.plot(times, self.audio[::step], color=ORANGE, linewidth=0.8)
        self.ax_wave.set_xlim(0, max(duration, 0.01))
        self.figure.tight_layout(pad=2.0)
        self.canvas.draw_idle()

    def play_audio(self):
        if self.audio is None:
            messagebox.showinfo("Sin audio", "Primero carga un archivo WAV o genera la simulación.")
            return
        if sd is None:
            messagebox.showwarning("Reproducción no disponible",
                                   "Instala sounddevice con:\npython -m pip install sounddevice")
            return
        try:
            sd.stop()
            sd.play(self.audio.astype(np.float32), self.sample_rate)
            self.log("Reproducción de audio iniciada.")
        except Exception as exc:
            messagebox.showerror("Error de reproducción", str(exc))
            self.log(f"Error de reproducción: {exc}")

    def run_dsp(self):
        if self.audio is None:
            return
        self.dsp_button.configure(state="disabled")
        self.predict_button.configure(state="disabled")
        self.report_button.configure(state="disabled")
        self.progress.start(10)
        self._set_status("analizando señal...", ORANGE)
        self.log("Iniciando análisis DSP: STFT y espectrograma.")

        def work():
            try:
                nperseg = min(2048, len(self.audio))
                if nperseg < 16:
                    raise ValueError("El audio es demasiado corto para analizarse.")
                noverlap = min(nperseg // 2, nperseg - 1)
                freqs, times, zxx = signal.stft(
                    self.audio, fs=self.sample_rate, window="hamming",
                    nperseg=nperseg, noverlap=noverlap
                )
                magnitude = np.abs(zxx)
                self.root.after(0, lambda: self._finish_dsp(freqs, times, magnitude))
            except Exception as exc:
                self.root.after(0, lambda e=str(exc): self._dsp_error(e))

        threading.Thread(target=work, daemon=True).start()

    def _finish_dsp(self, freqs, times, magnitude):
        self.progress.stop()
        self.ax_spec.clear()
        self._style_axes(self.ax_spec, "Espectrograma (STFT)", "Tiempo (s)", "Frecuencia (Hz)")
        if magnitude.size:
            mask = freqs <= min(5000, self.sample_rate / 2)
            db = 20 * np.log10(np.maximum(magnitude[mask], 1e-7))
            self.ax_spec.pcolormesh(times, freqs[mask], db, shading="auto",
                                    cmap="inferno", vmin=-80, vmax=-10)
            self.ax_spec.set_ylim(0, min(5000, self.sample_rate / 2))
        self.figure.tight_layout(pad=2.0)
        self.canvas.draw_idle()
        self.dsp_done = True
        self.predict_button.configure(state="normal")
        self._set_status("DSP completado", GREEN)
        self.log("DSP terminado correctamente. Espectrograma actualizado.")
        self.log("Puedes ejecutar la detección preliminar de notas.")

    def _dsp_error(self, error):
        self.progress.stop()
        self.dsp_button.configure(state="normal")
        self._set_status("error durante DSP", RED)
        self.log(f"Error DSP: {error}")
        messagebox.showerror("Error de procesamiento", error)

    def run_prediction(self):
        if self.audio is None or not self.dsp_done:
            messagebox.showinfo("Falta un paso", "Primero ejecuta el análisis DSP.")
            return
        self.predict_button.configure(state="disabled")
        self.progress.start(10)
        self._set_status("estimando frecuencias...", ORANGE)
        self.log("Iniciando estimación de frecuencia fundamental por autocorrelación.")

        def work():
            try:
                frame_size = max(1024, int(0.08 * self.sample_rate))
                hop = max(1, int(0.10 * self.sample_rate))
                results = []
                for start in range(0, max(1, len(self.audio) - frame_size + 1), hop):
                    frame = self.audio[start:start + frame_size]
                    if len(frame) < frame_size:
                        break
                    if np.sqrt(np.mean(frame ** 2)) < 0.008:
                        continue
                    freq = estimate_f0_autocorrelation(frame, self.sample_rate)
                    if freq:
                        note, midi = frequency_to_note(freq)
                        results.append({
                            "note": note,
                            "freq": freq,
                            "midi": midi
                        })
                compact = []
                for item in results:
                    if not compact or item["note"] != compact[-1]["note"]:
                        compact.append(item)
                self.root.after(0, lambda: self._finish_prediction(compact))
            except Exception as exc:
                self.root.after(0, lambda e=str(exc): self._prediction_error(e))

        threading.Thread(target=work, daemon=True).start()

    def _finish_prediction(self, results):
        self.progress.stop()
        self.results = results
        for item in self.table.get_children():
            self.table.delete(item)
        for item in results:
            self.table.insert("", "end", values=(
            ))
        if results:
            first = results[0]
            self.metrics_label.configure(
                text=f"Primera nota: {first['note']}\nFrecuencia: {first['freq']:.2f} Hz"
            )
            self._set_status(f"predicción terminada · {len(results)} notas", GREEN)
            self.log(f"Detección terminada. Se identificaron {len(results)} cambios de nota.")
            self._render_staff(results)
        else:
            self.metrics_label.configure(text="Frecuencia: -- Hz\nNota: no identificada")
            self._set_status("no se detectaron notas claras", RED)
            self.log("No se detectaron notas claras. Prueba con un audio más limpio o de mayor volumen.")
            self.ax_staff.clear()
            self._style_staff_axes(self.ax_staff, "Pentagrama (Notación Formal)")
            self.canvas.draw_idle()

        self.predictions_done = True
        self.report_button.configure(state="normal")
        self.predict_button.configure(state="normal")

    def _render_staff(self, results):
        """Genera y renderiza el pentagrama formal usando music21 o Matplotlib como fallback."""
        self.ax_staff.clear()
        self._style_staff_axes(self.ax_staff, "Pentagrama (Notación Formal)")

        if not results:
            self.canvas.draw_idle()
            return

        # Intento de uso de music21 para renderizado nativo
        rendered_music21 = False
        if music21 is not None:
            try:
                s = music21.stream.Stream()
                s.append(music21.clef.TrebleClef())
                s.append(music21.meter.TimeSignature('4/4'))

                for item in results[:12]:  # Limitar a las primeras 12 notas para legibilidad
                    if item.get("midi"):
                        n = music21.note.Note(item["midi"])
                        n.quarterLength = 1.0  # Duración por defecto (negra)
                        s.append(n)

                tmp_dir = tempfile.gettempdir()
                img_path = os.path.join(tmp_dir, "m21_staff.png")
                
                # Probar si el entorno permite escribir PNG directamente
                s.write('musicxml.png', fp=img_path)
                
                if os.path.exists(img_path) or os.path.exists(img_path + ".png"):
                    actual_path = img_path if os.path.exists(img_path) else img_path + ".png"
                    img = mpimg.imread(actual_path)
                    self.ax_staff.imshow(img)
                    self.ax_staff.axis("off")
                    rendered_music21 = True
            except Exception as e:
                self.log(f"Nota: music21 usará el motor gráfico vectorial interno ({e}).")

        # Fallback gráfico directo si LilyPond/MuseScore no están instalados externamente
        if not rendered_music21:
            self._draw_matplotlib_staff(results)

        self.figure.tight_layout(pad=2.0)
        self.canvas.draw_idle()

    def _draw_matplotlib_staff(self, results):
        """Dibujado de pentagrama vectorial optimizado en Matplotlib."""
        ax = self.ax_staff
        ax.clear()
        self._style_staff_axes(ax, "Pentagrama (Notación Formal)")

        # Límite vertical ampliado para dar espacio a etiquetas y alteraciones superiores/inferiores
        ax.set_xlim(0, 10)
        ax.set_ylim(-6, 10)

        # Dibujar las 5 líneas del pentagrama en clave de Sol (E4=2, G4=3, B4=4, D5=5, F5=6)
        staff_lines = [2, 3, 4, 5, 6]
        for y in staff_lines:
            ax.plot([0.5, 9.5], [y, y], color=WHITE, lw=1.2, zorder=1)

        # Barras laterales
        ax.plot([0.5, 0.5], [2, 6], color=WHITE, lw=2, zorder=1)
        ax.plot([9.5, 9.5], [2, 6], color=WHITE, lw=2, zorder=1)

        # Clave de Sol
        ax.text(0.7, 3.2, "𝄞", color=ORANGE, fontsize=36, va="center", ha="center", zorder=2)

        if not results:
            return

        # Distribución horizontal de las notas detectadas
        max_notes = min(len(results), 12)
        x_positions = np.linspace(2.0, 9.0, max_notes)

        for i, item in enumerate(results[:max_notes]):
            midi = item.get("midi")
            if not midi:
                continue

            # Mapeo posicional MIDI relativo a E4 (Mi4 = 64 = Posición 2.0)
            steps_from_e4 = (midi - 64) / 2.0
            y_pos = 2.0 + steps_from_e4
            x = x_positions[i]

            # 1. Líneas adicionales (superiores o inferiores) si la nota cae fuera de las 5 líneas
            if y_pos < 2.0:
                # Líneas adicionales inferiores
                for y_add in np.arange(1.0, y_pos - 0.5, -1.0):
                    ax.plot([x - 0.25, x + 0.25], [y_add, y_add], color="#888888", lw=1.0, zorder=1)
            elif y_pos > 6.0:
                # Líneas adicionales superiores
                for y_add in np.arange(7.0, y_pos + 0.5, 1.0):
                    ax.plot([x - 0.25, x + 0.25], [y_add, y_add], color="#888888", lw=1.0, zorder=1)

            # 2. Cabeza de la nota
            ax.plot(x, y_pos, "o", color=GREEN, markersize=8, zorder=4)

            # 3. Plica (Hacia arriba si está por debajo de B4/4.0, hacia abajo si está por encima)
            stem_dir = 1.8 if y_pos < 4.0 else -1.8
            ax.plot([x + 0.08, x + 0.08], [y_pos, y_pos + stem_dir], color=GREEN, lw=1.5, zorder=3)

            # 4. Alteración (Sostenido ♯) al lado izquierdo de la nota
            if "♯" in item["note"]:
                ax.text(x - 0.25, y_pos, "♯", color=GREEN, fontsize=11, fontweight="bold",
                        va="center", ha="right", zorder=5)

            # 5. Etiqueta de texto de la nota (rotada 30° para evitar colisiones)
            # Ubicada siempre por debajo de la nota con espacio de seguridad
            label_y = min(-2.5, y_pos - 2.5) if y_pos < 2.0 else -2.5
            ax.text(x, label_y, item["note"], color=WHITE, fontsize=8, fontweight="bold",
                    ha="right", va="top", rotation=35,
                    bbox=dict(boxstyle="round,pad=0.15", facecolor=PANEL, edgecolor="none", alpha=0.8),
                    zorder=6)
            
    def _prediction_error(self, error):
        self.progress.stop()
        self.predict_button.configure(state="normal")
        self._set_status("error al detectar notas", RED)
        self.log(f"Error de detección: {error}")
        messagebox.showerror("Error de detección", error)

    def export_pdf(self):
        if not self.predictions_done:
            messagebox.showinfo("Sin resultados", "Primero ejecuta la detección de notas.")
            return
        if SimpleDocTemplate is None:
            messagebox.showerror("Falta una biblioteca",
                                 "Instala reportlab con:\npython -m pip install reportlab")
            return
        path = filedialog.asksaveasfilename(
            title="Guardar reporte",
            defaultextension=".pdf",
            filetypes=[("Documento PDF", "*.pdf")],
            initialfile="reporte_notas_musicales.pdf"
        )
        if not path:
            return
        try:
            doc = SimpleDocTemplate(path, pagesize=letter)
            styles = getSampleStyleSheet()
            story = [
                Paragraph("Sistema Inteligente de Detección de Notas Musicales", styles["Title"]),
                Spacer(1, 10),
                Paragraph(f"Fecha del análisis: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}",
                          styles["Normal"]),
                Paragraph(f"Archivo: {os.path.basename(self.file_path) if self.file_path else 'Simulación 440 Hz'}",
                          styles["Normal"]),
                Paragraph(f"Frecuencia de muestreo: {self.sample_rate} Hz", styles["Normal"]),
                Spacer(1, 14),
                Paragraph("Notas detectadas", styles["Heading2"])
            ]
            for item in self.results:
            if len(data) == 1:
                data.append(["-", "No identificada", "-"])
            table = Table(data, repeatRows=1, colWidths=[110, 180, 150])
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(ORANGE)),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("BACKGROUND", (0, 1), (-1, -1), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]))
            story.extend([table, Spacer(1, 12),
                          Paragraph("Nota: la detección actual es una estimación preliminar "
                                    "por autocorrelación y no representa una clasificación SVM entrenada.",
                                    styles["Italic"])])
            doc.build(story)
            self.log(f"Reporte PDF guardado en: {path}")
            messagebox.showinfo("Reporte generado", f"Se guardó el reporte en:\n{path}")
        except Exception as exc:
            messagebox.showerror("Error al generar PDF", str(exc))
            self.log(f"Error al generar PDF: {exc}")


if __name__ == "__main__":
    root = tk.Tk()
    app = MusicNotesApp(root)
    root.mainloop()