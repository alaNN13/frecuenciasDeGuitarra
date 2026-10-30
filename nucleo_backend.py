"""
Módulo Backend: Procesamiento Digital y Lógica de Negocio (backend_core.py)
Contiene la Ingesta, el DSP (Fourier), la Predicción (Autocorrelación/SVM) y la Generación de PDF.
"""
import os
import numpy as np
from datetime import datetime
from scipy.io import wavfile
from scipy import signal

# Librerías para generación de PDF
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

# ==========================================
# 1. MÓDULO DE INGESTA DE DATOS
# ==========================================
def load_wav(path):
    """Lee un archivo WAV, lo convierte a monofónico y lo normaliza."""
    sample_rate, data = wavfile.read(path)
    
    # Conversión a monofónico promediando canales si es estéreo
    if data.ndim == 2:
        data = data.astype(np.float64).mean(axis=1)
    else:
        data = data.astype(np.float64)
        
    # Normalización de la señal en el rango [-1, 1] para evitar saturaciones
    max_abs = np.max(np.abs(data))
    if max_abs > 1.0:
        data = data / max_abs
        
    return int(sample_rate), data

def simulate_tone():
    """Genera un tono puro de prueba de 440 Hz (La4) en memoria."""
    sample_rate = 44100
    duration = 2.0
    t = np.arange(int(sample_rate * duration)) / sample_rate
    audio = 0.7 * np.sin(2 * np.pi * 440.0 * t)
    return sample_rate, audio

# ==========================================
# 2. MÓDULO DE PREPROCESAMIENTO (DSP)
# ==========================================
def run_dsp(audio, sample_rate):
    """Aplica la Transformada de Fourier de Tiempo Corto (STFT)."""
    nperseg = min(2048, len(audio))
    if nperseg < 16:
        raise ValueError("El audio es demasiado corto para analizarse.")
    
    noverlap = min(nperseg // 2, nperseg - 1)
    
    # Uso de ventana Hamming con 2048 muestras y 50% de solapamiento
    freqs, times, zxx = signal.stft(
        audio, fs=sample_rate, window="hamming",
        nperseg=nperseg, noverlap=noverlap
    )
    magnitude = np.abs(zxx)
    return freqs, times, magnitude

# ==========================================
# 3. MÓDULO DE PREDICCIÓN (IA PRELIMINAR)
# ==========================================
# Nota: Esta es una función auxiliar de tu código original. 
# En el futuro será reemplazada por el modelo SVM.
def estimate_f0_autocorrelation(frame, sample_rate):
    """
    Análisis crudo original: primer pico de la autocorrelación.
    Se conserva sin modificar para las escalas de tiempo s y ns, donde se muestra tal cual.
    """
    corr = signal.correlate(frame, frame, mode='full')
    corr = corr[len(corr)//2:]
    peaks, _ = signal.find_peaks(corr)
    if len(peaks) > 0:
        return sample_rate / peaks[0]
    return None

def estimate_f0_yin(frame, sample_rate, fmin=60.0, fmax=1500.0, threshold=0.15):
    """
    Estima la frecuencia fundamental (f0) de un fotograma con el método YIN
    (diferencia cuadrática normalizada, de la familia de la autocorrelación).

    A diferencia de tomar el primer pico de la autocorrelación, YIN busca el periodo
    más corto cuya diferencia normalizada cae bajo `threshold`, lo que evita los
    errores de octava (ej. detectar Si4 cuando se toca Si3) y los picos espurios.
    Devuelve None si el fotograma no es lo bastante periódico (ruido / silencio).
    """
    x = np.asarray(frame, dtype=np.float64)
    x = x - x.mean()
    n = len(x)

    tau_min = max(2, int(sample_rate / fmax))
    tau_max = min(int(sample_rate / fmin), n // 2)
    if tau_max <= tau_min + 2:
        return None
    w = n - tau_max                       # ventana de integración

    # d(tau) = sum_j (x_j - x_{j+tau})^2, calculada con correlación cruzada por FFT
    nfft = 1 << int(np.ceil(np.log2(n + tau_max)))
    cross = np.fft.irfft(np.conj(np.fft.rfft(x[:w], nfft)) * np.fft.rfft(x, nfft), nfft)[:tau_max + 1]
    csum = np.concatenate(([0.0], np.cumsum(x ** 2)))
    taus = np.arange(tau_max + 1)
    d = csum[w] + (csum[w + taus] - csum[taus]) - 2.0 * cross
    d[0] = 0.0

    # Diferencia media acumulada normalizada
    cmnd = np.ones(tau_max + 1)
    cmnd[1:] = d[1:] * taus[1:] / np.maximum(np.cumsum(d[1:]), 1e-12)

    # Primer valle por debajo del umbral (el periodo más corto = fundamental)
    tau = None
    for t in range(tau_min, tau_max):
        if cmnd[t] < threshold:
            while t + 1 < tau_max and cmnd[t + 1] < cmnd[t]:
                t += 1
            tau = t
            break
    if tau is None:                        # sin valle claro: mínimo global si es razonable
        t = int(np.argmin(cmnd[tau_min:tau_max])) + tau_min
        if cmnd[t] > 0.4:
            return None
        tau = t

    # Interpolación parabólica para afinar el periodo
    tau_f = float(tau)
    if 1 < tau < tau_max - 1:
        a0, b0, c0 = cmnd[tau - 1], cmnd[tau], cmnd[tau + 1]
        den = a0 - 2.0 * b0 + c0
        if den != 0:
            tau_f = tau + 0.5 * (a0 - c0) / den
    return sample_rate / tau_f

def frequency_to_note(freq):
    # Mapeo básico de frecuencia a nota MIDI (se asume existencia en tu código)
    if freq <= 0: return "Desconocida", 0
    midi = int(round(69 + 12 * np.log2(freq / 440.0)))
    notes = ['Do', 'Do♯', 'Re', 'Re♯', 'Mi', 'Fa', 'Fa♯', 'Sol', 'Sol♯', 'La', 'La♯', 'Si']
    note = f"{notes[midi % 12]}{midi // 12 - 1}"
    return note, midi

# Parámetros de análisis por fotogramas (compartidos con el agrupador de notas)
FRAME_SEC = 0.08          # duración de cada fotograma analizado
HOP_SEC = 0.10            # salto entre fotogramas
MIN_FRAMES_NOTA = 2       # un fotograma aislado se considera ruido, no una nota tocada


def run_ml_prediction(audio, sample_rate, method="autocorr"):
    """
    Estima las frecuencias y detecta las notas por fotograma.

    method="autocorr": análisis crudo original (primer pico de la autocorrelación).
    method="yin":      estimador YIN, usado para las escalas ms y ms - notas puntuales.
    """
    if method not in ("autocorr", "yin"):
        raise ValueError("method debe ser 'autocorr' o 'yin'")
    estimator = estimate_f0_yin if method == "yin" else estimate_f0_autocorrelation
    frame_size = max(1024, int(FRAME_SEC * sample_rate))
    hop = max(1, int(HOP_SEC * sample_rate))
    results = []
    
    for start in range(0, max(1, len(audio) - frame_size + 1), hop):
        frame = audio[start:start + frame_size]
        if len(frame) < frame_size or np.sqrt(np.mean(frame ** 2)) < 0.008:
            continue
            
        freq = estimator(frame, sample_rate)
        if freq:
            note, midi = frequency_to_note(freq)
            results.append({
                "time": int(start * 1_000_000_000 / sample_rate),
                "note": note,
                "freq": freq,
                "midi": midi
            })
            
            
    return results

def group_played_notes(results, sample_rate, min_frames=MIN_FRAMES_NOTA):
    """
    Agrupa las detecciones por fotograma en notas realmente tocadas (segmentos).

    - Fotogramas consecutivos con la misma nota (MIDI) forman un segmento.
    - Un segmento con menos de `min_frames` fotogramas se descarta (ruido / error de octava).
    - Si al descartar ruido quedan dos segmentos de la misma nota separados solo por ese
      ruido, se fusionan en uno (la nota sostenida no se parte en dos).
    - Una pausa (fotogramas silenciosos) separa dos segmentos aunque sean la misma nota.

    Devuelve una lista de dicts: start_ns, end_ns, midi, note, freq (mediana), frames.
    """
    if not results:
        return []

    frame_len = max(1024, int(FRAME_SEC * sample_rate))
    hop_len = max(1, int(HOP_SEC * sample_rate))
    frame_ns = int(frame_len * 1_000_000_000 / sample_rate)
    max_gap = int(hop_len * 1_000_000_000 / sample_rate * 1.5)

    # 1) Corridas de fotogramas contiguos con la misma nota
    runs = [[results[0]]]
    for prev, item in zip(results, results[1:]):
        if item["midi"] == prev["midi"] and (item["time"] - prev["time"]) <= max_gap:
            runs[-1].append(item)
        else:
            runs.append([item])

    # 2) Descartar ruido y fusionar la misma nota separada solo por ruido
    merged = []   # lista de (indice_de_corrida_original, corrida)
    for idx, run in enumerate(runs):
        if len(run) < min_frames:
            continue
        if merged:
            last_idx, last_run = merged[-1]
            dropped_between = idx - last_idx - 1
            gap = run[0]["time"] - last_run[-1]["time"]
            if (dropped_between > 0
                    and run[0]["midi"] == last_run[0]["midi"]
                    and gap <= (dropped_between + 1) * max_gap):
                merged[-1] = (idx, last_run + run)
                continue
        merged.append((idx, run))

    # 3) Construir segmentos
    segments = []
    for _, run in merged:
        segments.append({
            "start_ns": run[0]["time"],
            "end_ns": run[-1]["time"] + frame_ns,
            "midi": run[0]["midi"],
            "note": run[0]["note"],
            "freq": float(np.median([r["freq"] for r in run])),
            "frames": len(run),
        })
    return segments

# ==========================================
# 4. MÓDULO DE GENERACIÓN DE REPORTES
# ==========================================
def generate_pdf_report(path, results, file_path, sample_rate):
    """Empaca la transcripción en un archivo PDF usando ReportLab."""
    doc = SimpleDocTemplate(path, pagesize=letter)
    styles = getSampleStyleSheet()
    
    filename = os.path.basename(file_path) if file_path else 'Simulación 440 Hz'
    
    story = [
        Paragraph("Sistema Inteligente de Detección de Notas Musicales", styles["Title"]),
        Spacer(1, 10),
        Paragraph(f"Fecha del análisis: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}", styles["Normal"]),
        Paragraph(f"Archivo: {filename}", styles["Normal"]),
        Paragraph(f"Frecuencia de muestreo: {sample_rate} Hz", styles["Normal"]),
        Spacer(1, 14),
        Paragraph("Notas detectadas", styles["Heading2"])
    ]
    
    data = [["Tiempo (ns)", "Nota estimada", "Frecuencia (Hz)"]]
    for item in results:
        data.append([str(item["time"]), item["note"], f"{item['freq']:.2f}"])
        
    if len(data) == 1:
        data.append(["-", "No identificada", "-"])
        
    table = Table(data, repeatRows=1, colWidths=[110, 180, 150])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F57C00")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("BACKGROUND", (0, 1), (-1, -1), colors.whitesmoke),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    
    story.extend([table, Spacer(1, 12),
                  Paragraph("Nota: la detección actual es una estimación preliminar por autocorrelación (crudo en s/ns, YIN en ms) y no representa una clasificación SVM entrenada.", styles["Italic"])])
    
    doc.build(story)