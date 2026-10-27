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
    # Lógica simplificada de autocorrelación (se asume que existe en tu código base)
    corr = signal.correlate(frame, frame, mode='full')
    corr = corr[len(corr)//2:]
    peaks, _ = signal.find_peaks(corr)
    if len(peaks) > 0:
        return sample_rate / peaks[0]
    return None

def frequency_to_note(freq):
    # Mapeo básico de frecuencia a nota MIDI (se asume existencia en tu código)
    if freq <= 0: return "Desconocida", 0
    midi = int(round(69 + 12 * np.log2(freq / 440.0)))
    notes = ['Do', 'Do♯', 'Re', 'Re♯', 'Mi', 'Fa', 'Fa♯', 'Sol', 'Sol♯', 'La', 'La♯', 'Si']
    note = f"{notes[midi % 12]}{midi // 12 - 1}"
    return note, midi

def run_ml_prediction(audio, sample_rate):
    """Estima las frecuencias y detecta las notas usando autocorrelación."""
    frame_size = max(1024, int(0.08 * sample_rate))
    hop = max(1, int(0.10 * sample_rate))
    results = []
    
    for start in range(0, max(1, len(audio) - frame_size + 1), hop):
        frame = audio[start:start + frame_size]
        if len(frame) < frame_size or np.sqrt(np.mean(frame ** 2)) < 0.008:
            continue
            
        freq = estimate_f0_autocorrelation(frame, sample_rate)
        if freq:
            note, midi = frequency_to_note(freq)
            results.append({
                "time": int(start * 1_000_000_000 / sample_rate),
                "note": note,
                "freq": freq,
                "midi": midi
            })
            
    # Compactar notas repetidas continuas
    compact = []
    for item in results:
        if not compact or item["note"] != compact[-1]["note"]:
            compact.append(item)
            
    return compact

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
                  Paragraph("Nota: la detección actual es una estimación preliminar por autocorrelación y no representa una clasificación SVM entrenada.", styles["Italic"])])
    
    doc.build(story)