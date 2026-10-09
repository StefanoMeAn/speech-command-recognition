"""Local continuous speech-command recognition dashboard.
Run from repository root: python app.py
The microphone is captured by the Python process on this computer (not by
remote visitors' browsers). Do not expose this app publicly.
"""
from __future__ import annotations
from collections import deque
from datetime import datetime
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event, Lock, Thread
import json
import sys
import os
# Prefer reliable CPU inference for small, streaming batches.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import gradio as gr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
import tensorflow as tf
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from preprocessing import extract_mfcc, extract_stft
RATE = 16000
BLOCK_SECONDS = 0.05
BLOCK_SIZE = int(RATE * BLOCK_SECONDS)
START_RMS = 0.015  # Minimum trigger RMS
END_RMS = 0.010
NOISE_MULTIPLIER = 3.0
START_CONSECUTIVE_BLOCKS = 3
MIN_PEAK_RMS = 0.025
MIN_UTTERANCE_RMS = 0.012
COOLDOWN_SECONDS = 0.30
END_SILENCE_SECONDS = 0.25
MIN_ACTIVE_SECONDS = 0.18
MAX_RECORD_SECONDS = 1.4
PRE_ROLL_SECONDS = 0.15
CONFIDENCE_THRESHOLD = 0.70  # Heuristic, not calibrated
LABELS_PATH = ROOT / 'data' / 'processed' / 'labels.json'
if not LABELS_PATH.exists():
    raise FileNotFoundError(f'Missing labels: {LABELS_PATH}')
LABELS = json.loads(LABELS_PATH.read_text())
MODEL_FILES = {
    'BiLSTM · MFCC': ('bilstm_mfcc.keras', 'mfcc', False),
    'LSTM · MFCC': ('lstm_mfcc.keras', 'mfcc', False),
    'BiLSTM · STFT': ('bilstm_stft.keras', 'stft', False),
    'LSTM · STFT': ('lstm_stft.keras', 'stft', False),
    'CNN · STFT': ('cnn_stft.keras', 'stft', True),
    'CNN · MFCC': ('cnn_mfcc.keras', 'mfcc', True),
}
AVAILABLE = {k: v for k, v in MODEL_FILES.items() if (ROOT / 'models' / v[0]).is_file()}
if not AVAILABLE:
    raise FileNotFoundError('No trained .keras models found in models/')
DEFAULT = 'BiLSTM · MFCC' if 'BiLSTM · MFCC' in AVAILABLE else next(iter(AVAILABLE))
class Engine:
    def __init__(self):
        self.lock = Lock()
        self.stop_event = Event()
        self.thread = None
        self.queue = Queue(maxsize=100)
        self.models = {}
        self.model_name = DEFAULT
        self.status = 'Stopped'
        self.error = ''
        self.history = deque(maxlen=30)
        self.last_prediction = '—'
        self.last_confidence = None
        self.top_predictions = []
        self.level = 0.0
        self.waveform = np.zeros(RATE, dtype=np.float32)
    def get_model(self, name):
        if name not in self.models:
            filename, _, _ = AVAILABLE[name]
            model = tf.keras.models.load_model(ROOT / 'models' / filename)
            if model.output_shape[-1] != len(LABELS):
                raise ValueError(f'Label mismatch for {name}')
            self.models[name] = model
        return self.models[name]
    def classify(self, audio, name):
        _, feature, is_cnn = AVAILABLE[name]
        features = extract_mfcc(audio) if feature == 'mfcc' else extract_stft(audio)
        if not is_cnn:
            features = features[..., 0]
        features = features[None, ...]
        model = self.get_model(name)
        if tuple(model.input_shape[1:]) != tuple(features.shape[1:]):
            raise ValueError(f'Input mismatch: {name}: {features.shape} vs {model.input_shape}')
        scores = model(features, training=False).numpy()[0]
        indices = np.argsort(scores)[::-1][:5]
        return [(LABELS[int(i)], float(scores[int(i)])) for i in indices]
    def callback(self, indata, frames, time_info, status):
        try:
            self.queue.put_nowait(indata[:, 0].copy())
        except Full:
            pass
    def start(self, name):
        with self.lock:
            if self.thread is not None and self.thread.is_alive():
                return 'Already listening. Stop before changing models.'
            self.model_name = name
            self.error = ''
            self.status = 'Starting'
            self.stop_event.clear()
            self.queue = Queue(maxsize=100)
        try:
            self.get_model(name)  # Load before starting microphone thread
        except Exception as exc:
            with self.lock:
                self.status = 'Error'
                self.error = str(exc)
            return f'Model loading failed: {exc}'
        self.thread = Thread(target=self.listen, daemon=True)
        self.thread.start()
        return f'Starting continuous listening with {name}…'
    def stop(self):
        self.stop_event.set()
        with self.lock:
            self.status = 'Stopping' if self.thread and self.thread.is_alive() else 'Stopped'
        return 'Stopping microphone…'
    def clear(self):
        with self.lock:
            self.history.clear()
            self.last_prediction = '—'
            self.last_confidence = None
            self.top_predictions = []
        return 'History cleared.'
    def listen(self):
        pre = deque(maxlen=round(PRE_ROLL_SECONDS / BLOCK_SECONDS))
        recent = deque(maxlen=round(1 / BLOCK_SECONDS))
        recording = []
        speaking = False
        active_blocks = quiet_blocks = trigger_blocks = 0
        peak_rms = 0.0
        noise_floor = 0.003  # Updated only while not recording.
        cooldown_blocks = 0
        end_blocks = round(END_SILENCE_SECONDS / BLOCK_SECONDS)
        max_blocks = round(MAX_RECORD_SECONDS / BLOCK_SECONDS)
        cooldown_length = round(COOLDOWN_SECONDS / BLOCK_SECONDS)

        try:
            with sd.InputStream(samplerate=RATE, channels=1, dtype='float32',
                                blocksize=BLOCK_SIZE, callback=self.callback):
                with self.lock:
                    self.status = 'Listening'
                while not self.stop_event.is_set():
                    try:
                        block = self.queue.get(timeout=0.2)
                    except Empty:
                        continue
                    recent.append(block)
                    rms = float(np.sqrt(np.mean(block ** 2)))
                    with self.lock:
                        self.level = rms
                        self.waveform = np.concatenate(tuple(recent))[-RATE:].copy()

                    if not speaking:
                        pre.append(block)
                        if cooldown_blocks > 0:
                            cooldown_blocks -= 1
                            trigger_blocks = 0
                            continue
                        # Learn background level only from blocks below the
                        # current speech threshold; avoid learning speech as noise.
                        start_threshold = max(START_RMS, NOISE_MULTIPLIER * noise_floor)
                        if rms < start_threshold:
                            noise_floor = 0.98 * noise_floor + 0.02 * rms
                            trigger_blocks = 0
                        else:
                            trigger_blocks += 1
                        if trigger_blocks >= START_CONSECUTIVE_BLOCKS:
                            speaking = True
                            recording = list(pre)
                            active_blocks = trigger_blocks
                            quiet_blocks = 0
                            peak_rms = rms
                            with self.lock:
                                self.status = 'Recording command'
                    else:
                        recording.append(block)
                        peak_rms = max(peak_rms, rms)
                        end_threshold = max(END_RMS, 1.5 * noise_floor)
                        if rms >= end_threshold:
                            active_blocks += 1
                            quiet_blocks = 0
                        else:
                            quiet_blocks += 1
                        if quiet_blocks >= end_blocks or len(recording) >= max_blocks:
                            audio = np.concatenate(recording)
                            utterance_rms = float(np.sqrt(np.mean(audio ** 2)))
                            valid = (
                                active_blocks * BLOCK_SECONDS >= MIN_ACTIVE_SECONDS
                                and peak_rms >= MIN_PEAK_RMS
                                and utterance_rms >= MIN_UTTERANCE_RMS
                            )
                            if valid:
                                with self.lock:
                                    self.status = 'Classifying'
                                top = self.classify(audio, self.model_name)
                                word, confidence = top[0]
                                stamp = datetime.now().strftime('%H:%M:%S')
                                accepted = confidence >= CONFIDENCE_THRESHOLD
                                with self.lock:
                                    self.last_prediction = word.upper() if accepted else 'Uncertain'
                                    self.last_confidence = confidence
                                    self.top_predictions = top
                                    self.history.appendleft([
                                        stamp, word if accepted else f'Uncertain ({word})',
                                        f'{confidence:.1%}', self.model_name,
                                    ])
                            recording = []
                            pre.clear()
                            speaking = False
                            active_blocks = quiet_blocks = trigger_blocks = 0
                            peak_rms = 0.0
                            cooldown_blocks = cooldown_length
                            # Discard old microphone samples accumulated during inference.
                            while True:
                                try:
                                    self.queue.get_nowait()
                                except Empty:
                                    break
                            with self.lock:
                                self.status = 'Listening'
        except Exception as exc:
            with self.lock:
                self.error = str(exc)
                self.status = 'Error'
        finally:
            with self.lock:
                if self.status != 'Error':
                    self.status = 'Stopped'
                self.level = 0.0

    def snapshot(self):
        with self.lock:
            status = self.status
            error = self.error
            word = self.last_prediction
            confidence = self.last_confidence
            top = list(self.top_predictions)
            history = list(self.history)
            level = self.level
            waveform = self.waveform.copy()
        info = f'**Status:** {status}  |  **Mic RMS:** {level:.4f}'
        if error:
            info += f'\n\n**Error:** {error}'
        prediction = f'## {word}\nConfidence: {confidence:.1%}' if confidence is not None else '## Waiting for a command…'
        fig, ax = plt.subplots(figsize=(9, 2.1))
        x = np.arange(len(waveform)) / RATE
        ax.plot(x, waveform, linewidth=0.8)
        ax.set_xlim(0, 1)
        peak = max(float(np.max(np.abs(waveform))), 0.05)
        ax.set_ylim(-peak * 1.2, peak * 1.2)
        ax.set_xlabel('Recent audio (seconds)')
        ax.set_ylabel('Amplitude')
        ax.grid(alpha=0.2)
        fig.tight_layout()
        return info, prediction, {k: v for k, v in top}, history, fig
engine = Engine()
with gr.Blocks(title='Speech Command Recognition · Live') as demo:
    gr.Markdown('# 🎤 Speech Command Recognition\nContinuous local microphone inference · CNN / LSTM / BiLSTM · STFT / MFCC')
    gr.Markdown('**Local demo:** Start listening, say one command at a time, and pause briefly between commands. '
                'The Python process records your computer’s microphone; this is not a remote-browser microphone app.')
    with gr.Row():
        selector = gr.Dropdown(choices=list(AVAILABLE), value=DEFAULT, label='Model (change while stopped)')
        start = gr.Button('▶ Start listening', variant='primary')
        stop = gr.Button('■ Stop')
        clear = gr.Button('Clear history')
    action = gr.Markdown('Ready. Microphone is off.')
    with gr.Row():
        status = gr.Markdown('**Status:** Stopped')
        prediction = gr.Markdown('## Waiting for a command…')
    with gr.Row():
        waveform = gr.Plot(label='Live waveform')
        confidence = gr.Label(label='Top 5 predictions', num_top_classes=5)
    history = gr.Dataframe(headers=['Time', 'Prediction', 'Confidence', 'Model'],
                           datatype=['str'] * 4, interactive=False, label='Command history')
    gr.Markdown('**Limitations:** This uses an adaptive energy-based speech detector (not a trained VAD). '
                'The 70% threshold is heuristic, not a calibrated unknown-word detector. '
                'Silence, noise, and out-of-vocabulary speech can still trigger predictions.')
    start.click(engine.start, inputs=selector, outputs=action, concurrency_limit=1)
    stop.click(engine.stop, outputs=action)
    clear.click(engine.clear, outputs=action)
    timer = gr.Timer(value=0.7)
    timer.tick(engine.snapshot, outputs=[status, prediction, confidence, history, waveform],
               concurrency_limit=1, show_progress='hidden')
if __name__ == '__main__':
    demo.launch(server_name='127.0.0.1', share=False)
