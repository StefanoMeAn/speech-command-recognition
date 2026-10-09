# Speech Command Recognition with CNNs and Recurrent Neural Networks

I compared convolutional and recurrent neural networks for classifying **35 spoken commands** from the [Google Speech Commands v0.02 dataset](https://www.tensorflow.org/datasets/catalog/speech_commands). The experiments use Short-Time Fourier Transform (STFT) and Mel-Frequency Cepstral Coefficient (MFCC) representations, and the repository includes local microphone demos.

The best configuration in these experiments was **BiLSTM + MFCC**, reaching **91.79% test accuracy** and approximately **0.912 macro F1-score**.

## Project Overview

I investigated how the audio representation and network architecture affect isolated-word recognition. I trained and evaluated six configurations using consistent dataset partitions:

- CNN with STFT and MFCC inputs
- LSTM with STFT and MFCC sequences
- Bidirectional LSTM (BiLSTM) with STFT and MFCC sequences

The project also includes a terminal demo and a Gradio dashboard for trying trained models with a local microphone.

## Dataset

The experiments use **Google Speech Commands v0.02**, which contains 35 word classes. The notebooks use the dataset’s official validation and testing lists to create fixed training, validation, and test partitions. The original audio dataset is not included in the repository.

Audio is processed at **16 kHz** and standardized to **one-second** inputs by padding or trimming. The feature extraction code is in `src/preprocessing.py`.

| Representation | Feature shape | Description |
|---|---|---|
| STFT | `(124, 129, 1)` | Time-frequency magnitude representation |
| MFCC | `(99, 13, 1)` | Compact cepstral representation of the audio spectrum |

CNN inputs are resized to **32 × 32** feature maps in the CNN training pipeline. Recurrent networks process the extracted features as sequences, without the trailing channel dimension.

The exploration notebook illustrates background-noise augmentation. This does not mean augmentation was enabled in every training run; see the individual notebooks for each experiment’s settings.

## Methodology

I compared three architectures with two audio representations:

| Architecture | Input representation |
|---|---|
| CNN | STFT and MFCC feature maps |
| LSTM | STFT and MFCC sequences |
| BiLSTM | STFT and MFCC sequences |

The CNN training pipeline resizes features to 32 × 32. The recurrent models use temporal feature sequences. Shared preprocessing is implemented in `src/preprocessing.py`, and the training workflows are in the notebooks.

## Results

| Architecture | Features | Test accuracy |
|---|---|---:|
| CNN | STFT | 82.24% |
| CNN | MFCC | 81.45% |
| LSTM | STFT | 86.02% |
| LSTM | MFCC | 90.32% |
| BiLSTM | STFT | 89.46% |
| **BiLSTM** | **MFCC** | **91.79%** |

BiLSTM + MFCC performed best in this comparison. The recurrent models generally outperformed the CNN baselines, and MFCC features worked particularly well with the LSTM and BiLSTM architectures. These observations apply to this dataset and experimental setup; they do not establish a universally best architecture.

Detailed classification reports, confusion matrices, and training histories are available in [`results/cnn/`](results/cnn/) and [`results/rnn/`](results/rnn/).

![BiLSTM + MFCC confusion matrix](results/rnn/bilstm_mfcc_confusion_matrix.png)

![Recurrent model training curves](results/rnn/validation_curves.png)

## Repository Structure

```text
.
├── app.py                         # Local Gradio microphone dashboard
├── demo.py                        # Single-command CNN + STFT terminal demo
├── environment.yml
├── requirements.txt
├── src/
│   └── preprocessing.py           # Shared STFT and MFCC extraction
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_cnn_training.ipynb
│   └── 03_rnn_training.ipynb
├── data/
│   └── processed/                 # Manifest, labels, and summary CSVs
├── models/                        # Saved Keras models and label metadata
└── results/
    ├── cnn/                       # CNN metrics and confusion matrices
    └── rnn/                       # Recurrent metrics and figures
```

## Installation

Python **3.11** is recommended. From the repository root, create the Conda environment:

```bash
conda env create -f environment.yml
conda activate speech-command-recognition
```

Alternatively, install the Python packages in an existing Python 3.11 environment:

```bash
pip install -r requirements.txt
```

The microphone demos need a working audio input device and the system audio libraries required by `sounddevice`. On Ubuntu, PortAudio may need to be installed through the package manager. Ensure `requirements.txt` includes `gradio`, `sounddevice`, and `scikit-learn` (used in evaluation notebooks). CPU-only TensorFlow is sufficient for the demos; GPU support is optional.

## Usage

### Gradio dashboard

```bash
python app.py
```

Open the local URL printed in the terminal, usually `http://127.0.0.1:7860`. Select a model, click **Start listening**, and say one command at a time with a short pause between words. The dashboard displays a recent waveform, the top five predictions, and a history of detected commands. The microphone is captured by the Python process on the same computer running `app.py`; the app is intended for local use.

### Terminal demo

```bash
python demo.py
```

Press **Enter** to record one second of audio. This demo uses the saved **CNN + STFT** model and prints its top predictions.

### Reproducing the experiments

1. Download and extract [Speech Commands v0.02](https://download.tensorflow.org/data/speech_commands_v0.02.tar.gz) to `data/raw/speech_commands_v0.02/`.
2. Run `notebooks/01_data_exploration.ipynb` to explore the data and create the dataset manifest.
3. Run `notebooks/02_cnn_training.ipynb` for the CNN experiments.
4. Run `notebooks/03_rnn_training.ipynb` for the LSTM/BiLSTM experiments and model comparison.

Training times and exact metrics can vary with hardware and environment. Saved `.keras` files allow the demos to run without retraining.

## Limitations and Future Work

- Offline test accuracy is not live microphone accuracy; the reported metrics come from held-out dataset recordings.
- The Gradio app uses an energy-based speech detector, not a trained voice-activity detector. Background noise and silence can still trigger classifications.
- The live app’s **70% confidence threshold** is heuristic and is not a calibrated rejection mechanism for unknown words.
- The BiLSTM processes a buffered command; it is not a frame-by-frame streaming speech recognition model.

Potential improvements include a dedicated voice-activity detector, more robust noise handling, and evaluation on recordings from different microphones and environments.

## Technologies

Python · TensorFlow / Keras · NumPy · SciPy · `python_speech_features` · scikit-learn · Matplotlib · Gradio · `sounddevice`

## Author

**Stefano Meza** — Physicist with a Master’s degree in Physics of Data from the University of Padova, interested in machine learning, deep learning, computer vision, NLP, and scientific computing.

[LinkedIn](https://www.linkedin.com/in/stefanomean/) · [GitHub](https://github.com/StefanoMeAn)
