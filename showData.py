import wave
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

# file pat to open
fp = 'data/debussy.wav'

signal, sample_rate = librosa.load(fp, sr=None)

plt.figure(figsize=(15, 4))

# show bcs plot is deprecated
librosa.display.waveshow(signal, sr=sample_rate, alpha=0.5)

plt.title('Waveform')
plt.xlabel('Time')
plt.show()