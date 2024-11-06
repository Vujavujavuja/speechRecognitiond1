import wave
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

# file pat to open
fp = 'data/debussy.wav'

# signal and sample rate same as raw
signal, sample_rate = librosa.load(fp, sr=None)

# frame size
frame_size = 256

# rms square root of mean of square
rms = [
    np.sqrt(np.mean(signal[i:i+frame_size]**2))
    for i in range(0, len(signal), frame_size)
]

# rms time
rms_time = [
    (i + frame_size / 2) / sample_rate
    for i in range(0, len(signal), frame_size)
]

plt.figure(figsize=(15, 4))

# show bcs plot is deprecated
librosa.display.waveshow(signal, sr=sample_rate, alpha=0.5)

plt.plot(rms_time, rms, color='r', label='RMS', linewidth=0.5)

plt.title('Waveform')
plt.xlabel('Time')
plt.show()