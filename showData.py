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

# min max from signal array
max_amp = np.max(signal)
min_amp = np.min(signal)
min_abs_amp = np.min(np.abs(signal))

# when is min and max
max_time = np.argmax(signal) / sample_rate
min_time = np.argmin(signal) / sample_rate
min_abs_time = np.argmin(np.abs(signal)) / sample_rate

plt.plot(max_time, max_amp, 'go', label='Max', markersize=3)
plt.plot(min_time, min_amp, 'bo', label='Min', markersize=3)
# plt.plot(min_abs_time, min_abs_amp, 'ro', label='Min Abs')

plt.title('Waveform')
plt.xlabel('Time')

# average amplitude
avg_amp = np.mean(np.abs(signal))
print('Average Amplitude:', avg_amp)
plt.axhline(avg_amp, color='purple', linestyle='--', label='Average Amplitude', linewidth=0.5)
plt.axhline(-avg_amp, color='purple', linestyle='--', linewidth=0.5)

plt.legend()

plt.show()
