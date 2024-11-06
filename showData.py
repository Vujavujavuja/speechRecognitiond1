import wave
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from scipy.signal import find_peaks

# file pat to open
fp = 'data/ted.wav'

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

analytic_signal = hilbert(signal)
amplitude_envelope = np.abs(analytic_signal)

plt.figure(figsize=(15, 5))

# show bcs plot is deprecated
librosa.display.waveshow(signal, sr=sample_rate, alpha=0.5)

plt.plot(rms_time, rms, color='r', label='RMS', linewidth=0.5)
peaks, _ = find_peaks(signal)
valleys, _ = find_peaks(-signal)

# min max from signal array
max_amp = np.max(signal)
min_amp = np.min(signal)
min_abs_amp = np.min(np.abs(signal))

# when is min and max
max_time = np.argmax(signal) / sample_rate
min_time = np.argmin(signal) / sample_rate
min_abs_time = np.argmin(np.abs(signal)) / sample_rate


avg_amp = np.mean(np.abs(signal))
# speech detection
flag = 0
for i in range(len(rms)):
    if rms[i] > avg_amp and flag == 0:
        plt.axvline(i * frame_size / sample_rate, color='g', linewidth=0.5)
        flag = 1
    elif rms[i] < 0.005 and flag == 1:
        plt.axvline(i * frame_size / sample_rate, color='r', linewidth=0.5)
        flag = 0

# plt.plot(peaks / sample_rate, signal[peaks], 'ro', label='Detected Peaks', markersize=5)
# plt.plot(valleys / sample_rate, signal[valleys], 'yo', label='Detected Valleys', markersize=5)

plt.plot(max_time, max_amp, 'go', label='Max', markersize=3)
plt.plot(min_time, min_amp, 'bo', label='Min', markersize=3)

# plt.plot(min_abs_time, min_abs_amp, 'ro', label='Min Abs')

print('Max Amplitude:', max_amp)
print('Max Time:', max_time)
print('Max Index:', np.argmax(signal))

start = max(0, np.argmax(signal) - frame_size)
end = min(len(signal), np.argmax(signal) + frame_size)

print("Maximum value in isolated window:", np.max(signal[start:end]))
overall_max = np.max(signal)
print("Overall Maximum Amplitude in Waveform:", overall_max)


plt.title('Waveform')
plt.xlabel('Time')

# average amplitude
print('Average Amplitude:', avg_amp)
plt.axhline(avg_amp, color='purple', linestyle='--', label='Average Amplitude', linewidth=0.5)
plt.axhline(-avg_amp, color='purple', linestyle='--', linewidth=0.5)

plt.legend()

plt.show()

# ZCR
def calculate_zcr(signal_zcr, frame_length, hop_length):
    # Calculate ZCR for each frame
    z = [
        np.mean(np.abs(np.diff(np.sign(signal[i:i+frame_length])))) / 2
        for i in range(0, len(signal) - frame_length + 1, hop_length)
    ]
    return np.array(z)

zcr = calculate_zcr(signal, 256, 64)
t = librosa.frames_to_time(np.arange(len(zcr)), sr=sample_rate, hop_length=64)

plt.figure(figsize=(15, 5))
plt.plot(t, zcr, label='ZCR', color='r')
plt.title('Zero Crossing Rate')
plt.xlabel('Time')
plt.legend()
plt.show()
