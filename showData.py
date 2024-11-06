import wave
import numpy as np
import matplotlib.pyplot as plt

# file pat to open
fp = 'data/voice.wav'
wav = wave.open(fp, 'r')

# get params
frames = wav.readframes(-1)
signal = np.frombuffer(frames, dtype='int16')
fs = wav.getframerate()
time = np.linspace(0, len(signal)/fs, num=len(signal))

# plot
plt.figure()
plt.plot(time, signal)
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.title('Waveform')

plt.show()
wav.close()