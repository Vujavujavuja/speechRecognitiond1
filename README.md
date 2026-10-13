# Speech Recognition, part 1: looking at audio signals

The first exercise in a speech recognition course: load an audio file and inspect it in the time and frequency domains before doing any recognition.

`showData.py` loads a recording with librosa and plots:

- **Waveform** with the RMS energy per frame, the amplitude envelope (Hilbert transform), the average amplitude, and the global minimum and maximum
- **Speech segments**, marked with a simple energy threshold: green lines where speech starts, red lines where it ends
- **Zero crossing rate** over time, computed by hand
- **Spectrogram** (log-frequency STFT) with the fundamental frequency overlaid, estimated with pYIN

## Running it

```bash
pip install -r requirements.txt
python showData.py
```

To analyse a different file, change `fp` at the top of the script. `data/` holds sample recordings to try: speech, single instruments, scales and noise.

Part 2, isolated word recognition with MFCCs and dynamic time warping, is in [speechRecognitiond2](https://github.com/Vujavujavuja/speechRecognitiond2).
