import numpy as np
import matplotlib.pyplot as plt
import time

def DFT(x):
    start_time = time.perf_counter_ns()
    N = len(x)
    n = np.arange(N)
    k = n.reshape((N, 1))
    e = x * np.exp(-2j * np.pi * k * n / N)
    DFT_result = np.dot(x, e)
    DFT_result = np.abs(np.real(DFT_result) + np.imag(DFT_result))

    execution_time = time.perf_counter_ns() - start_time
    print("DFT TIME: ", execution_time / 1000, " ms")
    return DFT_result

def FFT(x):
    start_time = time.perf_counter_ns()

    FFT_result = np.fft.fft(x)

    execution_time = time.perf_counter_ns() - start_time
    print("FFT TIME: ", execution_time / 1000, " ms")
    return FFT_result

def generate_data():
    t = np.arange(0, 1, 0.01)

    freq = 5
    x = 0.5 * np.sin(2 * np.pi * freq * t)
    freq = 15
    x += 2 * np.sin(2 * np.pi * freq * t)
    freq = 25
    x += 3 * np.sin(2 * np.pi * freq * t)

    DFT_signal = DFT(x)
    FFT_signal = FFT(x)

    plt.specgram(x, NFFT=64, noverlap=32)  # Zmniejszono NFFT i dostosowano noverlap
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title("Spectrogram")

    plt.show()

generate_data()
