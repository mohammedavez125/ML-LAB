import numpy as np
from scipy.fft import fft
import matplotlib.pyplot as plt

# Create a sample signal
t = np.linspace(0, 1, 400)
signal = np.sin(2 * np.pi * 50 * t) + np.sin(2 * np.pi * 120 * t)

# Compute the Fourier Transform
fft_signal = fft(signal)

# Compute the frequencies
frequencies = np.fft.fftfreq(len(fft_signal))

# Plot the signal and its Fourier Transform
plt.subplot(2, 1, 1)
plt.plot(t, signal)
plt.title('Original Signal')
plt.subplot(2, 1, 2)
plt.plot(frequencies, np.abs(fft_signal))
plt.title('Fourier Transform')
plt.show()
