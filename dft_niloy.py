import numpy as np
import matplotlib.pyplot as plt

def _dft(x):
    N = len(x)
    X = np.zeros(N, dtype=complex)
    for m in range(N):
        for n in range(N):
            X[m] += x[n] * np.exp(-2j * np.pi * m * n / N)
    return X

def _inverse_dft(X):
    N = len(X)
    x = np.zeros(N, dtype=complex)
    for n in range(N):
        for m in range(N):
            x[n] += X[m] * np.exp(2j * np.pi * m * n / N)
        x[n] /= N
    return x

# Define parameters
fs = 8000  # Sampling frequency in Hz
N = 8      # Number of points in DFT
T = 1/fs   # Sampling period
f1 = 1000  # Frequency of the first sine wave in Hz
f2 = 2000  # Frequency of the second sine wave in Hz
phase_shift = 3*np.pi/4  # Phase shift of the second sine wave in radians

# Time vector for N samples
n = np.arange(N)
print("n : " , n)

# Define the input signal x(nT)
x_n = np.sin(2 * np.pi * f1 * n * T) + 0.5 * np.sin(2 * np.pi * f2 * n * T + phase_shift)

# Compute the 8-point DFT
# X_k = np.fft.fft(x_n, N)
X_k = _dft(x_n)

# Print the results
print("Input samples x(n):")
print(x_n)
print("\n8-point DFT X(k):")
print(X_k, len(X_k));
phase = np.angle(X_k)
print("phase: ", phase)

inv_x = _inverse_dft(X_k)
print("inv:", inv_x)

# Plot the results
plt.figure(figsize=(12, 6))

plt.subplot(4, 1, 1)
plt.plot(n, x_n)
plt.title('Input Samples x(n)')
plt.xlabel('Sample index n')
plt.ylabel('Amplitude')

plt.subplot(4, 1, 2)
k = np.arange(N)
plt.stem(k, np.abs(X_k))
plt.title('DFT Magnitude |X(k)|')
plt.xlabel('Frequency bin k')
plt.ylabel('Magnitude')

plt.subplot(4, 1, 3)
plt.stem(k, phase)
plt.title('DFT phase')

plt.subplot(4, 1, 4)
_in = np.arange(N)
plt.plot(_in, inv_x)
plt.title('Inverse DFT x(n)')
# plt.xlabel('Frequency bin k')
plt.ylabel('Amplitude')

plt.tight_layout()
plt.show()
