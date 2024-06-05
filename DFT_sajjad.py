import numpy as np
import matplotlib.pyplot as plt

def _dft(x):
    N = len(x)
    X = np.zeros(N, dtype=complex)
    for m in range(N):
        for n in range(N):
            X[m] += x[n] * np.exp(-2j * np.pi * m * n / N)
    return X

def _inverse_dft(x):
    N = len(x)
    X = np.zeros(N, dtype=complex)
    for m in range(N):
        for n in range(N):
            X[m] += x[n] * np.exp(2j * np.pi * m * n / N)
            X[m] /= N
    return X

fs = 8000
N = 8
T = 1/fs
f1 = 1000
f2 = 2000
phase_shift = 3*np.pi/4

n = np.arange(N)
print("n : " , n)

x_n = np.sin(2 * np.pi * f1 * n * T) + 0.5 * np.sin(2 * np.pi * f2 * n * T + phase_shift)

X_k = _dft(x_n)
print("Input samples x(n): ")
print(x_n)
print("\n8-point DFT X(k): ")
print(X_k, len(X_k))

phase = np.angle(X_k)
print("phase: ", phase)

in_x = _inverse_dft(X_k)
print("inv: ", in_x)

plt.figure(figsize=(12, 6))

plt.subplot(4, 1, 1)
plt.plot(n, x_n)
plt.title('Input Samples x(n)')
plt.xlabel('Sample index n')    
plt.ylabel('Amplitude')

plt.subplot(4, 1, 2)
k = np.arange(N)
plt.stem(k, np.abs(X_k), 'r')
plt.title('Magnitude of the DFT X(k)')
plt.xlabel('Frequency bin k')
plt.ylabel('Magnitude')

plt.subplot(4, 1, 3)
plt.stem(k, phase, 'g')
plt.title('Phase of the DFT X(k)')
plt.xlabel('Frequency bin k')
plt.ylabel('Phase (radians)')

plt.subplot(4, 1, 4)
plt.plot(n, in_x)
plt.title('Inverse DFT x(n)')
plt.xlabel('Sample index n')
plt.ylabel('Amplitude')


plt.tight_layout()
plt.show()
