import numpy as np
import matplotlib.pyplot as plt

file_path = 'data1.raw.bin'

complex_data = np.fromfile(file_path, dtype=np.int8)
print(complex_data[1:100])

num_samples_per_chirp = 128
num_chirps = len(complex_data) // num_samples_per_chirp

complex_data = complex_data[:num_chirps * num_samples_per_chirp]
complex_data = complex_data.reshape((num_chirps, num_samples_per_chirp))

range_fft= np.fft.fft(complex_data, axis=1)
print(range_fft[1:10])
#next extract the vital sign code is test1,2
#test vital sign for up and downs and validate
#then convert to heart rate and breath rate(study the paper)

if np.any(np.isnan(range_fft)):
    replacement_value = np.complex128(1e+298 - 1e+298j)
    range_fft[np.isnan(range_fft)] = replacement_value
    range_fft[np.isinf(range_fft)] = replacement_value

print(np.any(np.isinf(range_fft)))
print(np.any(np.isnan(range_fft)))

scaled_data = range_fft /np.median(range_fft)

print(np.any(np.isinf(scaled_data)))
print(np.any(np.isnan(scaled_data)))

variance = np.var(scaled_data, axis=1)
print(len(variance))
'''  # Compute variance along the range dimension
max_variance_index = np.argmax(variance)  # Find the index with the maximum variance

# Extract vital sign signal information
vital_sign_signal = scaled_data[max_variance_index, :]

print(np.abs(vital_sign_signal[1:10]))
'''

from scipy.signal import butter, filtfilt, find_peaks
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_data = filtfilt(b, a, data)
    return filtered_data

fs=2000

hr_lowcut = 0.5 
hr_highcut = 2.5 
hr_signal = bandpass_filter(np.abs(variance), hr_lowcut, hr_highcut, fs)

br_lowcut = 0.1  
br_highcut = 0.5  
br_signal = bandpass_filter(np.abs(variance), br_lowcut, br_highcut, fs)
print(br_signal)

hr_envelope = np.abs(hr_signal)
br_envelope = np.abs(br_signal)

print(br_envelope[1:10])
print(hr_envelope[1:10])



