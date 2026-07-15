import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft

fs = 120  # 视频帧率
N = 600  # 总帧数

# 使用给定的帧数和帧率定义时间t
t = np.arange(N) / fs  # 更准确的时间定义

# 模拟轨迹数据
displacements_x = 2.2 * np.sin(2 * np.pi * t * 3)  # 垂直运动频率3Hz，振幅2.2像素
displacements_y = 1.7 * np.sin(2 * np.pi * t * 5)  # 水平运动频率5Hz，振幅1.7像素

# 时域分析
plt.figure(figsize=(12, 6))
plt.subplot(211)
plt.title('Time Domain - Displacement')
plt.plot(t, displacements_x, label='Displacement X')
plt.plot(t, displacements_y, label='Displacement Y')
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('Displacement (pixels)')

# 频域分析
frequencies_x = fft(displacements_x)
frequencies_y = fft(displacements_y)
freq = np.fft.fftfreq(N, 1/fs)

# 幅值的归一化处理
amp_x = np.abs(frequencies_x) / N * 2  # 归一化幅值
amp_y = np.abs(frequencies_y) / N * 2

plt.subplot(212)
plt.title('Frequency Domain - FFT')
plt.plot(freq[:N//2], amp_x[:N//2], label='FFT X')
plt.plot(freq[:N//2], amp_y[:N//2], label='FFT Y')
plt.legend()
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')

plt.tight_layout()
plt.show()