import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft

# 定义参数
fs = 120  # 视频帧率
N = 600  # 总帧数

# 使用给定的帧数和帧率定义时间t
t = np.arange(N) / fs  # 更准确的时间定义

# 模拟轨迹数据
displacements_x = 2.2 * np.sin(2 * np.pi * t * 3)  # 垂直运动频率3Hz，振幅2.2像素
displacements_y = 1.7 * np.sin(2 * np.pi * t * 5)  # 水平运动频率5Hz，振幅1.7像素

# 计算合成位移
displacement_magnitude = np.sqrt(displacements_x**2 + displacements_y**2)

# 时域分析
plt.figure(figsize=(12, 6))
plt.title('Time Domain - Composite Displacement')
plt.plot(t, displacement_magnitude, label='Composite Displacement')
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('Composite Displacement (pixels)')
plt.show()

# 对合成位移进行傅里叶变换
freqs = np.fft.fftfreq(N, 1/fs)  # 计算频率
fft_displacement = np.fft.fft(displacement_magnitude)  # 对位移进行傅里叶变换

# 归一化处理
fft_displacement /= np.max(np.abs(fft_displacement)) //2 # 归一化处理

# 绘制频域图像
plt.figure(figsize=(12, 6))
plt.title('Frequency Domain - Composite Displacement')
plt.plot(freqs[:N//2], np.abs(fft_displacement)[:N//2], label='Composite Displacement')
plt.legend()
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.show()