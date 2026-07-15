import numpy as np
import cv2
import time
import datetime
from scipy.fftpack import fft
from sklearn.metrics import mean_squared_error, mean_absolute_error

cap = cv2.VideoCapture("test_video/test.avi")  # 打开一个视频

fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 设置保存图片格式
out = cv2.VideoWriter(
    datetime.datetime.now().strftime("%A_%d_%B_%Y_%I_%M_%S%p") + '.avi', fourcc, 10.0, (768, 576))  # 分辨率要和原视频对应

# ShiTomasi 角点检测参数
feature_params = dict(maxCorners=100,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)

# lucas kanade光流法参数
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# 创建随机颜色
color = np.random.randint(0, 255, (100, 3))

# 获取第一帧，找到角点
ret, old_frame = cap.read()
# 找到原始灰度图
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

# 获取图像中的角点，返回到p0中
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# 创建一个蒙版用来画轨迹
mask = np.zeros_like(old_frame)

# 初始化用于存储轨迹的字典
# 键是追踪点的索引，值是该追踪点的位置历史记录
tracks = {}

while(1):
    ret, frame = cap.read()  # 读取图像帧
    if not ret:
        break
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 灰度化

    # 计算光流
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    # 选取好的跟踪点
    good_new = p1[st == 1]
    good_old = p0[st == 1]

    # 画出轨迹
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()  # 多维数据转一维,将坐标转换后赋值给a,b
        c, d = old.ravel()
        mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2) # 画直线
        frame = cv2.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)  # 画点
    img = cv2.add(frame, mask)  # 将画出的线条进行图像叠加

    cv2.imshow('frame', img)  # 显示图像

    out.write(img)  # 保存每一帧画面

    k = cv2.waitKey(30) & 0xff  # 按Esc退出检测
    if k == 27:
        break

    # 更新上一帧的图像和追踪点
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

    # 对每个轨迹进行历史记录
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        if i not in tracks:
            tracks[i] = []  # 如果是新的跟踪点，初始化其轨迹
        tracks[i].append((a, b))  # 将当前位置添加到轨迹中

# 释放文件
out.release()
cap.release()
cv2.destroyAllWindows()  # 关闭所有窗口
import numpy as np
import cv2
import datetime
import matplotlib.pyplot as plt


def calculate_displacement(tracks, frame_rate):

    displacements = {}
    for track_id, points in tracks.items():
        track_displacement = []
        initial_point = points[0]
        for point in points:
            dx = point[0] - initial_point[0]
            dy = point[1] - initial_point[1]
            displacement = np.sqrt(dx**2 + dy**2)
            track_displacement.append(displacement)
        displacements[track_id] = track_displacement
    return displacements
frame_rate = 120  # 假设的帧率

displacements = calculate_displacement(tracks, frame_rate)
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
#回归评价指标from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt


def calculate_displacement_simulated():
    # 定义时间参数
    fs = 120  # 视频帧率
    N = 600  # 总帧数
    t = np.arange(N) / fs  # 时间定义

    # 模拟数据
    displacements_x = 2.2 * np.sin(2 * np.pi * t * 3)  # 垂直运动频率3Hz，振幅2.2像素
    displacements_y = 1.7 * np.sin(2 * np.pi * t * 5)  # 水平运动频率5Hz，振幅1.7像素

    # 计算合成位移
    displacement_magnitude = np.sqrt(displacements_x ** 2 + displacements_y ** 2)
    return displacement_magnitude
displacements_video = calculate_displacement_simulated()

def analyze_displacement_time(displacements_video, displacements_simulated, frame_rate):
    time_seconds_video = np.arange(len(displacements_video)) / frame_rate
    time_seconds_simulated = np.arange(len(displacements_simulated)) / frame_rate

    plt.figure(figsize=(12, 6))
    plt.plot(time_seconds_video, displacements_video, label='Video Tracking Displacement')
    plt.plot(time_seconds_simulated, displacements_simulated, label='Simulated Displacement')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Displacement (pixels)')
    plt.title('Comparison of Video Tracking and Simulated Displacements')
    plt.xlim(0, 5)
    plt.legend()
    plt.tight_layout()
    plt.show()


# 误差评估：取track 0的跟踪位移与合成位移对比
if 0 in displacements:
    tracked_disp = np.array(displacements[0])
    sim_disp = displacements_video[:len(tracked_disp)]  # displacements_video 即合成位移
    mse = mean_squared_error(tracked_disp, sim_disp)
    rmse = sqrt(mse)
    mae = mean_absolute_error(tracked_disp, sim_disp)

    print('均方误差: %.6f' % mse)
    print('均方根误差: %.6f' % rmse)
    print('平均绝对误差: %.6f' % mae)

fs = 120  # 视频帧率
N = 600  # 总帧数
t = np.arange(N) / fs  # 时间定义
# 模拟数据
displacements_x = 2.2 * np.sin(2 * np.pi * t * 3)  # 垂直运动频率3Hz，振幅2.2像素
displacements_y = 1.7 * np.sin(2 * np.pi * t * 5)  # 水平运动频率5Hz，振幅1.7像素

# 计算合成位移
displacement_magnitude = np.sqrt(displacements_x ** 2 + displacements_y ** 2)
# 在while循环完成之后（所有视频帧处理完成之后）:
displacements = calculate_displacement(tracks, frame_rate=120)  # 从视频跟踪数据中计算位移
track_id = 0  # 示范性轨迹
if track_id in displacements:
    # 此处传递模拟位移数据 displacement_magnitude 作为第二个参数而不是 track_id
    analyze_displacement_time(displacements[track_id], displacement_magnitude, frame_rate=120)
else:
    print(f"未找到轨迹ID {track_id}。")


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
def analyze_frequency_domain(fft_values, samplerate, track_id):
    T = 1.0 / samplerate
    N = len(fft_values)
    xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
    plt.figure(figsize=(10, 4))
    plt.plot(xf, 2.0/N * np.abs(fft_values[:N//2]), label=f"Track {track_id}")
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.title('Frequency domain')
    plt.legend()
    plt.show()

for track_id, track in tracks.items():
    positions = np.array(track)
    xs = positions[:, 0]
    ys = positions[:, 1]

    velocities = np.sqrt(np.diff(xs)**2 + np.diff(ys)**2)
    analyze_displacement_time(displacements[track_id], displacement_magnitude, frame_rate=120)

    fft_xs = fft(xs - np.mean(xs))
    fft_ys = fft(ys - np.mean(ys))
    # 假设视频的采样率是120帧/秒，请根据你的视频进行调整
    analyze_frequency_domain(fft_xs, 120, f"X {track_id}")
    analyze_frequency_domain(fft_ys, 120, f"Y {track_id}")