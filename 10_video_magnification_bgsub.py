import cv2
import numpy as np
from scipy.fftpack import fft, ifft, fftfreq


def detect_objects(frame, background_subtractor):
    """
    简单物体检测，使用背景减除法
    """
    fg_mask = background_subtractor.apply(frame)
    return fg_mask


def temporal_ideal_filter(roi_sequence, low, high, fps):
    """
    应用理想时域滤波到ROI序列
    """
    fft = np.fft.fft(roi_sequence, axis=0)
    frequencies = np.fft.fftfreq(roi_sequence.shape[0], d=1.0 / fps)
    bound_low = (np.abs(frequencies - low)).argmin()
    bound_high = (np.abs(frequencies - high)).argmin()
    fft[:bound_low] = 0
    fft[bound_high:-bound_high] = 0
    fft[-bound_low:] = 0
    iff = np.fft.ifft(fft, axis=0)
    return np.abs(iff)


def load_video(video_filename):
    """
    加载视频文件，返回视频帧的numpy数组
    """
    cap = cv2.VideoCapture(video_filename)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    video_tensor = np.zeros((frame_count, height, width, 3), dtype='float')

    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        video_tensor[i] = frame
    cap.release()
    return video_tensor, fps


def magnify_motion(video_tensor, fps, low=0.4, high=3, amplification=20):
    """
    放大视频中的运动
    """
    background_subtractor = cv2.createBackgroundSubtractorMOG2()
    motion_magnified_video = np.copy(video_tensor)

    for i, frame in enumerate(video_tensor):
        fg_mask = detect_objects(frame.astype(np.uint8), background_subtractor)
        if i == 0:
            # 初始化ROI序列存储
            roi_sequence = np.zeros((video_tensor.shape[0], fg_mask.shape[0], fg_mask.shape[1]), dtype=np.float32)

        # 将当前帧的感兴趣区域加入序列
        roi_sequence[i] = fg_mask

    # 对ROI序列应用时域滤波
    filtered_sequence = temporal_ideal_filter(roi_sequence, low, high, fps)
    filtered_sequence *= amplification  # 放大滤波后的信号

    # 将放大后的运动添加回视频
    for i in range(video_tensor.shape[0]):
        magnified_frame = video_tensor[i]
        fg_mask = filtered_sequence[i]
        # 将单通道掩膜扩展为3通道以匹配图像维度
        fg_mask_3d = np.stack([fg_mask] * 3, axis=-1)
        magnified_frame[fg_mask_3d > 0] += fg_mask_3d[fg_mask_3d > 0]
        motion_magnified_video[i] = magnified_frame

    return motion_magnified_video


# 示例主程序
if __name__ == "__main__":
    video_name = "test_video/test.avi"  # 修改为你的视频路径
    video_tensor, fps = load_video(video_name)
    magnified_video = magnify_motion(video_tensor, fps)
    # 保存或展示放大后的视频
    # save_video(magnified_video)  # 保存函数需自行实现