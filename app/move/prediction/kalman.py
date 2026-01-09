import numpy as np
from filterpy.kalman import KalmanFilter


def init_pet_tracker():
    # 状态向量 dim_x=4: [x, y, vx, vy] (位置和速度)
    # 观测向量 dim_z=2: [x, y] (视觉算法识别到的坐标)
    tracker = KalmanFilter(dim_x=4, dim_z=2)

    # 1. 状态转移矩阵 (F): 假设 x = x + vx*dt
    dt = 1.0 / 30.0  # 假设摄像头是 30 FPS
    tracker.F = np.array([[1, 0, dt, 0],
                          [0, 1, 0, dt],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]])

    # 2. 观测矩阵 (H): 我们只能直接看到位置 x, y
    tracker.H = np.array([[1, 0, 0, 0],
                          [0, 1, 0, 0]])

    # 3. 测量噪声协方差 (R): 视觉识别算法的误差精度
    # 数值越大，越不相信视觉识别结果，轨迹越平滑
    tracker.R *= 5

    # 4. 预测噪声/过程噪声 (Q): 宠物运动的不可预测性
    # 如果宠物经常突然变向，增大这个值
    from filterpy.common import Q_discrete_white_noise
    tracker.Q = Q_discrete_white_noise(dim=2, dt=dt, var=0.1, block_size=2)

    # 5. 初始状态和协方差
    tracker.x = np.array([0, 0, 0, 0])
    tracker.P *= 10.

    return tracker


# --- 模拟追踪过程 ---
tracker = init_pet_tracker()

# 假设视觉算法识别到的宠物坐标序列（带噪声）
measurements = [[10, 10], [11, 12], [13, 15], None, [17, 20]]  # None 代表被遮挡了

for z in measurements:
    tracker.predict()  # 无论是否看到，都先进行物理预测

    if z is not None:
        tracker.update(z)  # 如果看到了，用实际观测值修正
        print(f"检测到宠物，修正后位置: {tracker.x[:2]}")
    else:
        print(f"宠物被遮挡！预测其位置: {tracker.x[:2]}")