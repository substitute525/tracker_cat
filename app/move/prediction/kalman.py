import numpy as np
from filterpy.kalman import KalmanFilter


def init_pet_tracker(initial_dt=1.0/30.0, initial_R_factor=5.0, initial_Q_var=0.1):
    """
    初始化宠物追踪器
    
    参数:
    - initial_dt: 时间步长，默认为1/30秒(30FPS)
    - initial_R_factor: 初始测量噪声因子，默认为5.0
    - initial_Q_var: 初始过程噪声方差，默认为0.1
    """
    # 状态向量 dim_x=4: [x, y, vx, vy] (位置和速度)
    # 观测向量 dim_z=2: [x, y] (视觉算法识别到的坐标)
    tracker = KalmanFilter(dim_x=4, dim_z=2)

    # 1. 状态转移矩阵 (F): 假设 x = x + vx*dt
    dt = initial_dt  # 假设摄像头是 30 FPS
    tracker.F = np.array([[1, 0, dt, 0],
                          [0, 1, 0, dt],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]])

    # 2. 观测矩阵 (H): 我们只能直接看到位置 x, y
    tracker.H = np.array([[1, 0, 0, 0],
                          [0, 1, 0, 0]])

    # 3. 测量噪声协方差 (R): 视觉识别算法的误差精度
    # 数值越大，越不相信视觉识别结果，轨迹越平滑
    tracker.R *= initial_R_factor

    # 4. 预测噪声/过程噪声 (Q): 宠物运动的不可预测性
    # 如果宠物经常突然变向，增大这个值
    from filterpy.common import Q_discrete_white_noise
    tracker.Q = Q_discrete_white_noise(dim=2, dt=dt, var=initial_Q_var, block_size=2)

    # 5. 初始状态和协方差
    tracker.x = np.array([0, 0, 0, 0])
    tracker.P *= 10.

    return tracker


def adjust_process_noise(tracker, q_var_factor=1.0):
    """
    动态调整过程噪声Q矩阵
    
    参数:
    - tracker: KalmanFilter对象
    - q_var_factor: Q矩阵方差调整因子
    """
    dt = tracker.F[0, 2]  # 从状态转移矩阵获取时间步长
    from filterpy.common import Q_discrete_white_noise
    tracker.Q = Q_discrete_white_noise(dim=2, dt=dt, var=q_var_factor, block_size=2)


def adjust_measurement_noise(tracker, r_factor=1.0):
    """
    动态调整测量噪声R矩阵
    
    参数:
    - tracker: KalmanFilter对象
    - r_factor: R矩阵调整因子
    """
    # 重新设置R矩阵，保持原有的结构
    tracker.R = np.array([[r_factor, 0],
                         [0, r_factor]])


def dynamic_adjust_noise(tracker, q_var_factor=0.1, r_factor=5.0):
    """
    动态调整Q和R矩阵
    
    参数:
    - tracker: KalmanFilter对象
    - q_var_factor: Q矩阵方差调整因子
    - r_factor: R矩阵调整因子
    """
    adjust_process_noise(tracker, q_var_factor)
    adjust_measurement_noise(tracker, r_factor)


def get_position_uncertainty(tracker):
    """
    获取位置的不确定性（标准差）
    
    参数:
    - tracker: KalmanFilter对象
    
    返回:
    - (x_std, y_std): x和y方向的标准差
    """
    x_uncertainty = np.sqrt(tracker.P[0, 0])  # x位置的不确定性
    y_uncertainty = np.sqrt(tracker.P[1, 1])  # y位置的不确定性
    return x_uncertainty, y_uncertainty


def get_prediction_confidence(tracker):
    """
    获取当前预测的置信度
    
    参数:
    - tracker: KalmanFilter对象
    
    返回:
    - confidence: 置信度值 (0-1之间，值越大表示越可信)
    """
    x_uncertainty, y_uncertainty = get_position_uncertainty(tracker)
    
    # 将不确定性转换为置信度 (使用高斯函数形式)
    # 不确定性越小，置信度越高
    avg_uncertainty = (x_uncertainty + y_uncertainty) / 2
    
    # 使用sigmoid函数或其他函数将不确定性映射到[0,1]区间
    # 这里使用指数衰减函数: confidence = exp(-uncertainty/scale)
    scale_factor = 10.0  # 可调参数，决定置信度下降的速度
    confidence = np.exp(-avg_uncertainty / scale_factor)
    
    # 确保置信度在[0,1]范围内
    return np.clip(confidence, 0.0, 1.0)


# --- 模拟追踪过程 ---
tracker = init_pet_tracker()

print("初始Q矩阵:", tracker.Q)
print("初始R矩阵:", tracker.R)

# 动态调整噪声矩阵示例
adjust_process_noise(tracker, q_var_factor=0.5)  # 调整过程噪声
print("\n调整Q矩阵后:", tracker.Q)

adjust_measurement_noise(tracker, r_factor=10.0)  # 调整测量噪声
print("调整R矩阵后:", tracker.R)

# 重置并使用综合调整方法
tracker = init_pet_tracker()
dynamic_adjust_noise(tracker, q_var_factor=0.2, r_factor=3.0)
print("\n使用综合调整方法后:")
print("Q矩阵:", tracker.Q)
print("R矩阵:", tracker.R)

# 测试置信度功能
print("\n测试置信度功能:")
uncertainty = get_position_uncertainty(tracker)
confidence = get_prediction_confidence(tracker)
print(f"位置不确定性: x_std={uncertainty[0]:.3f}, y_std={uncertainty[1]:.3f}")
print(f"预测置信度: {confidence:.3f}")

# 椭圆参数
ellipse_params = get_confidence_ellipse_params(tracker)
print(f"置信椭圆参数: 中心({ellipse_params[0]:.2f}, {ellipse_params[1]:.2f}), "
      f"主轴={ellipse_params[2]:.2f}, 次轴={ellipse_params[3]:.2f}, 方向={ellipse_params[4]:.2f}弧度")

# 假设视觉算法识别到的宠物坐标序列（带噪声）
measurements = [[10, 10], [11, 12], [13, 15], None, [17, 20]]  # None 代表被遮挡了

for i, z in enumerate(measurements):
    tracker.predict()  # 无论是否看到，都先进行物理预测
    
    # 获取当前预测的置信度
    current_confidence = get_prediction_confidence(tracker)
    current_uncertainty = get_position_uncertainty(tracker)

    if z is not None:
        tracker.update(z)  # 如果看到了，用实际观测值修正
        print(f"第{i+1}帧 - 检测到宠物，修正后位置: {tracker.x[:2]}, 置信度: {current_confidence:.3f}")
    else:
        print(f"第{i+1}帧 - 宠物被遮挡！预测其位置: {tracker.x[:2]}, 置信度: {current_confidence:.3f}")
    
    # 根据追踪情况动态调整噪声
    if i == 2:  # 在第3次测量后调整噪声
        dynamic_adjust_noise(tracker, q_var_factor=0.15, r_factor=7.0)
        print(f"第{i+1}帧后动态调整了噪声矩阵")
        
        # 检查调整后置信度的变化
        new_confidence = get_prediction_confidence(tracker)
        print(f"调整后置信度: {new_confidence:.3f}")
        
        # 显示置信椭圆参数
        ellipse_params = get_confidence_ellipse_params(tracker)
        print(f"置信椭圆参数: 中心({ellipse_params[0]:.2f}, {ellipse_params[1]:.2f}), "
              f"主轴={ellipse_params[2]:.2f}, 次轴={ellipse_params[3]:.2f}")