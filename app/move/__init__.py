import argparse
import time

import pigpio

import args
from move.pid import PIDController

pi = pigpio.pi()

args.add_argument("-p", "--pin", type=int, nargs=2, default=[18, 19], help="Steering gear pin. default: [18,19]")
args.add_argument("-u", "--pulse", type=int, nargs=2, default=[500, 2500], help="pulse range. default: [500, 2500]")
# 舵机参数
pan_pin = args.get_args().pin[0]    # 横向舵机引脚
tilt_pin = args.get_args().pin[1]   # 纵向舵机引脚
min_pulse = args.get_args().pulse[0]     # 脉宽范围
max_pulse = args.get_args().pulse[1]    # 脉宽范围
center_pulse = (max_pulse + min_pulse) / 2  # 初始位置
per_pulse = (max_pulse - min_pulse) / 180

# 屏幕中心坐标
screen_width = 960
screen_height = 544
target_x = screen_width / 2
target_y = screen_height / 2

# 视野参数（根据摄像头调整）
h_fov = 60  # 水平视野角度
v_fov = 40  # 垂直视野角度


# 初始化PID控制器
pid_x = PIDController(Kp=0.5, Ki=0.01, Kd=0.1, max_output=180)
pid_y = PIDController(Kp=0.5, Ki=0.01, Kd=0.1, max_output=180)

current_pan = center_pulse
current_tilt = center_pulse

def move_by_position(x, y):
    try:
        while True:
            # 计算像素误差
            error_x = target_x - x
            error_y = target_y - y

            # 转换为角度误差
            error_x_deg = (error_x / screen_width) * h_fov
            error_y_deg = (error_y / screen_height) * v_fov

            # 计算PID输出（角度调整量）
            output_x_deg = pid_x.compute(error_x_deg)
            output_y_deg = pid_y.compute(error_y_deg)

            # 转换为脉宽（11.11微秒/度）
            pan_pulse = current_pan + output_x_deg * per_pulse
            tilt_pulse = current_tilt + output_y_deg * per_pulse

            # 限制脉宽范围
            pan_pulse = max(min_pulse, min(pan_pulse, max_pulse))
            tilt_pulse = max(min_pulse, min(tilt_pulse, max_pulse))

            # 更新舵机
            pi.set_servo_pulsewidth(pan_pin, pan_pulse)
            pi.set_servo_pulsewidth(tilt_pin, tilt_pulse)

            # 记录当前脉宽
            current_pan = pan_pulse
            current_tilt = tilt_pulse

            time.sleep(0.1)  # 控制频率

    except KeyboardInterrupt:
        pi.set_servo_pulsewidth(pan_pin, 0)
        pi.set_servo_pulsewidth(tilt_pin, 0)
        pi.stop()