import struct
import os
import torch

class JoystickReader:
    def __init__(self, device="/dev/input/js0", command_ranges=None):
        self.device_path = device
        self.command_ranges = command_ranges or {
            "lin_vel_x": [-1.3, 1.3],
            "lin_vel_y": [-0.3, 0.3],
            "ang_vel_yaw": [-1.5, 1.5],
        }

        self.axis_states = {}  # 存储最新的轴值

        # 打开设备
        if not os.path.exists(self.device_path):
            raise FileNotFoundError(f"Joystick device not found: {self.device_path}")
        self.dev = open(self.device_path, "rb")
        print(f"[JoystickReader] Using device: {self.device_path}")

    def read_event(self):
        JS_EVENT_FORMAT = "IhBB"
        JS_EVENT_SIZE = struct.calcsize(JS_EVENT_FORMAT)

        # 非阻塞读取（不等待）
        import fcntl
        import os
        fcntl.fcntl(self.dev, fcntl.F_SETFL, os.O_NONBLOCK)

        try:
            while True:
                data = self.dev.read(JS_EVENT_SIZE)
                if not data:
                    break
                time, value, type_, number = struct.unpack(JS_EVENT_FORMAT, data)
                if type_ & 0x02:  # 轴移动
                    self.axis_states[number] = value / 32767.0  # 标准化为 [-1, 1]
        except BlockingIOError:
            pass  # 没有新事件时继续

    def get_commands(self):
        self.read_event()

        # 默认值
        lin_vel_x_raw = -self.axis_states.get(1, 0.0)  # 向前是负
        lin_vel_y_raw = -self.axis_states.get(0, 0.0)  # 向右是负
        ang_vel_raw   = -self.axis_states.get(3, 0.0)  # 右转为负

        # 缩放到命令范围
        lin_vel_x = lin_vel_x_raw * (
            abs(self.command_ranges["lin_vel_x"][1]) if lin_vel_x_raw >= 0 else abs(self.command_ranges["lin_vel_x"][0])
        )
        lin_vel_y = lin_vel_y_raw * (
            abs(self.command_ranges["lin_vel_y"][1]) if lin_vel_y_raw >= 0 else abs(self.command_ranges["lin_vel_y"][0])
        )
        ang_vel = ang_vel_raw * (
            abs(self.command_ranges["ang_vel_yaw"][1]) if ang_vel_raw >= 0 else abs(self.command_ranges["ang_vel_yaw"][0])
        )

        return lin_vel_x, lin_vel_y, ang_vel
