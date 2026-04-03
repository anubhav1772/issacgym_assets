import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton,
    QVBoxLayout, QHBoxLayout, QSlider, QLabel
)
from PyQt5.QtCore import Qt


class VelocityUI(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Robot Velocity Control")

        self.cmd = {"vx": 0.0, "vy": 0.0, "w": 0.0}

        layout = QVBoxLayout()

        # Buttons
        btn_layout = QVBoxLayout()

        self.btn_forward = QPushButton("Forward")
        self.btn_backward = QPushButton("Backward")
        self.btn_left = QPushButton("Left")
        self.btn_right = QPushButton("Right")
        self.btn_stop = QPushButton("STOP")

        btn_layout.addWidget(self.btn_forward)

        mid = QHBoxLayout()
        mid.addWidget(self.btn_left)
        mid.addWidget(self.btn_stop)
        mid.addWidget(self.btn_right)

        btn_layout.addLayout(mid)
        btn_layout.addWidget(self.btn_backward)

        layout.addLayout(btn_layout)

        # Sliders
        self.vx_slider = self.create_slider("vx")
        self.vy_slider = self.create_slider("vy")
        self.w_slider = self.create_slider("omega")

        layout.addWidget(self.vx_slider["label"])
        layout.addWidget(self.vx_slider["slider"])

        layout.addWidget(self.vy_slider["label"])
        layout.addWidget(self.vy_slider["slider"])

        layout.addWidget(self.w_slider["label"])
        layout.addWidget(self.w_slider["slider"])

        self.setLayout(layout)

        # Button connections
        self.btn_forward.clicked.connect(lambda: self.set_cmd(vx=1.0))
        self.btn_backward.clicked.connect(lambda: self.set_cmd(vx=-1.0))
        self.btn_left.clicked.connect(lambda: self.set_cmd(vy=1.0))
        self.btn_right.clicked.connect(lambda: self.set_cmd(vy=-1.0))
        self.btn_stop.clicked.connect(self.reset_cmd)

    def create_slider(self, name):
        label = QLabel(f"{name}: 0.0")

        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(-100)
        slider.setMaximum(100)
        slider.setValue(0)

        def update(val):
            v = val / 100.0
            self.cmd[name if name != "omega" else "w"] = v
            label.setText(f"{name}: {v:.2f}")

        slider.valueChanged.connect(update)

        return {"slider": slider, "label": label}

    def set_cmd(self, vx=None, vy=None, w=None):
        if vx is not None:
            self.cmd["vx"] = vx
        if vy is not None:
            self.cmd["vy"] = vy
        if w is not None:
            self.cmd["w"] = w

    def reset_cmd(self):
        self.cmd = {"vx": 0.0, "vy": 0.0, "w": 0.0}
