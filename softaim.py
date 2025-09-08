import threading
import time
import os
import ctypes
import numpy as np
import torch
import win32api
from ultralytics import YOLO
from preprocess import preprocess_frame
import mss
import base64
import requests
import sys

import os
os.environ["KIVY_NO_CONSOLELOG"] = "1"
os.environ["KIVY_LOG_LEVEL"] = "error"

from kivy.lang import Builder
from kivy.core.window import Window
from kivymd.app import MDApp
from kivy.properties import ListProperty
from kivymd.uix.slider import MDSlider
from kivy.graphics import Color, Rectangle, Ellipse

import logging
logging.getLogger("ultralytics").setLevel(logging.ERROR)

CURRENT_VERSION = 2

def show_message(title, message):
    ctypes.windll.user32.MessageBoxW(0, message, title, 0x40)

def UpdateTool():
    try:
        response = requests.get('https://raw.githubusercontent.com/NormandyRobloxer/versions/refs/heads/main/softaim.py')
        new_code = response.text
        current_file = sys.argv[0]

        with open(current_file, 'w', encoding='utf-8') as f:
            f.write(new_code)
        
        show_message("Update", "[+] Updated! Please restart the application!")
    except Exception as e:
        show_message("Error", f"[!] Failed to update: {e}")

def CheckUpdate():
    try:
        response = requests.get('https://raw.githubusercontent.com/NormandyRobloxer/versions/refs/heads/main/latest_version.txt')
        latest_version = int(response.text.strip())

        if latest_version > CURRENT_VERSION:
            show_message("Update Available", f"[!] New version available: {latest_version}")
            UpdateTool()
    except Exception as e:
        show_message("Error", f"[!] Failed to check for updates: {e}")

Window.size = (480, 280)
Window.clearcolor = (0.07, 0.07, 0.07, 1)

mouse_dll = ctypes.WinDLL(os.path.abspath("dlls/dd40605x64.dll"))
mouse_dll.DD_btn.argtypes = [ctypes.c_int]
mouse_dll.DD_btn.restype = ctypes.c_float
mouse_dll.DD_movR.argtypes = [ctypes.c_int, ctypes.c_int]
mouse_dll.DD_movR.restype = ctypes.c_int
mouse_dll.DD_btn(0)

drawing_dll = ctypes.CDLL(os.path.abspath("dlls/drawingx64.dll"))

model = YOLO("assets/weights.pt")
model.fuse()
model.model.eval()
model.model = torch.compile(model.model)
model.model = model.model.half().to("cuda")
torch.backends.cudnn.benchmark = True
target_class = 0

sct = mss.mss()
monitor = sct.monitors[1]
screen_width, screen_height = monitor["width"], monitor["height"]
screen_center = (screen_width // 2, screen_height // 2)
capture_radius = 100
capture_size = capture_radius * 2

running = False
ads_assist = False
show_fov = False
last_target = None
sensitivity = 4.7
fov = 100.0

def save_config():
    config_path = os.path.abspath("config/default.cfg")
    os.makedirs(os.path.dirname(config_path), exist_ok=True)

    data = f"{sensitivity}\n{fov}"
    encoded = base64.b64encode(data.encode("utf-8")).decode("utf-8")

    with open(config_path, "w") as f:
        f.write(encoded)

def load_config():
    config_path = os.path.abspath("config/default.cfg")
    global sensitivity, fov, capture_size, capture_radius
    try:
        with open(config_path, "r") as f:
            encoded = f.read().strip()
            decoded = base64.b64decode(encoded).decode("utf-8")
            lines = decoded.split("\n")
            if len(lines) >= 2:
                sensitivity = float(lines[0].strip())
                fov = float(lines[1].strip())
                capture_radius = int(fov)
                capture_size = capture_radius * 2
    except FileNotFoundError:
        sensitivity = 4.7
        fov = 100.0
        save_config()

class OrigSlider(MDSlider):
    bar_color = ListProperty([0.279, 0.051, 0.496, 1])
    handle_color = ListProperty([0.279, 0.051, 0.496, 1])
    bg_color = ListProperty([0.15, 0.03, 0.25, 1])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.background_normal = ''
        self.background_down = ''
        self.thumb_normal = ''
        self.thumb_down = ''
        self.color = [0, 0, 0, 0]
        self.track_color = [0, 0, 0, 0]
        self.thumb_color_active = [0.279, 0.051, 0.496, 1]
        self.hint = False
        self.bind(pos=self.update_canvas, size=self.update_canvas, value=self.update_canvas)
        self.update_canvas()

    def update_canvas(self, *args):
        self.canvas.after.clear()
        offset = -2
        reduce_width = 32
        x_offset = reduce_width / 2
        w = self.width - reduce_width

        with self.canvas.after:
            Color(*self.bg_color)
            Rectangle(pos=(self.x + x_offset, self.center_y - 2 + offset), size=(w, 4))
            Color(*self.bar_color)
            Rectangle(pos=(self.x + x_offset, self.center_y - 2 + offset), size=(w * self.value_normalized, 4))
            Color(*self.handle_color)
            Ellipse(pos=(self.x + x_offset + w * self.value_normalized - 10, self.center_y - 10 + offset), size=(20, 20))

def aimbot_loop():
    global last_target, running, sensitivity
    sct_local = mss.mss()

    frame_count = 0
    start_time = time.time()

    while running:
        if win32api.GetKeyState(0x01) < 0 or (ads_assist and win32api.GetKeyState(0x02) < 0):
            left = screen_center[0] - capture_radius
            top = screen_center[1] - capture_radius
            raw_frame = np.array(
                sct_local.grab({"left": left, "top": top, "width": capture_size, "height": capture_size}),
                dtype=np.uint8
            )[:, :, :3]
            
            tensor = preprocess_frame(raw_frame, size=224)

            with torch.inference_mode():
                results = model(tensor, device="cuda", half=True, verbose=False)

            target_coords = None
            for result in results:
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        if int(box.cls) == target_class and box.conf > 0.2:
                            coords = (box.xyxy[0] * (capture_size / 224)).to(torch.float16)
                            x1, y1, x2, y2 = coords
                            target_coords = ((x1 + x2)//2, (y1 + y2)//2.12)
                            break
                if target_coords:
                    break

            last_target = target_coords

            if target_coords:
                left_pressed = win32api.GetKeyState(0x01) < 0
                right_pressed = win32api.GetKeyState(0x02) < 0
                if left_pressed or (ads_assist and right_pressed):
                    dx = float(target_coords[0]) - capture_radius
                    dy = float(target_coords[1]) - capture_radius
                    mouse_dll.DD_movR(round(dx*sensitivity/2), round(dy*sensitivity/2))

            frame_count += 1
            elapsed = time.time() - start_time
            if elapsed >= 0.25:
                fps = frame_count / elapsed
                frame_count = 0
                start_time = time.time()

                try:
                    if App.get_running_app().root:
                        App.get_running_app().root.ids.fps_label.text = f"FPS: {int(fps)}"
                except:
                    pass

            time.sleep(0)

class App(MDApp):
    def build(self):
        self.title = "Aero Softaim"
        Window.set_title("Aero Softaim")
        Window.set_icon(os.path.abspath("assets/icon.png"))
        self.theme_cls.theme_style = "Dark"
        self.theme_cls.primary_palette = "BlueGray"
        root = Builder.load_file(os.path.abspath("assets/design.kv"))

        root.ids.sens_slider.value = sensitivity
        root.ids.sens_label.text = f"Sensitivity: {sensitivity:.1f}"
        root.ids.fov_slider.value = fov
        root.ids.fov_label.text = f"FOV: {fov:.1f}"

        root.ids.aimbot_switch.bind(active=self.toggle_aimbot)
        root.ids.ads_switch.bind(active=self.toggle_ads)
        root.ids.show_fov_switch.bind(active=self.toggle_fov)
        root.ids.sens_slider.bind(value=self.update_sensitivity)
        root.ids.fov_slider.bind(value=self.update_fov)

        return root

    def toggle_aimbot(self, instance, value):
        global running
        if value:
            running = True
            threading.Thread(target=aimbot_loop, daemon=True).start()
        else:
            running = False

    def toggle_ads(self, instance, value):
        global ads_assist
        ads_assist = value

    def toggle_fov(self, instance, value):
        global show_fov
        show_fov = value
        if value:
            drawing_dll.StartOverlay()
            drawing_dll.UpdateFOV(int(fov))
        else:
            drawing_dll.StopOverlay()

    def update_sensitivity(self, instance, value):
        global sensitivity
        sensitivity = value
        self.root.ids.sens_label.text = f"Sensitivity: {value:.1f}"

    def update_fov(self, instance, value):
        global fov, capture_size, capture_radius
        fov = value
        capture_radius = int(fov)
        capture_size = capture_radius * 2
        self.root.ids.fov_label.text = f"FOV: {value:.1f}"
        if show_fov:
            drawing_dll.UpdateFOV(int(value))

    def on_stop(self):
        save_config()

if __name__ == "__main__":
    CheckUpdate()
    load_config()
    App().run()
