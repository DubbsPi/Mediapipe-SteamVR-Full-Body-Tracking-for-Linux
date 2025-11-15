import time
import os
import socket
import cv2
import numpy as np
import argparse
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageDraw, ImageFont
from collections import deque
import json
import struct
import threading
import pygame
import math
import quaternion

from SimpleDisplay import SimpleImageDisplay
from gui import RealTimeGUI

FRAMERATE = 30

# Set MediaPipe path
script_path = os.path.realpath(__file__)
script_directory = os.path.dirname(script_path)
model_path = f"{script_directory}/pose_landmarker_heavy.task"
print(model_path)

def process_frame(video_frame, brightness):
    # Brighten
    video_frame = cv2.convertScaleAbs(video_frame, alpha=1.5, beta=brightness)

    # Convert BGR to RGB for MediaPipe
    return cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB)

def draw_landmarks(image, points, fps, other=False):
    width, height = image.size
    draw = ImageDraw.Draw(image)

    if not other:
        for key, value in points.items():
            x, y = value[0], value[1]
            x, y = width- x*width, y*height
            draw.ellipse([x-2,y-2, x+2,y+2], fill='green')
            draw.point((x,y), fill='white')
            draw.text((x,y), str(key), fill='black')
    else:
        for key, value in points.items():
            x, y = value[2], value[1]
            x, y = width- x*width, y*height
            draw.ellipse([x-2,y-2, x+2,y+2], fill='green')
            draw.point((x,y), fill='white')
            draw.text((x,y), str(key), fill='black')

    color_dif = int(max(0, (FRAMERATE - fps) * 8.5))
    draw.text((5,5), f"FPS: {fps}", fill=(color_dif, 255 - color_dif, 0))
    return image

def smooth_landmark(idx, new_point, max_history):
    now = time.time()

    if idx not in landmark_history:
        landmark_history[idx] = {"points": deque(maxlen=max_history), "last_seen": now}

    landmark_history[idx]["points"].append(new_point)
    landmark_history[idx]["last_seen"] = now

    pts = landmark_history[idx]["points"]
    return (
        sum(p[0] for p in pts) / len(pts),
        sum(p[1] for p in pts) / len(pts),
        sum(p[2] for p in pts) / len(pts)
    )

def cleanup(timeout=0.5):
    # Remove stale landmarks not seen for bit
    now = time.time()
    to_remove = [idx for idx, data in landmark_history.items()
                 if now - data["last_seen"] > timeout]
    for idx in to_remove:
        del landmark_history[idx]

def send_data(socket, points):
    for key, value in points.items():
        try:
            packed_data = struct.pack('>i3f', int(key),
                                    (float(gui.slider_vars[0].get()) - float(value[0])) * 2,
                                    float(gui.slider_vars[1].get()) - float(value[1]),
                                    float(gui.slider_vars[2].get()) + float(value[2]))

            socket.sendall(packed_data)
        except Exception as e:
            print(e)
            continue

def get_monitor_size_pygame():
    pygame.display.init()
    infoObject = pygame.display.Info()
    screen_width = infoObject.current_w
    screen_height = infoObject.current_h
    pygame.display.quit()
    return screen_width, screen_height

def main():
    args = gui.get_current_values()

    # Socket setup
    socket_path = "/tmp/vr_unix_socket.sock"
    client_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)

    # Setup camera
    try:
        camera=int(args['camera'])
    except:
        camera=args['camera']

    print(f"Camera: {camera}")
    cap = cv2.VideoCapture(camera)

    # Get actual camera dimensions first
    ret, test_frame = cap.read()
    if not ret:
        print("Failed to get test frame from camera")
        return

    # Set camera properties to match what we want
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, FRAMERATE)


    # Set display window position
    screen_width, screen_height = get_monitor_size_pygame()
    os.environ['SDL_VIDEO_WINDOW_POS'] = f"{screen_width // 2 + 250},{screen_height // 2 - 240}"

    # Setup display window
    display = SimpleImageDisplay(640, 480, caption="VR Tracker", fullscreen=False)

    # Create a PoseLandmarker object
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=True)
    detector = vision.PoseLandmarker.create_from_options(options)

    try:
        client_socket.connect(socket_path)
        connected = True
        print("Connected to SteamVR")
    except socket.error:
        print("Cannot connect to SteamVR")
        connected = False

    try:
        video_frame = None
        fps = 0
        frame_count = 0
        start_time = time.time()

        while True:
            args = gui.get_current_values()

            try:
                if not args['paused']:
                    ret, video_frame = cap.read()
                    if not ret:
                        print("Failed to grab frame")
                        continue
                    video_frame = cv2.resize(video_frame, (640, 480))

            except: continue

            # Convert for mediapipe
            rgb_frame = process_frame(video_frame, gui.slider_vars[4].get())
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            # Convert for pillow
            image = Image.fromarray(rgb_frame)
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

            # Process the frame with MediaPipe
            detection_results = detector.detect(mp_image)
            landmarks_list = detection_results.pose_landmarks
            worldlandmarks_list = detection_results.pose_world_landmarks

            if landmarks_list:
                points = [(lm.x, lm.y, lm.z) for lm in landmarks_list[0]]
                world_points = [(lm.x, lm.y, lm.z) for lm in worldlandmarks_list[0]]

                smooth_points = {}
                for index, landmark in enumerate(landmarks_list[0]):
                    if landmark.visibility > 0.5:  # skip if too uncertain
                        smoothed = smooth_landmark(index, points[index], int(gui.slider_vars[3].get()))
                        smooth_points[index] = smoothed

                world_smooth_points = {}
                for index, landmark in enumerate(worldlandmarks_list[0]):
                    if landmark.visibility > 0.5:  # skip if too uncertain
                        smoothed = smooth_landmark(index+99, world_points[index], int(math.ceil(FRAMERATE /30) * gui.slider_vars[3].get()))
                        world_smooth_points[index] = smoothed

                if connected:
                    # Send data
                    send_data(client_socket, world_smooth_points)

                image = draw_landmarks(image, smooth_points, fps)

            cleanup()

            try:
                display.display_image(image)
            except: continue

            if display.check_for_quit():
                break

            # Frame timing
            elapsed_time = time.time() - start_time

            # FPS Calculation
            frame_count += 1
            if frame_count % 10 == 0:
                if elapsed_time > 0:
                    fps = round(frame_count / elapsed_time, 1)

            time.sleep(max(0, 1/FRAMERATE - elapsed_time))

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        # Cleanup
        cap.release()
        display.quit()
        client_socket.close()

if __name__ == "__main__":
    try:
        # Create the GUI
        gui = RealTimeGUI()
        landmark_history = {}

        # Start main thread
        threading.Thread(target=main, daemon=True).start()

        gui.run()
    except Exception as e:
        messagebox.showerror("Error: ", str(e))
