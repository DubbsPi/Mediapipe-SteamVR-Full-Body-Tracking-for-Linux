import struct
import time
import math
import os
import socket
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw
import pygame
import argparse

# Initialize MediaPipe
mp_holistic = mp.solutions.holistic

# Common tracker ids
TRACKER_WAIST = 0
TRACKER_LEFT_FOOT = 1
TRACKER_RIGHT_FOOT = 2
TRACKER_LEFT_HAND = 3
TRACKER_RIGHT_HAND = 4
TRACKER_LEFT_ELBOW = 5
TRACKER_RIGHT_ELBOW = 6
TRACKER_LEFT_KNEE = 7
TRACKER_RIGHT_KNEE = 8
TRACKER_CHEST = 9

# Torso
TRACKER_LEFT_SHOULDER = 10
TRACKER_RIGHT_SHOULDER = 11
TRACKER_UPPER_CHEST = 12
TRACKER_NECK = 13
TRACKER_LEFT_HIP = 14
TRACKER_RIGHT_HIP = 15

# Left hand fingers
TRACKER_LEFT_WRIST = 16
TRACKER_LEFT_THUMB = 17
TRACKER_LEFT_INDEX = 18
TRACKER_LEFT_MIDDLE = 19
TRACKER_LEFT_RING = 20
TRACKER_LEFT_PINKY = 21

# Right hand fingers
TRACKER_RIGHT_WRIST = 22
TRACKER_RIGHT_THUMB = 23
TRACKER_RIGHT_INDEX = 24
TRACKER_RIGHT_MIDDLE = 25
TRACKER_RIGHT_RING = 26
TRACKER_RIGHT_PINKY = 27

# Feet detail
TRACKER_LEFT_ANKLE = 28
TRACKER_RIGHT_ANKLE = 29
TRACKER_LEFT_HEEL = 30
TRACKER_RIGHT_HEEL = 31
TRACKER_LEFT_FOOT_INDEX = 32
TRACKER_RIGHT_FOOT_INDEX = 33

class SimpleImageDisplay:
    def __init__(self, width, height, fullscreen=True):
        pygame.init()

        self.width = width
        self.height = height

        # Create window
        if fullscreen:
            self.screen = pygame.display.set_mode((self.width, self.height), pygame.NOFRAME)
        else:
            self.screen = pygame.display.set_mode((self.width, self.height))

        pygame.display.set_caption("Simple Display")

        self.current_image = None
        self.image_x = 0
        self.image_y = 0
        self.clock = pygame.time.Clock()

        blank = pygame.Surface((16, 16), pygame.SRCALPHA)  # fully transparent
        blank_cursor = pygame.cursors.Cursor((0, 0), blank)
        pygame.mouse.set_cursor(blank_cursor)

        print(f"Display initialized: {self.width}x{self.height}")

    def pil_to_pygame(self, pil_image):
        """Convert PIL Image to pygame Surface"""
        if pil_image.mode != 'RGBA':
            pil_image = pil_image.convert('RGBA')

        raw_image = pil_image.tobytes()
        pygame_image = pygame.image.fromstring(raw_image, pil_image.size, 'RGBA')
        return pygame_image

    def load_and_display_image(self, image_path_or_pil):
        """Load and display an image"""
        try:
            if isinstance(image_path_or_pil, str):
                # File path
                image = pygame.image.load(image_path_or_pil)
            else:
                # PIL Image
                image = self.pil_to_pygame(image_path_or_pil)

            # Scale to fit screen
            orig_width, orig_height = image.get_size()
            scale_x = self.width / orig_width
            scale_y = self.height / orig_height
            scale = min(scale_x, scale_y)

            new_width = int(orig_width * scale)
            new_height = int(orig_height * scale)

            self.current_image = pygame.transform.scale(image, (new_width, new_height))
            self.image_x = (self.width - new_width) // 2
            self.image_y = (self.height - new_height) // 2

            # Update display immediately
            self.update_display()

        except Exception as e:
            print(f"Error loading image: {e}")

    def update_display(self):
        """Update the display"""
        self.screen.fill((0, 0, 0))
        if self.current_image:
            self.screen.blit(self.current_image, (self.image_x, self.image_y))
        pygame.display.flip()

    def check_for_quit(self):
        """Check for quit events - call this periodically"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                    return True
        return False

    def quit(self):
        """Clean up and quit"""
        pygame.quit()

class VRDriverClient:
    def __init__(self):
        self.connection = None
        self.connected = False

    def connect(self):
        """Connect to the VR driver"""
        # Linux unix socket
        try:
            self.connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.connection.connect('/tmp/mediapipe_vr.sock')
            self.connected = True
            print("Connected to VR driver!")
            return True
        except Exception as e:
            print(f"Failed to connect: {e}")
            return False

    def send_tracker_data(self, tracker_id, position, quaternion):
        """Send tracker data to the driver

        Args:
            tracker_id: int (0=waist, 1=left_foot, 2=right_foot, etc.)
            position: tuple (x, y, z) in meters
            quaternion: tuple (w, x, y, z) - normalized quaternion
        """
        if not self.connected:
            return False

        # Pack data: tracker_id(int) + position(3 doubles) + quaternion(4 doubles)
        data = struct.pack('i7d',
                           tracker_id,
                           position[0], 1-position[1], position[2],
                           quaternion[0], quaternion[1], quaternion[2], quaternion[3])

        try:
            self.connection.send(data)
            return True
        except Exception as e:
            print(f"Failed to send data: {e}")
            self.connected = False
            return False

    def disconnect(self):
        """Disconnect from the driver"""
        if self.connection:
            self.connection.close()
            self.connected = False
            print("Disconnected from VR driver.")

class LandmarkDrawer:
    def draw_body_shape_pil(draw, landmarks, width, height):
        """Draw a simplified body shape using Pillow's ImageDraw"""
        if not landmarks:
            return

        # Convert normalized coordinates to pixel coordinates
        def get_landmark_pos(landmark_id):
            landmark = landmarks.landmark[landmark_id]
            return int(landmark.x * width), int(landmark.y * height)

        # Key body landmarks for drawing shapes
        try:
            # Body core landmarks
            left_shoulder = get_landmark_pos(11)
            right_shoulder = get_landmark_pos(12)
            left_hip = get_landmark_pos(23)
            right_hip = get_landmark_pos(24)

            # Arms
            left_elbow = get_landmark_pos(13)
            right_elbow = get_landmark_pos(14)
            left_wrist = get_landmark_pos(15)
            right_wrist = get_landmark_pos(16)

            # Legs
            left_knee = get_landmark_pos(25)
            right_knee = get_landmark_pos(26)
            left_ankle = get_landmark_pos(27)
            right_ankle = get_landmark_pos(28)

            # Draw torso (rectangle) - using line width for thickness
            line_width = 3
            draw.line([left_shoulder, right_shoulder], fill=(255, 0, 0), width=line_width)
            draw.line([left_shoulder, left_hip], fill=(255, 0, 0), width=line_width)
            draw.line([right_shoulder, right_hip], fill=(255, 0, 0), width=line_width)
            draw.line([left_hip, right_hip], fill=(255, 0, 0), width=line_width)

            # Draw arms
            draw.line([left_shoulder, left_elbow], fill=(0, 255, 255), width=line_width)
            draw.line([left_elbow, left_wrist], fill=(0, 255, 255), width=line_width)
            draw.line([right_shoulder, right_elbow], fill=(0, 255, 255), width=line_width)
            draw.line([right_elbow, right_wrist], fill=(0, 255, 255), width=line_width)

            # Draw legs
            draw.line([left_hip, left_knee], fill=(255, 255, 0), width=line_width)
            draw.line([left_knee, left_ankle], fill=(255, 255, 0), width=line_width)
            draw.line([right_hip, right_knee], fill=(255, 255, 0), width=line_width)
            draw.line([right_knee, right_ankle], fill=(255, 255, 0), width=line_width)

            # Draw joint circles - using ellipse for circles
            joint_radius = 5
            joints = [left_shoulder, right_shoulder, left_elbow, right_elbow,
                      left_wrist, right_wrist, left_hip, right_hip,
                      left_knee, right_knee, left_ankle, right_ankle]

            for joint in joints:
                x, y = joint
                draw.ellipse([x-joint_radius, y-joint_radius, x+joint_radius, y+joint_radius],
                             fill=(0, 0, 255))

        except IndexError:
            # Some landmarks might not be detected
            pass

    def draw_hand_landmarks_pil(draw, landmarks, connections, width, height, color=(255, 255, 255)):
        """Draw hand landmarks using Pillow"""
        if not landmarks:
            return

        # Convert normalized coordinates to pixel coordinates
        def get_landmark_pos(landmark):
            return int(landmark.x * width), int(landmark.y * height)

        # Draw connections
        for connection in connections:
            start_idx, end_idx = connection
            start_pos = get_landmark_pos(landmarks.landmark[start_idx])
            end_pos = get_landmark_pos(landmarks.landmark[end_idx])
            draw.line([start_pos, end_pos], fill=color, width=2)

        # Draw landmarks
        for landmark in landmarks.landmark:
            pos = get_landmark_pos(landmark)
            x, y = pos
            draw.ellipse([x-2, y-2, x+2, y+2], fill=color)

    def draw_face_landmarks_pil(draw, landmarks, width, height, color=(128, 255, 128)):
        """Draw detailed face landmarks using Pillow"""
        if not landmarks:
            return
        # This function remains unchanged...

def send_all_landmarks_to_vr(vr_client, results):
    """
    Extracts, transforms, and sends all relevant pose landmarks to the VR driver
    This version connects the hand landmarks to the body's wrist position
    """
    # Define a  rotation for trackers
    neutral_rotation = (1.0, 0.0, 0.0, 0.0)

    # We need pose_world_landmarks to anchor the hands, so exit if it's not available
    if not vr_client.connected or not results.pose_world_landmarks:
        return

    # --- 1. Process Body Landmarks (using world coordinates) ---
    pose_landmarks = results.pose_world_landmarks.landmark

    def get_transformed_pos(landmark_id):
        lm = pose_landmarks[landmark_id]
        # Invert Z-axis to match VR's coordinate system
        return np.array([lm.x, lm.y, -lm.z])

    # Map and send main body trackers
    body_tracker_map = {
        TRACKER_LEFT_FOOT: mp_holistic.PoseLandmark.LEFT_FOOT_INDEX,
        TRACKER_RIGHT_FOOT: mp_holistic.PoseLandmark.RIGHT_FOOT_INDEX,
        TRACKER_LEFT_ELBOW: mp_holistic.PoseLandmark.LEFT_ELBOW,
        TRACKER_RIGHT_ELBOW: mp_holistic.PoseLandmark.RIGHT_ELBOW,
        TRACKER_LEFT_KNEE: mp_holistic.PoseLandmark.LEFT_KNEE,
        TRACKER_RIGHT_KNEE: mp_holistic.PoseLandmark.RIGHT_KNEE,
    }
    for tracker_id, landmark_id in body_tracker_map.items():
        pos = get_transformed_pos(landmark_id.value)
        vr_client.send_tracker_data(tracker_id, tuple(pos), neutral_rotation)

    # Handle composite trackers (waist and chest)
    left_hip_pos = get_transformed_pos(mp_holistic.PoseLandmark.LEFT_HIP.value)
    right_hip_pos = get_transformed_pos(mp_holistic.PoseLandmark.RIGHT_HIP.value)
    waist_pos = np.mean([left_hip_pos, right_hip_pos], axis=0)
    vr_client.send_tracker_data(TRACKER_WAIST, tuple(waist_pos), neutral_rotation)

    left_shoulder_pos = get_transformed_pos(mp_holistic.PoseLandmark.LEFT_SHOULDER.value)
    right_shoulder_pos = get_transformed_pos(mp_holistic.PoseLandmark.RIGHT_SHOULDER.value)
    chest_pos = np.mean([left_shoulder_pos, right_shoulder_pos], axis=0)
    vr_client.send_tracker_data(TRACKER_CHEST, tuple(chest_pos), neutral_rotation)

    # --- 2. Process Left Hand (Anchored to Body Wrist) ---
    if results.left_hand_landmarks:
        hand_landmarks = results.left_hand_landmarks.landmark

        # Get the world position of the wrist (our anchor)
        wrist_world_pos = get_transformed_pos(mp_holistic.PoseLandmark.LEFT_WRIST.value)

        # Get hand-space positions
        wrist_hand_pos = np.array([hand_landmarks[0].x, hand_landmarks[0].y, hand_landmarks[0].z])
        pinky_mcp_hand_pos = np.array([hand_landmarks[17].x, hand_landmarks[17].y, hand_landmarks[17].z])

        # Get world-space positions for scaling reference
        pinky_mcp_world_pos = get_transformed_pos(mp_holistic.PoseLandmark.LEFT_PINKY.value)

        # Calculate distance in both spaces to find the scaling factor
        dist_hand = np.linalg.norm(wrist_hand_pos - pinky_mcp_hand_pos)
        dist_world = np.linalg.norm(wrist_world_pos - pinky_mcp_world_pos)

        scale_factor = dist_world / dist_hand if dist_hand > 0 else 1.0

        left_hand_map = {
            TRACKER_LEFT_WRIST: mp_holistic.HandLandmark.WRIST,
            TRACKER_LEFT_THUMB: mp_holistic.HandLandmark.THUMB_TIP,
            TRACKER_LEFT_INDEX: mp_holistic.HandLandmark.INDEX_FINGER_TIP,
            TRACKER_LEFT_MIDDLE: mp_holistic.HandLandmark.MIDDLE_FINGER_TIP,
            TRACKER_LEFT_RING: mp_holistic.HandLandmark.RING_FINGER_TIP,
            TRACKER_LEFT_PINKY: mp_holistic.HandLandmark.PINKY_TIP,
        }

        for tracker_id, landmark_id in left_hand_map.items():
            lm = hand_landmarks[landmark_id.value]
            # Get the finger position in hand-space
            finger_hand_pos = np.array([lm.x, lm.y, lm.z])
            # Calculate the offset from the wrist in hand-space
            offset_vector = finger_hand_pos - wrist_hand_pos
            # Apply scaling and add to the world-space wrist anchor
            # We flip the Y and Z of the offset to match the world coordinate system
            final_pos = wrist_world_pos + (np.array([offset_vector[0], offset_vector[1], offset_vector[2]]) * scale_factor)
            vr_client.send_tracker_data(tracker_id, tuple(final_pos), neutral_rotation)

    # --- 3. Process Right Hand (Anchored to Body Wrist) ---
    if results.right_hand_landmarks:
        hand_landmarks = results.right_hand_landmarks.landmark

        # Get the world position of the wrist (our anchor)
        wrist_world_pos = get_transformed_pos(mp_holistic.PoseLandmark.RIGHT_WRIST.value)

        # Get hand-space positions
        wrist_hand_pos = np.array([hand_landmarks[0].x, hand_landmarks[0].y, hand_landmarks[0].z])
        pinky_mcp_hand_pos = np.array([hand_landmarks[17].x, hand_landmarks[17].y, hand_landmarks[17].z])

        # Get world-space positions for scaling reference
        pinky_mcp_world_pos = get_transformed_pos(mp_holistic.PoseLandmark.RIGHT_PINKY.value)

        # Calculate distance in both spaces to find the scaling factor
        dist_hand = np.linalg.norm(wrist_hand_pos - pinky_mcp_hand_pos)
        dist_world = np.linalg.norm(wrist_world_pos - pinky_mcp_world_pos)

        scale_factor = dist_world / dist_hand if dist_hand > 0 else 1.0

        right_hand_map = {
            TRACKER_RIGHT_WRIST: mp_holistic.HandLandmark.WRIST,
            TRACKER_RIGHT_THUMB: mp_holistic.HandLandmark.THUMB_TIP,
            TRACKER_RIGHT_INDEX: mp_holistic.HandLandmark.INDEX_FINGER_TIP,
            TRACKER_RIGHT_MIDDLE: mp_holistic.HandLandmark.MIDDLE_FINGER_TIP,
            TRACKER_RIGHT_RING: mp_holistic.HandLandmark.RING_FINGER_TIP,
            TRACKER_RIGHT_PINKY: mp_holistic.HandLandmark.PINKY_TIP,
        }

        for tracker_id, landmark_id in right_hand_map.items():
            lm = hand_landmarks[landmark_id.value]
            finger_hand_pos = np.array([lm.x, lm.y, lm.z])
            offset_vector = finger_hand_pos - wrist_hand_pos
            final_pos = wrist_world_pos + (np.array([offset_vector[0], offset_vector[1], offset_vector[2]]) * scale_factor)
            vr_client.send_tracker_data(tracker_id, tuple(final_pos), neutral_rotation)



def setup_arguments():
    parser = argparse.ArgumentParser(description="A simple vr tracking tool for steam using mediapipe")

    parser.add_argument("--camera", type=str, default=0, help="Camera id or ip address")
    parser.add_argument("--width", type=int, default=None, help="Camera width, should be automatic")
    parser.add_argument("--height", type=int, default=None, help="Camera height, should be automatic")
    parser.add_argument("--model", type=int, default=2, help="Model complexity ranging from 0 to 2, with 2 being the heaviest to run")
    parser.add_argument("--head", type=bool, default=False, help="Shows location and shape of face, should not be needed for tracking")

    return parser.parse_args()

def main():

    args = setup_arguments()

    # Initialize and connect the VR client
    vr_client = VRDriverClient()
    vr_client.connect()

    # Setup camera
    try:
        camera=int(args.camera)
    except:
        camera=args.camera

    print(f"Camera: {camera}")
    cap = cv2.VideoCapture(camera)

    # Get actual camera dimensions first
    ret, test_frame = cap.read()
    if not ret:
        print("Failed to get test frame from camera")
        return

    HEIGHT, WIDTH = args.height, args.width

    if HEIGHT is None or WIDTH is None:
        # Use camera dimensions
        HEIGHT, WIDTH = test_frame.shape[:2]
        print(f"Camera dimensions: {HEIGHT}x{WIDTH}")
    else:
        print(f"Chosen dimenstions: {HEIGHT}x{WIDTH}")

    # Set camera properties to match what we have
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    # Setup display with chosen dimensions
    display = SimpleImageDisplay(WIDTH, HEIGHT, False)

    # Initialize MediaPipe Holistic with heavy model
    with mp_holistic.Holistic(
        min_detection_confidence=0.7,      # Higher confidence threshold
        min_tracking_confidence=0.5,
        model_complexity=args.model,       # Use heavy model (0=lite, 1=full, 2=heavy)
        smooth_landmarks=True,             # Enable smoothing
        enable_segmentation=True,          # Enables body segmentation mask
        smooth_segmentation=True,          # Smooth the segmentation
    ) as holistic:

        frame_count = 0

        try:
            while True:
                ret, video_frame = cap.read()
                if not ret:
                    print("Failed to grab frame")
                    continue

                # Get actual frame dimensions for this frame
                frame_height, frame_width = video_frame.shape[:2]

                # Flip horizontally for mirror effect (optional)
                video_frame = cv2.flip(video_frame, 1)

                # Convert BGR to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB)

                # Process the frame with MediaPipe
                results = holistic.process(rgb_frame)

                # Send data to the VR side
                send_all_landmarks_to_vr(vr_client, results)

                # Create a PIL image for drawing using the frame dimensions
                pil_frame = Image.fromarray(rgb_frame)
                draw = ImageDraw.Draw(pil_frame)

                # Use frame dimensions for coordinate conversion
                if results.pose_landmarks:
                    LandmarkDrawer.draw_body_shape_pil(draw, results.pose_landmarks, frame_width, frame_height)

                # Draw hand landmarks
                if results.left_hand_landmarks:
                    LandmarkDrawer.draw_hand_landmarks_pil(draw, results.left_hand_landmarks,
                                                          mp_holistic.HAND_CONNECTIONS, frame_width, frame_height,
                                                          color=(255, 255, 255))
                if results.right_hand_landmarks:
                    LandmarkDrawer.draw_hand_landmarks_pil(draw, results.right_hand_landmarks,
                                                          mp_holistic.HAND_CONNECTIONS, frame_width, frame_height,
                                                          color=(255, 255, 255))

                if args.head:
                    # Draw face landmarks
                    if results.face_landmarks:
                        LandmarkDrawer.draw_face_landmarks_pil(draw, results.face_landmarks, frame_width, frame_height)

                # Show the frame
                display.load_and_display_image(pil_frame)

                # Check for quit
                if display.check_for_quit() or cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            # Close display on exit
            display.quit()
            # Disconnect the VR client on exit
            vr_client.disconnect()

if __name__ == "__main__":
    main()
