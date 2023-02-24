import math

import cv2
import matplotlib.pyplot as plt
import moviepy.editor as movp
import numpy as np
import mediapipe as mp


class PoseLandmarkExtractor:
    def __init__(self, filename: str = ""):
        if not filename.endswith((".mov")):
            raise ValueError(
                f"{filename} is not a compatible video file for this module."
            )
        self.filename = filename
        self.clip = movp.VideoFileClip(self.filename)
        self.landmarks = None
        self.landmark_overlay = None

    def extract_landmarks(self, num_face: int = 1, show_preview: bool = False):
        cap = cv2.VideoCapture(self.filename)
        mp_draw = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(static_image_mode=False)
        landmark_overlay = []
        success = True
        while success:
            success, img = cap.read()
            try:
                annotated_image = img.copy()
            except:
                break
            results = pose.process(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))

            image_height, image_width, _ = annotated_image.shape
            if not results.pose_landmarks:
                continue

            # Draw pose landmarks.
            annotated_image = annotated_image.copy()
            mp_draw.draw_landmarks(
                annotated_image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
            )
            landmark_overlay.append(annotated_image)
            if show_preview:
                DESIRED_HEIGHT = 480
                DESIRED_WIDTH = 480
                h, w = annotated_image.shape[:2]
                if h < w:
                    img = cv2.resize(
                        annotated_image,
                        (DESIRED_WIDTH, math.floor(h / (w / DESIRED_WIDTH))),
                    )
                else:
                    img = cv2.resize(
                        annotated_image,
                        (math.floor(w / (h / DESIRED_HEIGHT)), DESIRED_HEIGHT),
                    )
                plt.imshow(img)
                plt.show()
        self.landmarks = results
        self.landmark_overlay = landmark_overlay
        return results, landmark_overlay

    def preview_video(self) -> np.ndarray:
        cap = cv2.VideoCapture(self.filename)
        _, img = cap.read()
        plt.imshow(img, interpolation="nearest")
        plt.show()
        return img

    def get_audio(self):
        audio = self.audio.to_soundarray()[:, 0]
        audio_ts = np.linspace(
            self.clip.audio.start,
            self.clip.audio.end,
            int(self.clip.audio.end * self.clip.audio.fps),
        )
        return audio, audio_ts
