import math

import cv2
import matplotlib.pyplot as plt
import moviepy.editor as movp
import numpy as np
import mediapipe as mp


class FaceLandmarkExtractor:
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
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(max_num_faces=num_face)
        landmark_overlay = []
        success = True
        while success:
            success, img = cap.read()
            try:
                imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            except:
                break
            results = face_mesh.process(imgRGB)
            annotated_image = img.copy()
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    mp_draw.draw_landmarks(
                        image=annotated_image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),
                    )
                    mp_draw.draw_landmarks(
                        image=annotated_image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style(),
                    )
                    try:
                        mp_draw.draw_landmarks(
                            image=annotated_image,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_IRISES,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style(),
                        )
                    except:
                        pass
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
