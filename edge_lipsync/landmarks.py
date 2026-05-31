from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import cv2
import numpy as np

from edge_lipsync.preprocess import BBox, Point, landmarks_to_duix_roi


class MediaPipeFaceLandmarkerDetector:
    def __init__(
        self,
        *,
        model_asset_path: str | None = None,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        refine_landmarks: bool = True,
    ) -> None:
        try:
            import mediapipe as mp  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "bbox_detector='mediapipe_face_landmarker' requires mediapipe. "
                "Install mediapipe in the dataset build environment."
            ) from exc

        self._mp: Any = mp
        self._face_mesh: Any | None = None
        self._face_landmarker: Any | None = None
        solutions = getattr(mp, "solutions", None)
        face_mesh_module = getattr(solutions, "face_mesh", None)
        if face_mesh_module is not None:
            self._face_mesh = face_mesh_module.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=refine_landmarks,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
            return

        if model_asset_path is None:
            raise FileNotFoundError(
                "Current mediapipe uses Tasks FaceLandmarker and requires "
                "landmark_model_asset_path pointing to face_landmarker.task."
            )

        from mediapipe.tasks.python import vision  # type: ignore[import-not-found]
        from mediapipe.tasks.python.core.base_options import (  # type: ignore[import-not-found]
            BaseOptions,
        )

        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_asset_path),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._face_landmarker = vision.FaceLandmarker.create_from_options(options)

    def detect_landmarks(self, frame_bgr: np.ndarray) -> Mapping[int, Point] | None:
        frame_height, frame_width = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        if self._face_mesh is not None:
            rgb.flags.writeable = False
            results = self._face_mesh.process(rgb)
            if not results.multi_face_landmarks:
                return None
            face_landmarks = results.multi_face_landmarks[0].landmark
        else:
            mp_image = self._mp.Image(
                image_format=self._mp.ImageFormat.SRGB,
                data=np.ascontiguousarray(rgb),
            )
            face_landmarker = self._face_landmarker
            if face_landmarker is None:
                raise RuntimeError("MediaPipe FaceLandmarker is not initialized")
            results = face_landmarker.detect(mp_image)
            if not results.face_landmarks:
                return None
            face_landmarks = results.face_landmarks[0]

        return {
            index: (landmark.x * frame_width, landmark.y * frame_height)
            for index, landmark in enumerate(face_landmarks)
        }

    def detect_bbox(self, frame_bgr: np.ndarray) -> BBox | None:
        landmarks = self.detect_landmarks(frame_bgr)
        if landmarks is None:
            return None
        return landmarks_to_duix_roi(landmarks, frame_bgr.shape)

    def close(self) -> None:
        if self._face_mesh is not None:
            self._face_mesh.close()
        if self._face_landmarker is not None:
            self._face_landmarker.close()


MediaPipeFaceMeshDetector = MediaPipeFaceLandmarkerDetector
