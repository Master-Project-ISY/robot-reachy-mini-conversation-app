"""Tests for utility helpers."""

import argparse
import sys
from unittest.mock import MagicMock, patch

import pytest

from reachy_mini_conversation_app.utils import (
    CameraVisionInitializationError,
    initialize_camera_and_vision,
)


def test_initialize_camera_and_vision_propagates_local_vision_init_failures() -> None:
    """Explicit local vision requests should preserve unexpected initialization errors."""
    args = argparse.Namespace(
        no_camera=False,
        head_tracker=None,
        local_vision=True,
    )

    with (
        patch("reachy_mini_conversation_app.utils.CameraWorker") as mock_camera_worker,
        patch("reachy_mini_conversation_app.utils.subprocess.run", return_value=MagicMock(returncode=0)),
        patch(
            "reachy_mini_conversation_app.vision.local_vision.initialize_vision_processor",
            side_effect=RuntimeError("Vision processor initialization failed"),
        ),
    ):
        with pytest.raises(RuntimeError, match="Vision processor initialization failed"):
            initialize_camera_and_vision(args, MagicMock())

    mock_camera_worker.assert_called_once()


def test_initialize_camera_and_vision_raises_when_local_vision_import_crashes() -> None:
    """Explicit local vision requests should fail cleanly on native import crashes."""
    args = argparse.Namespace(
        no_camera=False,
        head_tracker=None,
        local_vision=True,
    )

    with (
        patch("reachy_mini_conversation_app.utils.CameraWorker") as mock_camera_worker,
        patch("reachy_mini_conversation_app.utils.subprocess.run", return_value=MagicMock(returncode=-4)),
    ):
        with pytest.raises(CameraVisionInitializationError, match="Local vision import crashed"):
            initialize_camera_and_vision(args, MagicMock())

    mock_camera_worker.assert_called_once()


def test_initialize_camera_and_vision_raises_when_head_tracker_init_fails() -> None:
    """Head-tracker startup failures should be reported through the clean init error path."""
    args = argparse.Namespace(
        no_camera=False,
        head_tracker="yolo",
        local_vision=False,
    )

    with (
        patch("reachy_mini_conversation_app.utils.CameraWorker") as mock_camera_worker,
        patch(
            "reachy_mini_conversation_app.vision.head_tracking.yolo_process.YoloHeadTrackerProcess",
            side_effect=RuntimeError("tracker init failed"),
        ),
    ):
        with pytest.raises(
            CameraVisionInitializationError,
            match="Failed to initialize yolo head tracker: tracker init failed",
        ):
            initialize_camera_and_vision(args, MagicMock())

    mock_camera_worker.assert_not_called()


def test_initialize_camera_and_vision_uses_mediapipe_head_tracker_in_process() -> None:
    """MediaPipe head tracking should use the in-process toolbox tracker."""
    args = argparse.Namespace(
        no_camera=False,
        head_tracker="mediapipe",
        local_vision=False,
    )

    current_robot = MagicMock()
    mediapipe_head_tracker = MagicMock()
    with (
        patch("reachy_mini_conversation_app.utils.CameraWorker") as mock_camera_worker,
        patch(
            "reachy_mini_conversation_app.vision.head_tracking.mediapipe.MediapipeHeadTracker",
            return_value=mediapipe_head_tracker,
        ),
    ):
        initialize_camera_and_vision(args, current_robot)

    mock_camera_worker.assert_called_once_with(current_robot, mediapipe_head_tracker, None)


def test_initialize_camera_and_vision_uses_webcam_when_sim_cam_set() -> None:
    """When --sim-cam is set, a webcam frame source callable is passed to CameraWorker."""
    args = argparse.Namespace(
        no_camera=False,
        head_tracker=None,
        local_vision=False,
        sim_cam=0,
    )
    mock_cv2 = MagicMock()
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cv2.VideoCapture.return_value = mock_cap
    current_robot = MagicMock()

    with (
        patch("reachy_mini_conversation_app.utils.CameraWorker") as mock_camera_worker,
        patch.dict(sys.modules, {"cv2": mock_cv2}),
    ):
        initialize_camera_and_vision(args, current_robot)

    mock_cv2.VideoCapture.assert_called_once_with(0)
    call_args = mock_camera_worker.call_args[0]
    assert call_args[0] is current_robot
    assert call_args[1] is None  # no head tracker
    assert callable(call_args[2])  # frame source passed


def test_initialize_camera_and_vision_raises_when_sim_cam_not_opened() -> None:
    """When --sim-cam names an inaccessible camera, a clean error is raised."""
    args = argparse.Namespace(
        no_camera=False,
        head_tracker=None,
        local_vision=False,
        sim_cam=5,
    )
    mock_cv2 = MagicMock()
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = False
    mock_cv2.VideoCapture.return_value = mock_cap

    with (
        patch("reachy_mini_conversation_app.utils.CameraWorker"),
        patch.dict(sys.modules, {"cv2": mock_cv2}),
        pytest.raises(CameraVisionInitializationError, match="Could not open webcam at index 5"),
    ):
        initialize_camera_and_vision(args, MagicMock())
