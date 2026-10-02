"""Semi-transparent statistics panel drawn over the video feed."""

from __future__ import annotations

import cv2
import numpy as np

from facedetected.config import HudConfig

_PANEL_BG = (30, 22, 18)
_TEXT_MAIN = (240, 245, 245)
_TEXT_ACCENT = (80, 200, 255)
_TEXT_DIM = (150, 160, 160)
_FONT = cv2.FONT_HERSHEY_SIMPLEX


class Hud:
    """Collects per-frame stats and renders them as a translucent panel.

    The panel is toggled at runtime (``H`` key) and every line can be
    disabled through :class:`~facedetected.config.HudConfig`.
    """

    def __init__(self, cfg: HudConfig | None = None) -> None:
        self.cfg = cfg or HudConfig()
        self.enabled = self.cfg.enabled
        self._lines: list[tuple[str, tuple[int, int, int]]] = []

    def update(
        self,
        *,
        fps: float | None = None,
        face_count: int | None = None,
        blink_count: int | None = None,
        mouth_open: bool | None = None,
        head_pose: str | None = None,
        extra: list[tuple[str, str]] | None = None,
    ) -> None:
        self._lines = []
        if self.cfg.show_fps and fps is not None:
            self._lines.append((f"FPS: {fps:5.1f}", _TEXT_ACCENT))
        if self.cfg.show_face_count and face_count is not None:
            self._lines.append((f"faces: {face_count}", _TEXT_MAIN))
        if self.cfg.show_blink_count and blink_count is not None:
            self._lines.append((f"blinks: {blink_count}", _TEXT_MAIN))
        if self.cfg.show_mouth_state and mouth_open is not None:
            state = "OPEN" if mouth_open else "closed"
            self._lines.append((f"mouth: {state}", _TEXT_MAIN))
        if self.cfg.show_head_pose and head_pose is not None:
            self._lines.append((f"pose: {head_pose}", _TEXT_MAIN))
        for key, value in extra or []:
            self._lines.append((f"{key}: {value}", _TEXT_MAIN))

    def toggle(self) -> None:
        self.enabled = not self.enabled

    def draw(self, img: np.ndarray, *, recording: bool = False) -> np.ndarray:
        """Render the panel onto ``img`` (returns img for chaining)."""
        if not self.enabled or not self._lines:
            return img

        pad, line_h, scale = 10, 22, 0.55
        width = 0
        for text, _ in self._lines:
            (tw, _), _ = cv2.getTextSize(text, _FONT, scale, 1)
            width = max(width, tw)
        height = pad * 2 + line_h * len(self._lines)
        width += pad * 2

        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (width, height), _PANEL_BG, cv2.FILLED)
        cv2.addWeighted(overlay, self.cfg.panel_opacity, img, 1.0 - self.cfg.panel_opacity, 0, img)

        y = pad + 15
        for text, color in self._lines:
            cv2.putText(img, text, (pad, y), _FONT, scale, color, 1, cv2.LINE_AA)
            y += line_h

        if recording:
            cv2.circle(img, (width - 16, height - 16), 6, (60, 60, 240), -1, cv2.LINE_AA)
            cv2.putText(img, "REC", (width - 60, height - 10), _FONT, 0.5, (60, 60, 240), 1, cv2.LINE_AA)
        return img

    def draw_help_footer(self, img: np.ndarray) -> np.ndarray:
        """One-line key legend pinned to the bottom-left."""
        text = "q quit | s snapshot | r record | m mesh | b boxes | h hud | i ids"
        (tw, th), _ = cv2.getTextSize(text, _FONT, 0.45, 1)
        h, w = img.shape[:2]
        overlay = img.copy()
        cv2.rectangle(overlay, (0, h - th - 14), (tw + 12, h), _PANEL_BG, cv2.FILLED)
        cv2.addWeighted(overlay, 0.45, img, 0.55, 0, img)
        cv2.putText(img, text, (6, h - 8), _FONT, 0.45, _TEXT_DIM, 1, cv2.LINE_AA)
        return img
