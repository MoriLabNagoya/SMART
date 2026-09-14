# -*- coding: utf-8 -*-
from __future__ import annotations

import colorsys
import csv
import math
import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import QPoint, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QAction,
    QActionGroup,
    QColor,
    QFont,
    QImage,
    QKeySequence,
    QPainter,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QGraphicsPixmapItem,
    QGraphicsLineItem,
    QGraphicsSimpleTextItem,
    QGraphicsScene,
    QGraphicsView,
    QHeaderView,
    QLabel,
    QListWidget,
    QComboBox,
    QLineEdit,
    QPushButton,
    QToolButton,
    QSplitter,
    QTabWidget,
    QHBoxLayout,
    QVBoxLayout,
    QWidget,
    QMainWindow,
    QMessageBox,
    QSlider,
    QSizePolicy,
    QSpinBox,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QToolBar,
)


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def label_color(label_value: int) -> QColor:
    """为 1~255 的标签值生成稳定且较易区分的颜色。"""
    value = max(1, min(255, int(label_value)))
    hue = (value * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.78, 1.0)
    return QColor(int(r * 255), int(g * 255), int(b * 255))


def numpy_to_qimage(image: np.ndarray) -> QImage:
    if image.ndim == 2:
        arr = np.ascontiguousarray(image)
        h, w = arr.shape
        return QImage(
            arr.data, w, h, arr.strides[0], QImage.Format_Grayscale8
        ).copy()

    if image.ndim == 3 and image.shape[2] == 3:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)
        h, w, _ = rgb.shape
        return QImage(
            rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888
        ).copy()

    raise ValueError(f"不支持的图像形状：{image.shape}")


def read_image(path: Path, flags: int) -> Optional[np.ndarray]:
    try:
        raw = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    return cv2.imdecode(raw, flags)


def write_image(path: Path, image: np.ndarray) -> bool:
    suffix = path.suffix or ".png"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        return False
    try:
        encoded.tofile(str(path))
        return True
    except OSError:
        return False


class AnnotationView(QGraphicsView):
    maskChanged = Signal()
    cursorPositionChanged = Signal(int, int, int)
    zoomChanged = Signal(float)
    labelPicked = Signal(int)
    brushSizeRequested = Signal(int)
    labelValueRequested = Signal(int)
    measurementCreated = Signal(float)
    measurementsChanged = Signal()
    measurementUndoPerformed = Signal(str)
    fillCompleted = Signal(int)
    historyStepRequested = Signal(str)

    # YOLO rectangle / OBB annotations.
    yoloAnnotationsChanged = Signal()
    yoloSelectionChanged = Signal(int, int, str)  # index, class_id, type

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene().addItem(self.pixmap_item)

        self.image_bgr: Optional[np.ndarray] = None
        self.mask: Optional[np.ndarray] = None

        self.mode = "paint"
        self.current_label = 1
        self.brush_size = 24
        self.overlay_alpha = 0.45
        self.overlay_visible = True
        self.resolution_um_per_pixel: Optional[float] = None
        self.measurement_start: Optional[QPoint] = None
        self.measurement_current: Optional[QPoint] = None
        self.measurement_items: list[tuple[QGraphicsLineItem, QGraphicsSimpleTextItem]] = []
        self.measurement_records: list[dict] = []
        self.selected_measurement_index = -1
        # When an existing measurement endpoint is undone, remember its original
        # list position so the corrected measurement returns to the same ID.
        self.pending_measurement_index: Optional[int] = None

        # Annotation dictionaries use:
        # {"type": "rect"|"obb", "class_id": int, "points": np.ndarray(4,2)}
        self.yolo_annotations: list[dict] = []
        self.selected_yolo_index = -1
        self.current_yolo_class = 0
        self._box_start: Optional[QPoint] = None
        self._box_current: Optional[QPoint] = None

        self._drawing = False
        self._panning = False
        self._last_image_point: Optional[QPoint] = None
        self._last_pan_pos: Optional[QPoint] = None

        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setBackgroundBrush(QColor(35, 35, 35))
        self.setCursor(Qt.CrossCursor)

    def has_image(self) -> bool:
        return self.image_bgr is not None and self.mask is not None

    def set_image(self, image_bgr: np.ndarray) -> None:
        if image_bgr is None or image_bgr.size == 0:
            raise ValueError("图像无效。")
        self.image_bgr = image_bgr.copy()
        h, w = image_bgr.shape[:2]
        self.mask = np.zeros((h, w), dtype=np.uint8)
        self.yolo_annotations = []
        self.selected_yolo_index = -1
        self._box_start = None
        self._box_current = None
        self.measurement_start = None
        self.measurement_current = None
        self.selected_measurement_index = -1
        self.pending_measurement_index = None
        self.resetTransform()
        self.refresh()
        self.fit_image()

    def set_mask(self, mask: np.ndarray, emit_change: bool = True) -> None:
        if self.image_bgr is None:
            raise RuntimeError("请先打开彩色图像。")

        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        h, w = self.image_bgr.shape[:2]
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        self.mask = np.clip(mask, 0, 255).astype(np.uint8)
        self.refresh()
        if emit_change:
            self.maskChanged.emit()

    def set_mode(self, mode: str) -> None:
        if mode not in {"paint", "erase", "pan", "measure", "bbox", "obb"}:
            raise ValueError(mode)
        if self.mode == "measure" and mode != "measure":
            self.measurement_start = None
            self.measurement_current = None
            self.pending_measurement_index = None
        self.mode = mode
        self._box_start = None
        self._box_current = None
        self.setCursor(Qt.OpenHandCursor if mode == "pan" else Qt.CrossCursor)
        self.viewport().update()

    def set_current_label(self, value: int) -> None:
        self.current_label = max(1, min(255, int(value)))
        self.viewport().update()

    def set_current_yolo_class(self, value: int) -> None:
        value = max(0, int(value))
        self.current_yolo_class = value
        if 0 <= self.selected_yolo_index < len(self.yolo_annotations):
            annotation = self.yolo_annotations[self.selected_yolo_index]
            if int(annotation["class_id"]) != value:
                annotation["class_id"] = value
                self.yoloAnnotationsChanged.emit()
                self.yoloSelectionChanged.emit(
                    self.selected_yolo_index, value, str(annotation["type"])
                )
                self.viewport().update()

    def set_yolo_annotations(self, annotations: list[dict]) -> None:
        copied = []
        for annotation in annotations:
            points = np.asarray(annotation.get("points"), dtype=float).reshape(4, 2)
            copied.append({
                "type": "obb" if annotation.get("type") == "obb" else "rect",
                "class_id": max(0, int(annotation.get("class_id", 0))),
                "points": points.copy(),
            })
        self.yolo_annotations = copied
        self.selected_yolo_index = -1
        self.viewport().update()

    def get_yolo_annotations(self) -> list[dict]:
        return [
            {
                "type": str(annotation["type"]),
                "class_id": int(annotation["class_id"]),
                "points": np.asarray(annotation["points"], dtype=float).copy(),
            }
            for annotation in self.yolo_annotations
        ]

    def set_brush_size(self, size: int) -> None:
        self.brush_size = max(1, int(size))
        self.viewport().update()

    def set_overlay_alpha(self, value: int) -> None:
        self.overlay_alpha = float(np.clip(value / 100.0, 0.0, 1.0))
        self.refresh()

    def set_overlay_visible(self, visible: bool) -> None:
        self.overlay_visible = bool(visible)
        self.refresh()

    def set_resolution(self, resolution_um_per_pixel: Optional[float]) -> None:
        self.resolution_um_per_pixel = resolution_um_per_pixel
        self._refresh_measurement_labels()

    def clear_measurements(self) -> None:
        self.measurement_start = None
        self.measurement_current = None
        self.selected_measurement_index = -1
        self.pending_measurement_index = None
        for line_item, text_item in self.measurement_items:
            self.scene().removeItem(line_item)
            self.scene().removeItem(text_item)
        self.measurement_items.clear()
        self.measurement_records.clear()
        self.viewport().update()

    def _measurement_text(self, length_pixels: float, index: Optional[int] = None) -> str:
        # 已完成的测量只在图像上显示编号，完整数据放在右侧测量列表。
        # 实时预览（index is None）仍显示当前 px / μm 长度。
        if index is not None:
            return f"ID {index + 1}"
        if self.resolution_um_per_pixel is None:
            return f"{length_pixels:.2f} px"
        physical = length_pixels * self.resolution_um_per_pixel
        return f"{length_pixels:.2f} px | {physical:.4f} μm"

    def _measurement_style(self, index: int) -> tuple[QColor, float]:
        if index == self.selected_measurement_index:
            return QColor(255, 230, 0), 3.0
        return QColor(0, 255, 255), 2.0

    def _add_measurement_graphic(self, record: dict, index: int) -> None:
        x1, y1 = int(record["x1"]), int(record["y1"])
        x2, y2 = int(record["x2"]), int(record["y2"])
        length_pixels = math.hypot(x2 - x1, y2 - y1)
        color, width = self._measurement_style(index)
        pen = QPen(color)
        pen.setWidthF(max(1.0, width / max(self.transform().m11(), 0.01)))
        line_item = QGraphicsLineItem(x1, y1, x2, y2)
        line_item.setPen(pen)
        line_item.setZValue(10)
        self.scene().addItem(line_item)
        text_item = QGraphicsSimpleTextItem(self._measurement_text(length_pixels, index))
        text_item.setBrush(color)
        text_item.setPos((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        text_item.setZValue(11)
        self.scene().addItem(text_item)
        self.measurement_items.append((line_item, text_item))

    def set_selected_measurement(self, index: int) -> None:
        index = int(index)
        self.selected_measurement_index = (
            index if 0 <= index < len(self.measurement_records) else -1
        )
        self._update_measurement_selection_style()

    def _update_measurement_selection_style(self) -> None:
        scale = max(self.transform().m11(), 0.01)
        for index, (line_item, text_item) in enumerate(self.measurement_items):
            color, width = self._measurement_style(index)
            pen = QPen(color)
            pen.setWidthF(max(1.0, width / scale))
            line_item.setPen(pen)
            text_item.setBrush(color)
        self.viewport().update()

    def set_measurements(self, records: list[dict], selected_index: Optional[int] = None) -> None:
        previous_selection = self.selected_measurement_index if selected_index is None else int(selected_index)
        self.measurement_start = None
        self.measurement_current = None
        self.pending_measurement_index = None
        for line_item, text_item in self.measurement_items:
            self.scene().removeItem(line_item)
            self.scene().removeItem(text_item)
        self.measurement_items.clear()
        self.measurement_records = [
            {
                "x1": int(item["x1"]), "y1": int(item["y1"]),
                "x2": int(item["x2"]), "y2": int(item["y2"]),
            }
            for item in records
        ]
        self.selected_measurement_index = (
            previous_selection if 0 <= previous_selection < len(self.measurement_records) else -1
        )
        for index, record in enumerate(self.measurement_records):
            self._add_measurement_graphic(record, index)
        self.viewport().update()

    def get_measurements(self) -> list[dict]:
        return [dict(item) for item in self.measurement_records]

    def _refresh_measurement_labels(self) -> None:
        for index, (line_item, text_item) in enumerate(self.measurement_items):
            line = line_item.line()
            length_pixels = math.hypot(line.x2() - line.x1(), line.y2() - line.y1())
            text_item.setText(self._measurement_text(length_pixels, index))

    def clear_mask(self) -> None:
        if self.mask is None:
            return
        self.mask.fill(0)
        self.refresh()
        self.maskChanged.emit()

    def create_blend(self) -> Optional[np.ndarray]:
        if not self.has_image():
            return None

        result = self.image_bgr.copy()
        if not self.overlay_visible:
            return result

        for value in np.unique(self.mask):
            value = int(value)
            if value == 0:
                continue

            selected = self.mask == value
            color = label_color(value)
            overlay_bgr = np.array(
                [color.blue(), color.green(), color.red()], dtype=np.float32
            )
            base = result[selected].astype(np.float32)
            mixed = base * (1.0 - self.overlay_alpha) + overlay_bgr * self.overlay_alpha
            result[selected] = np.clip(mixed, 0, 255).astype(np.uint8)

        return result

    def refresh(self) -> None:
        blend = self.create_blend()
        if blend is None:
            self.pixmap_item.setPixmap(QPixmap())
            return

        pixmap = QPixmap.fromImage(numpy_to_qimage(blend))
        self.pixmap_item.setPixmap(pixmap)
        self.scene().setSceneRect(QRectF(pixmap.rect()))
        self.viewport().update()

    def fit_image(self) -> None:
        if self.has_image():
            self.fitInView(self.pixmap_item, Qt.KeepAspectRatio)
            self.zoomChanged.emit(self.transform().m11())

    def actual_size(self) -> None:
        if self.has_image():
            self.resetTransform()
            self.zoomChanged.emit(1.0)

    def zoom_by(self, factor: float) -> None:
        if not self.has_image():
            return
        target = self.transform().m11() * factor
        if 0.05 <= target <= 40.0:
            self.scale(factor, factor)
            self.zoomChanged.emit(self.transform().m11())

    def _image_point(self, viewport_pos: QPoint) -> Optional[QPoint]:
        if not self.has_image():
            return None

        item_pos = self.pixmap_item.mapFromScene(self.mapToScene(viewport_pos))
        x, y = int(item_pos.x()), int(item_pos.y())
        h, w = self.mask.shape

        if 0 <= x < w and 0 <= y < h:
            return QPoint(x, y)
        return None

    def _draw_segment(self, p1: QPoint, p2: QPoint) -> None:
        if self.mask is None:
            return

        value = 0 if self.mode == "erase" else self.current_label
        thickness = max(1, self.brush_size)
        cv2.line(
            self.mask,
            (p1.x(), p1.y()),
            (p2.x(), p2.y()),
            int(value),
            thickness=thickness,
            lineType=cv2.LINE_8,
        )
        cv2.circle(
            self.mask,
            (p2.x(), p2.y()),
            max(1, thickness // 2),
            int(value),
            thickness=-1,
            lineType=cv2.LINE_8,
        )
        self.refresh()

    def _flood_fill_zero_region(self, point: QPoint) -> int:
        if self.mask is None:
            return 0

        x, y = point.x(), point.y()
        if not (0 <= y < self.mask.shape[0] and 0 <= x < self.mask.shape[1]):
            return 0
        if int(self.mask[y, x]) != 0:
            return 0

        label_value = int(self.current_label)
        working = self.mask.copy()
        flood_mask = np.zeros(
            (working.shape[0] + 2, working.shape[1] + 2), dtype=np.uint8
        )
        filled_pixels, _, _, _ = cv2.floodFill(
            working,
            flood_mask,
            seedPoint=(x, y),
            newVal=label_value,
            loDiff=0,
            upDiff=0,
            flags=4 | cv2.FLOODFILL_FIXED_RANGE,
        )
        if filled_pixels > 0:
            self.mask = working
            self.refresh()
            self.maskChanged.emit()
        return int(filled_pixels)

    def _select_yolo_at(self, point: QPoint) -> int:
        p = (float(point.x()), float(point.y()))
        for index in range(len(self.yolo_annotations) - 1, -1, -1):
            polygon = np.asarray(self.yolo_annotations[index]["points"], dtype=np.float32)
            if cv2.pointPolygonTest(polygon, p, False) >= 0:
                self.selected_yolo_index = index
                annotation = self.yolo_annotations[index]
                self.yoloSelectionChanged.emit(
                    index, int(annotation["class_id"]), str(annotation["type"])
                )
                self.viewport().update()
                return index
        self.selected_yolo_index = -1
        self.yoloSelectionChanged.emit(-1, self.current_yolo_class, "")
        self.viewport().update()
        return -1

    def _create_yolo_box(self, start: QPoint, end: QPoint, annotation_type: str) -> bool:
        x1, x2 = sorted((float(start.x()), float(end.x())))
        y1, y2 = sorted((float(start.y()), float(end.y())))
        if x2 - x1 < 3.0 or y2 - y1 < 3.0:
            return False
        points = np.array(
            [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=float
        )
        self.yolo_annotations.append({
            "type": "obb" if annotation_type == "obb" else "rect",
            "class_id": int(self.current_yolo_class),
            "points": points,
        })
        self.selected_yolo_index = len(self.yolo_annotations) - 1
        self.yoloAnnotationsChanged.emit()
        self.yoloSelectionChanged.emit(
            self.selected_yolo_index, self.current_yolo_class, annotation_type
        )
        self.viewport().update()
        return True

    def delete_selected_yolo(self) -> bool:
        if not (0 <= self.selected_yolo_index < len(self.yolo_annotations)):
            return False
        del self.yolo_annotations[self.selected_yolo_index]
        self.selected_yolo_index = -1
        self.yoloAnnotationsChanged.emit()
        self.yoloSelectionChanged.emit(-1, self.current_yolo_class, "")
        self.viewport().update()
        return True

    def rotate_selected_obb(self, degrees: float) -> bool:
        if not (0 <= self.selected_yolo_index < len(self.yolo_annotations)):
            return False
        annotation = self.yolo_annotations[self.selected_yolo_index]
        if annotation["type"] != "obb":
            return False

        points = np.asarray(annotation["points"], dtype=float)
        center = points.mean(axis=0)
        radians = np.deg2rad(float(degrees))
        rotation = np.array(
            [[np.cos(radians), -np.sin(radians)],
             [np.sin(radians),  np.cos(radians)]],
            dtype=float,
        )
        rotated = (points - center) @ rotation.T + center

        # Keep YOLO coordinates valid.  Reject a rotation that would place a
        # corner outside the source image rather than silently distorting it.
        h, w = self.mask.shape
        if (
            np.any(rotated[:, 0] < 0) or np.any(rotated[:, 0] > w - 1)
            or np.any(rotated[:, 1] < 0) or np.any(rotated[:, 1] > h - 1)
        ):
            return False

        annotation["points"] = rotated
        self.yoloAnnotationsChanged.emit()
        self.viewport().update()
        return True

    def mousePressEvent(self, event):
        if event.button() == Qt.MiddleButton or (
            event.button() == Qt.LeftButton and self.mode == "pan"
        ):
            self._panning = True
            self._last_pan_pos = event.position().toPoint()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return

        if event.button() == Qt.LeftButton and self.mode in {"bbox", "obb"}:
            point = self._image_point(event.position().toPoint())
            if point is not None:
                if self._box_start is None:
                    # First click: select an existing annotation when hit; otherwise
                    # lock the first corner and begin a live rectangle/OBB preview.
                    if self._select_yolo_at(point) < 0:
                        self._box_start = point
                        self._box_current = point
                        self.viewport().update()
                else:
                    # Second click: commit the live preview.  OBB starts as an
                    # axis-aligned quadrilateral and can then be rotated with [ ].
                    self._box_current = point
                    self._create_yolo_box(self._box_start, point, self.mode)
                    self._box_start = None
                    self._box_current = None
                    self.viewport().update()
                event.accept()
                return

        if event.button() == Qt.RightButton and self.mode in {"bbox", "obb"}:
            # Right-click cancels an unfinished YOLO preview without touching
            # already committed annotations.
            if self._box_start is not None:
                self._box_start = None
                self._box_current = None
                self.viewport().update()
                event.accept()
                return

        if (
            event.button() == Qt.LeftButton
            and event.modifiers() & Qt.ControlModifier
            and self.mode in {"paint", "erase"}
        ):
            point = self._image_point(event.position().toPoint())
            if point is not None:
                filled_pixels = self._flood_fill_zero_region(point)
                if filled_pixels > 0:
                    self.fillCompleted.emit(filled_pixels)
                    self.historyStepRequested.emit("区域填充")
                event.accept()
                return

        if event.button() == Qt.LeftButton and self.mode == "measure":
            point = self._image_point(event.position().toPoint())
            if point is not None:
                if self.measurement_start is None:
                    # First click: lock the start point and begin a live preview.
                    self.measurement_start = point
                    self.measurement_current = point
                    self.pending_measurement_index = None
                    self.viewport().update()
                else:
                    # Second click: commit the preview. If this measurement came
                    # from a right-click endpoint undo, restore it at the same ID.
                    start = self.measurement_start
                    end = point
                    insert_index = self.pending_measurement_index
                    self.measurement_start = None
                    self.measurement_current = None
                    self.pending_measurement_index = None

                    record = {
                        "x1": start.x(), "y1": start.y(),
                        "x2": end.x(), "y2": end.y(),
                    }
                    if insert_index is not None:
                        insert_index = max(0, min(int(insert_index), len(self.measurement_records)))
                        self.measurement_records.insert(insert_index, record)
                        index = insert_index
                    else:
                        self.measurement_records.append(record)
                        index = len(self.measurement_records) - 1
                    # Rebuild so IDs and selection highlighting remain correct.
                    self.set_measurements(self.measurement_records, selected_index=index)
                    length_pixels = math.hypot(end.x() - start.x(), end.y() - start.y())
                    self.measurementCreated.emit(length_pixels)
                    self.measurementsChanged.emit()
                    self.viewport().update()
                event.accept()
                return

        if event.button() == Qt.RightButton and self.mode == "measure":
            # Step-wise undo for measurements.
            # 1) If a start/live preview exists, cancel that pending start point.
            # 2) Otherwise undo the endpoint of the currently selected measurement.
            # 3) If no measurement is selected, fall back to the last measurement.
            if self.measurement_start is not None:
                self.measurement_start = None
                self.measurement_current = None
                self.pending_measurement_index = None
                self.measurementUndoPerformed.emit("cancel_pending")
                self.viewport().update()
                event.accept()
                return

            if self.measurement_records:
                target_index = self.selected_measurement_index
                if not (0 <= target_index < len(self.measurement_records)):
                    target_index = len(self.measurement_records) - 1

                target_record = dict(self.measurement_records[target_index])
                remaining = [dict(item) for item in self.measurement_records]
                remaining.pop(target_index)
                self.set_measurements(remaining, selected_index=-1)
                self.pending_measurement_index = target_index
                self.measurement_start = QPoint(
                    int(target_record["x1"]), int(target_record["y1"])
                )
                cursor_point = self._image_point(event.position().toPoint())
                self.measurement_current = cursor_point or QPoint(
                    int(target_record["x2"]), int(target_record["y2"])
                )
                self.measurementsChanged.emit()
                self.measurementUndoPerformed.emit("undo_endpoint")
                self.viewport().update()
                event.accept()
                return

            self.measurementUndoPerformed.emit("nothing")
            event.accept()
            return

        if event.button() == Qt.RightButton and self.mode in {"paint", "erase"}:
            point = self._image_point(event.position().toPoint())
            if point is not None and self.mask is not None:
                self.labelPicked.emit(int(self.mask[point.y(), point.x()]))
                event.accept()
                return

        if event.button() == Qt.LeftButton and self.mode in {"paint", "erase"}:
            point = self._image_point(event.position().toPoint())
            if point is not None:
                self._drawing = True
                self._last_image_point = point
                self._draw_segment(point, point)
                event.accept()
                return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        viewport_pos = event.position().toPoint()
        point = self._image_point(viewport_pos)

        if point is not None and self.mask is not None:
            self.cursorPositionChanged.emit(
                point.x(), point.y(), int(self.mask[point.y(), point.x()])
            )

        if self._panning and self._last_pan_pos is not None:
            delta = viewport_pos - self._last_pan_pos
            self._last_pan_pos = viewport_pos
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - delta.x()
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - delta.y()
            )
            event.accept()
            return

        if self.mode == "measure" and self.measurement_start is not None:
            if point is not None:
                self.measurement_current = point
                self.viewport().update()
            event.accept()
            return

        if self.mode in {"bbox", "obb"} and self._box_start is not None:
            if point is not None:
                self._box_current = point
                self.viewport().update()
            event.accept()
            return

        if self._drawing and point is not None and self._last_image_point is not None:
            self._draw_segment(self._last_image_point, point)
            self._last_image_point = point
            event.accept()
            return

        self.viewport().update()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._drawing and event.button() == Qt.LeftButton:
            self._drawing = False
            self._last_image_point = None
            self.maskChanged.emit()
            self.historyStepRequested.emit(
                "画笔" if self.mode == "paint" else "橡皮擦"
            )
            event.accept()
            return

        if self._panning and event.button() in {Qt.LeftButton, Qt.MiddleButton}:
            self._panning = False
            self._last_pan_pos = None
            self.setCursor(Qt.OpenHandCursor if self.mode == "pan" else Qt.CrossCursor)
            event.accept()
            return

        super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        if not self.has_image():
            super().wheelEvent(event)
            return

        direction = 1 if event.angleDelta().y() > 0 else -1
        modifiers = event.modifiers()

        # Ctrl + wheel is the primary zoom shortcut in every mode, including
        # measurement.  Alt + wheel remains as a backward-compatible alias.
        if modifiers & Qt.ControlModifier or modifiers & Qt.AltModifier:
            self.zoom_by(1.2 if direction > 0 else 1 / 1.2)
            event.accept()
            return

        # A plain wheel only changes brush size when that behavior is relevant.
        if self.mode in {"paint", "erase"}:
            self.brushSizeRequested.emit(direction)
            event.accept()
            return

        # In measurement / YOLO / pan modes, preserve normal view scrolling.
        super().wheelEvent(event)

    def drawForeground(self, painter: QPainter, rect: QRectF) -> None:
        super().drawForeground(painter, rect)
        if not self.has_image():
            return

        scale = max(self.transform().m11(), 0.01)
        painter.save()
        painter.setBrush(Qt.NoBrush)

        # YOLO rectangles / OBBs are always visible, independent of mask overlay.
        for index, annotation in enumerate(self.yolo_annotations):
            points = np.asarray(annotation["points"], dtype=float)
            selected = index == self.selected_yolo_index
            color = QColor(255, 230, 0) if selected else label_color(int(annotation["class_id"]) + 1)
            pen = QPen(color)
            pen.setWidthF((3.0 if selected else 2.0) / scale)
            painter.setPen(pen)
            for i in range(4):
                p1 = points[i]
                p2 = points[(i + 1) % 4]
                painter.drawLine(QPointF(float(p1[0]), float(p1[1])),
                                 QPointF(float(p2[0]), float(p2[1])))
            label = f"{annotation['type']}  cls:{int(annotation['class_id'])}"
            painter.drawText(
                QPointF(float(points[0, 0]) + 2.0 / scale,
                        float(points[0, 1]) - 4.0 / scale),
                label,
            )

        # Live YOLO preview: first click fixes one corner, mouse movement changes
        # the opposite corner, and the second click commits the annotation.
        if self.mode in {"bbox", "obb"} and self._box_start is not None and self._box_current is not None:
            x1, x2 = sorted((self._box_start.x(), self._box_current.x()))
            y1, y2 = sorted((self._box_start.y(), self._box_current.y()))
            preview_color = QColor(255, 255, 255)
            preview_pen = QPen(preview_color)
            preview_pen.setStyle(Qt.DashLine)
            preview_pen.setWidthF(2.0 / scale)
            painter.setPen(preview_pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(QRectF(float(x1), float(y1), float(x2 - x1), float(y2 - y1)))
            # Fixed-size corner markers make the two-click workflow obvious.
            marker = 4.0 / scale
            painter.setBrush(preview_color)
            for px, py in ((x1, y1), (x2, y1), (x2, y2), (x1, y2)):
                painter.drawEllipse(QPointF(float(px), float(py)), marker, marker)
            painter.setBrush(Qt.NoBrush)
            preview_name = "OBB" if self.mode == "obb" else "RECT"
            painter.drawText(
                QPointF(float(x1) + 6.0 / scale, float(y1) - 6.0 / scale),
                f"{preview_name}  cls:{self.current_yolo_class}",
            )

        # Measurement preview: after the first click, keep the start point visible
        # and draw a live line/length to the current mouse position.
        if self.mode == "measure" and self.measurement_start is not None:
            start = self.measurement_start
            current = self.measurement_current or start

            measure_pen = QPen(QColor(0, 255, 255))
            measure_pen.setWidthF(2.0 / scale)
            measure_pen.setStyle(Qt.DashLine)
            painter.setPen(measure_pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawLine(
                QPointF(float(start.x()), float(start.y())),
                QPointF(float(current.x()), float(current.y())),
            )

            # Fixed-size start marker, independent of zoom factor.
            marker_radius = 5.0 / scale
            painter.setBrush(QColor(0, 255, 255))
            painter.drawEllipse(
                QPointF(float(start.x()), float(start.y())),
                marker_radius, marker_radius,
            )

            length_pixels = math.hypot(
                current.x() - start.x(), current.y() - start.y()
            )
            live_text = self._measurement_text(length_pixels)
            text_offset = 8.0 / scale
            painter.setBrush(Qt.NoBrush)
            painter.drawText(
                QPointF(
                    (start.x() + current.x()) / 2.0 + text_offset,
                    (start.y() + current.y()) / 2.0 - text_offset,
                ),
                live_text,
            )

        painter.restore()

        # Brush cursor is only relevant in paint/erase modes.
        if self.mode not in {"paint", "erase"}:
            return
        mouse_pos = self.mapFromGlobal(self.cursor().pos())
        if not self.viewport().rect().contains(mouse_pos):
            return
        item_pos = self.pixmap_item.mapFromScene(self.mapToScene(mouse_pos))
        h, w = self.mask.shape
        if not (0 <= item_pos.x() < w and 0 <= item_pos.y() < h):
            return
        color = QColor(255, 230, 0) if self.mode == "erase" else label_color(self.current_label)
        pen = QPen(color)
        pen.setWidthF(max(0.5, 2.0 / scale))
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        radius = self.brush_size / 2.0
        painter.drawEllipse(item_pos, radius, radius)



LANGUAGES = ("en", "ja", "zh")
LANGUAGE_NAMES = {"en": "English", "ja": "日本語", "zh": "中文"}

TR = {
    "en": {
        "app_title": "Segmentation / YOLO / Measurement Annotator",
        "open_image": "Open Image", "open_folder": "Image Folder", "mask_folder": "Mask Folder",
        "yolo_folder": "YOLO Folder", "previous": "Previous", "next": "Next", "delete_sample": "Delete Sample",
        "fit": "Fit", "actual": "1:1", "zoom_in": "Zoom In", "zoom_out": "Zoom Out", "language": "Language",
        "tab_mask": "Mask", "tab_yolo": "YOLO", "tab_measure": "Measurement", "tab_history": "History",
        "mask_tools": "Mask tools", "paint": "Paint", "erase": "Erase", "pan": "Pan",
        "mask_label": "Label", "brush": "Brush", "opacity": "Opacity", "open_mask": "Open Mask",
        "save_mask": "Save Mask", "save_blend": "Save Blend", "area": "Area Statistics",
        "toggle_mask": "Show / Hide Mask", "clear_mask": "Clear Mask",
        "yolo_tools": "YOLO annotation", "rectangle": "Rectangle", "obb": "OBB", "class_id": "Class ID",
        "delete_annotation": "Delete Selected", "rotate_left": "Rotate −1°", "rotate_right": "Rotate +1°",
        "rotate_left_fast": "Rotate −5°", "rotate_right_fast": "Rotate +5°",
        "open_yolo": "Open YOLO", "save_yolo": "Save YOLO",
        "yolo_help": "R: rectangle   O: OBB   First click: set corner   Move: live preview   Second click: finish   Right-click: cancel preview   [ / ]: rotate OBB 1°   Shift+[ / ]: 5°   Backspace: delete",
        "measure_tools": "Distance measurement", "measure": "Measure", "resolution": "Resolution",
        "author": "Author", "author_placeholder": "Enter author name",
        "measurement_list": "Stoma measurements", "delete_measure": "Delete Selected", "clear_measure": "Clear All",
        "save_csv": "Export CSV", "save_xml": "Save XML", "load_xml": "Load XML",
        "export_images": "Export Annotated Images",
        "export_detail_title": "Measurement details",
        "export_detail_author": "Author",
        "export_detail_resolution": "Resolution",
        "measure_help": "Measurement: left-click once to set the start point, move for a live preview, and left-click again to finish. Select a measurement in the list, then right-click on the image to undo that measurement endpoint; if none is selected, the last measurement is used. Click a new endpoint to keep the same ID. Right-click again cancels the pending start point. Ctrl+mouse wheel: zoom.",
        "measurement_undo_pending": "Start point cancelled.", "measurement_undo_endpoint": "Endpoint undone. Choose a new endpoint, or right-click again to cancel the start point.", "measurement_undo_none": "No measurement to undo.",
        "history": "Mask history", "no_image": "No image opened", "position": "x: -, y: -, value: -", "zoom": "Zoom: 100%",
        "unsaved": "Unsaved", "ready": "Ready", "error": "Error", "info": "Information", "warning": "Warning",
        "select_image_first": "Please open an image first.", "invalid_image": "Unable to read image:\n{path}",
        "no_supported_images": "No supported images were found in this folder.",
        "mask_not_found": "not found", "mask": "Mask", "yolo": "YOLO", "measurements": "Measurements",
        "resolution_title": "Set Image Resolution", "resolution_label": "Resolution", "clear_value": "Clear",
        "resolution_set": "Resolution: {value:g} μm/pixel", "resolution_cleared": "Image resolution was cleared.",
        "confirm_clear_mask": "Clear the entire mask?", "confirm_clear_measure": "Clear all measurements for the current image?",
        "confirm_delete_sample": "These files will be permanently deleted and cannot be undone:\n\n{files}\n\nContinue?",
        "confirm_unsaved": "There are unsaved changes. Continue and discard them?",
        "mask_saved": "Mask saved: {path}", "blend_saved": "Blend saved: {path}", "yolo_saved": "YOLO labels saved: {path} ({count})",
        "auto_save_failed": "Automatic save failed:\n{path}", "measurement_added": "Measurement: {px:.2f} px{physical}",
        "measurement_item": "Stoma {id}: {px:.2f} px{physical}", "measurement_none": "No measurements to export.",
        "csv_saved": "Measurement CSV saved: {path}", "xml_saved": "Measurement XML saved: {path}",
        "xml_loaded": "Measurement XML loaded: {count} image(s), {measures} measurement(s).",
        "xml_invalid": "Failed to load measurement XML:\n{error}",
        "images_exported": "Exported {count} annotated image(s) to:\n{folder}",
        "rotation_need_obb": "Select an OBB annotation first.", "rotation_outside": "Rotation was blocked because a corner would leave the image.",
        "selected_yolo": "Selected {type} #{index}, class {class_id}", "no_selected_yolo": "No YOLO annotation selected",
        "area_title": "Mask Area Statistics", "area_summary_none": "No mask loaded",
        "area_total": "Image area: {total:,} px    Foreground: {fg:,} px    Labels: {labels}",
        "area_headers": ["Color", "Label", "Pixel area", "Physical area", "% image", "% foreground"],
        "not_set": "Not set", "history_initial": "Initial state", "history_paint": "Paint", "history_erase": "Erase", "history_fill": "Flood fill", "history_clear": "Clear mask",
        "deleted": "Deleted current sample: {name}", "delete_failed": "Some files could not be deleted:\n\n{files}",
        "mask_loaded": "Mask loaded", "yolo_ignored": "Ignored {count} invalid YOLO line(s).",
        "folder_status": "Image folder: {path}", "mask_folder_status": "Mask folder: {path}", "yolo_folder_status": "YOLO folder: {path}",
        "clear_resolution": "Clear resolution", "cancel": "Cancel", "ok": "OK",
    },
    "ja": {
        "app_title": "セグメンテーション / YOLO / 計測アノテータ",
        "open_image": "画像を開く", "open_folder": "画像フォルダ", "mask_folder": "マスクフォルダ",
        "yolo_folder": "YOLOフォルダ", "previous": "前へ", "next": "次へ", "delete_sample": "サンプル削除",
        "fit": "全体表示", "actual": "1:1", "zoom_in": "拡大", "zoom_out": "縮小", "language": "言語",
        "tab_mask": "マスク", "tab_yolo": "YOLO", "tab_measure": "計測", "tab_history": "履歴",
        "mask_tools": "マスクツール", "paint": "ブラシ", "erase": "消しゴム", "pan": "移動",
        "mask_label": "ラベル", "brush": "ブラシ幅", "opacity": "透明度", "open_mask": "マスクを開く",
        "save_mask": "マスク保存", "save_blend": "Blend保存", "area": "面積統計",
        "toggle_mask": "マスク表示/非表示", "clear_mask": "マスク消去",
        "yolo_tools": "YOLOアノテーション", "rectangle": "矩形", "obb": "OBB", "class_id": "クラスID",
        "delete_annotation": "選択を削除", "rotate_left": "−1°回転", "rotate_right": "+1°回転",
        "rotate_left_fast": "−5°回転", "rotate_right_fast": "+5°回転",
        "open_yolo": "YOLOを開く", "save_yolo": "YOLO保存",
        "yolo_help": "R: 矩形   O: OBB   1回目クリック: 始点   マウス移動: ライブプレビュー   2回目クリック: 確定   右クリック: プレビュー取消   [ / ]: OBBを1°回転   Shift+[ / ]: 5°   Backspace: 削除",
        "measure_tools": "距離計測", "measure": "計測", "resolution": "解像度",
        "author": "作成者", "author_placeholder": "作成者名を入力",
        "measurement_list": "気孔計測一覧", "delete_measure": "選択を削除", "clear_measure": "すべて消去",
        "save_csv": "CSV出力", "save_xml": "XML保存", "load_xml": "XML読込",
        "export_images": "計測画像を出力",
        "export_detail_title": "計測詳細",
        "export_detail_author": "作成者",
        "export_detail_resolution": "解像度",
        "measure_help": "計測: 1回目の左クリックで始点、マウス移動でライブプレビュー、2回目の左クリックで確定します。右側リストで計測を選択して画像上を右クリックすると、その計測の終点だけを戻せます。未選択なら最後の計測を対象にします。新しい終点をクリックすると同じID位置に戻ります。もう一度右クリックすると始点も取消します。Ctrl+マウスホイール: ズーム。",
        "measurement_undo_pending": "始点をキャンセルしました。", "measurement_undo_endpoint": "終点を戻しました。新しい終点を選ぶか、もう一度右クリックして始点もキャンセルできます。", "measurement_undo_none": "戻せる計測がありません。",
        "history": "マスク履歴", "no_image": "画像未選択", "position": "x: -, y: -, 値: -", "zoom": "ズーム: 100%",
        "unsaved": "未保存", "ready": "準備完了", "error": "エラー", "info": "情報", "warning": "警告",
        "select_image_first": "先に画像を開いてください。", "invalid_image": "画像を読み込めません:\n{path}",
        "no_supported_images": "このフォルダに対応画像がありません。",
        "mask_not_found": "なし", "mask": "マスク", "yolo": "YOLO", "measurements": "計測",
        "resolution_title": "画像解像度の設定", "resolution_label": "解像度", "clear_value": "クリア",
        "resolution_set": "解像度: {value:g} μm/pixel", "resolution_cleared": "画像解像度をクリアしました。",
        "confirm_clear_mask": "マスク全体を消去しますか？", "confirm_clear_measure": "現在画像の計測をすべて消去しますか？",
        "confirm_delete_sample": "次のファイルを完全に削除します。元に戻せません:\n\n{files}\n\n続行しますか？",
        "confirm_unsaved": "未保存の変更があります。破棄して続行しますか？",
        "mask_saved": "マスクを保存しました: {path}", "blend_saved": "Blendを保存しました: {path}", "yolo_saved": "YOLOラベルを保存しました: {path} ({count})",
        "auto_save_failed": "自動保存に失敗しました:\n{path}", "measurement_added": "計測: {px:.2f} px{physical}",
        "measurement_item": "気孔 {id}: {px:.2f} px{physical}", "measurement_none": "出力する計測データがありません。",
        "csv_saved": "計測CSVを保存しました: {path}", "xml_saved": "計測XMLを保存しました: {path}",
        "xml_loaded": "計測XMLを読み込みました: {count}画像、{measures}件。",
        "xml_invalid": "計測XMLを読み込めません:\n{error}",
        "images_exported": "計測画像 {count} 枚を出力しました:\n{folder}",
        "rotation_need_obb": "先にOBBを選択してください。", "rotation_outside": "回転後の頂点が画像外になるため回転を中止しました。",
        "selected_yolo": "選択: {type} #{index}, class {class_id}", "no_selected_yolo": "YOLO選択なし",
        "area_title": "マスク面積統計", "area_summary_none": "マスク未読込",
        "area_total": "画像面積: {total:,} px    前景: {fg:,} px    ラベル数: {labels}",
        "area_headers": ["色", "ラベル", "画素面積", "実面積", "画像比率", "前景比率"],
        "not_set": "未設定", "history_initial": "初期状態", "history_paint": "ブラシ", "history_erase": "消しゴム", "history_fill": "領域塗り", "history_clear": "マスク消去",
        "deleted": "現在のサンプルを削除しました: {name}", "delete_failed": "削除できないファイルがあります:\n\n{files}",
        "mask_loaded": "マスク読込", "yolo_ignored": "無効なYOLO行を {count} 件無視しました。",
        "folder_status": "画像フォルダ: {path}", "mask_folder_status": "マスクフォルダ: {path}", "yolo_folder_status": "YOLOフォルダ: {path}",
        "clear_resolution": "解像度をクリア", "cancel": "キャンセル", "ok": "OK",
    },
    "zh": {
        "app_title": "分割 / YOLO / 测量标注工具",
        "open_image": "打开图像", "open_folder": "图像文件夹", "mask_folder": "掩码文件夹",
        "yolo_folder": "YOLO文件夹", "previous": "上一张", "next": "下一张", "delete_sample": "删除样本",
        "fit": "适应窗口", "actual": "1:1", "zoom_in": "放大", "zoom_out": "缩小", "language": "语言",
        "tab_mask": "Mask", "tab_yolo": "YOLO", "tab_measure": "测量", "tab_history": "历史",
        "mask_tools": "掩码工具", "paint": "画笔", "erase": "橡皮擦", "pan": "拖动",
        "mask_label": "标签值", "brush": "笔刷", "opacity": "透明度", "open_mask": "打开掩码",
        "save_mask": "保存掩码", "save_blend": "保存 Blend", "area": "面积统计",
        "toggle_mask": "显示/隐藏掩码", "clear_mask": "清空掩码",
        "yolo_tools": "YOLO 标注", "rectangle": "矩形框", "obb": "OBB", "class_id": "类别 ID",
        "delete_annotation": "删除选中标注", "rotate_left": "旋转 −1°", "rotate_right": "旋转 +1°",
        "rotate_left_fast": "旋转 −5°", "rotate_right_fast": "旋转 +5°",
        "open_yolo": "打开 YOLO", "save_yolo": "保存 YOLO",
        "yolo_help": "R：矩形框   O：OBB   第一次点击：确定角点   移动鼠标：实时预览   第二次点击：完成   右键：取消预览   [ / ]：OBB旋转1°   Shift+[ / ]：5°   Backspace：删除",
        "measure_tools": "距离测量", "measure": "测量", "resolution": "分辨率",
        "author": "作者", "author_placeholder": "请输入作者姓名",
        "measurement_list": "气孔测量列表", "delete_measure": "删除选中测量", "clear_measure": "清空全部",
        "save_csv": "导出 CSV", "save_xml": "保存 XML", "load_xml": "加载 XML",
        "export_images": "导出测量结果图",
        "export_detail_title": "测量详情",
        "export_detail_author": "作者",
        "export_detail_resolution": "分辨率",
        "measure_help": "测量：左键第一次点击确定起点，移动鼠标实时预览，左键第二次点击完成。先在右侧列表选中一条测量，再在图像上右键，可回退该测量的终点；未选中时默认回退最后一条。重新点击终点后会保持原来的ID位置；再右键一次则取消起点。Ctrl+鼠标滚轮：缩放。",
        "measurement_undo_pending": "已取消测量起点。", "measurement_undo_endpoint": "已回退终点，请重新选择终点；再次右键可继续取消起点。", "measurement_undo_none": "当前没有可回退的测量。",
        "history": "Mask 历史", "no_image": "未打开图像", "position": "x: -, y: -, 值: -", "zoom": "缩放: 100%",
        "unsaved": "未保存", "ready": "就绪", "error": "错误", "info": "提示", "warning": "警告",
        "select_image_first": "请先打开彩色图像。", "invalid_image": "无法读取图像：\n{path}",
        "no_supported_images": "图像文件夹中没有支持的图像。",
        "mask_not_found": "未找到", "mask": "掩码", "yolo": "YOLO", "measurements": "测量",
        "resolution_title": "设置图像分辨率", "resolution_label": "分辨率", "clear_value": "清除",
        "resolution_set": "图像分辨率：{value:g} μm/pixel", "resolution_cleared": "已清除图像分辨率。",
        "confirm_clear_mask": "确定要清空整个掩码吗？", "confirm_clear_measure": "确定清空当前图片的所有测量吗？",
        "confirm_delete_sample": "以下文件将被永久删除，无法撤销：\n\n{files}\n\n确定继续吗？",
        "confirm_unsaved": "当前有未保存的修改，是否继续并放弃修改？",
        "mask_saved": "掩码已保存：{path}", "blend_saved": "Blend 图像已保存：{path}", "yolo_saved": "YOLO 标签已保存：{path}（{count} 个）",
        "auto_save_failed": "自动保存失败：\n{path}", "measurement_added": "测量长度：{px:.2f} px{physical}",
        "measurement_item": "Stoma {id}: {px:.2f} px{physical}", "measurement_none": "没有可以导出的测量数据。",
        "csv_saved": "测量 CSV 已保存：{path}", "xml_saved": "测量 XML 已保存：{path}",
        "xml_loaded": "测量 XML 已加载：{count} 张图，{measures} 条测量。",
        "xml_invalid": "加载测量 XML 失败：\n{error}",
        "images_exported": "已导出 {count} 张测量结果图到：\n{folder}",
        "rotation_need_obb": "请先选中一个 OBB 标注。", "rotation_outside": "旋转后的角点将超出图像，已取消本次旋转。",
        "selected_yolo": "已选择 {type} #{index}，类别 {class_id}", "no_selected_yolo": "未选择 YOLO 标注",
        "area_title": "掩码面积统计", "area_summary_none": "尚未加载掩码",
        "area_total": "图像总面积：{total:,} px    前景总面积：{fg:,} px    标签种类：{labels}",
        "area_headers": ["颜色", "标签值", "像素面积", "物理面积", "占整图比例", "占前景比例"],
        "not_set": "未设置", "history_initial": "初始状态", "history_paint": "画笔", "history_erase": "橡皮擦", "history_fill": "区域填充", "history_clear": "清空掩码",
        "deleted": "已删除当前样本：{name}", "delete_failed": "以下文件未能删除：\n\n{files}",
        "mask_loaded": "载入掩码", "yolo_ignored": "已忽略 {count} 行无效 YOLO 标签。",
        "folder_status": "图像文件夹：{path}", "mask_folder_status": "掩码文件夹：{path}", "yolo_folder_status": "YOLO 文件夹：{path}",
        "clear_resolution": "清除分辨率", "cancel": "取消", "ok": "确定",
    },
}


def t(lang: str, key: str, **kwargs):
    value = TR.get(lang, TR["en"]).get(key, TR["en"].get(key, key))
    if isinstance(value, str) and kwargs:
        return value.format(**kwargs)
    return value


class AreaStatisticsWindow(QMainWindow):
    def __init__(self, language: str = "zh", parent=None):
        super().__init__(parent)
        self.language = language
        self.resolution_um_per_pixel: Optional[float] = None
        self.summary_label = QLabel()
        self.summary_label.setContentsMargins(8, 8, 8, 8)
        self.table = QTableWidget(0, 6)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.addWidget(self.summary_label)
        layout.addWidget(self.table)
        self.setCentralWidget(widget)
        self.resize(620, 430)
        self.retranslate(language)

    def retranslate(self, language: str) -> None:
        self.language = language
        self.setWindowTitle(t(language, "area_title"))
        self.table.setHorizontalHeaderLabels(t(language, "area_headers"))
        if self.table.rowCount() == 0:
            self.summary_label.setText(t(language, "area_summary_none"))

    def set_resolution(self, resolution_um_per_pixel: Optional[float]) -> None:
        self.resolution_um_per_pixel = resolution_um_per_pixel

    def update_from_mask(self, mask: Optional[np.ndarray]) -> None:
        self.table.setRowCount(0)
        if mask is None or mask.size == 0:
            self.summary_label.setText(t(self.language, "area_summary_none"))
            return
        total_pixels = int(mask.size)
        labels, counts = np.unique(mask, return_counts=True)
        foreground_pixels = int(np.count_nonzero(mask))
        nonzero = [(int(label), int(count)) for label, count in zip(labels, counts) if int(label) != 0]
        nonzero.sort(key=lambda item: item[0])
        self.summary_label.setText(t(self.language, "area_total", total=total_pixels, fg=foreground_pixels, labels=len(nonzero)))
        self.table.setRowCount(len(nonzero))
        for row, (label, count) in enumerate(nonzero):
            color = label_color(label)
            swatch = QTableWidgetItem("")
            swatch.setBackground(color)
            label_item = QTableWidgetItem(str(label))
            area_item = QTableWidgetItem(f"{count:,}")
            if self.resolution_um_per_pixel is not None:
                physical_area = count * (self.resolution_um_per_pixel ** 2)
                physical_item = QTableWidgetItem(f"{physical_area:,.4f} μm²")
            else:
                physical_item = QTableWidgetItem(t(self.language, "not_set"))
            whole_ratio = QTableWidgetItem(f"{count / total_pixels * 100:.4f}%")
            foreground_ratio = QTableWidgetItem(f"{count / foreground_pixels * 100:.4f}%" if foreground_pixels else "0.0000%")
            for column, item in enumerate([swatch, label_item, area_item, physical_item, whole_ratio, foreground_ratio]):
                item.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(row, column, item)


class ResolutionDialog(QDialog):
    def __init__(self, current_value: Optional[float], language: str, parent=None):
        super().__init__(parent)
        self.language = language
        self.cleared = False
        self.value_spin = QDoubleSpinBox()
        self.value_spin.setRange(0.000001, 1000000.0)
        self.value_spin.setDecimals(6)
        self.value_spin.setSingleStep(0.01)
        self.value_spin.setSuffix(" μm/pixel")
        self.value_spin.setValue(current_value if current_value is not None else 0.15)
        form = QFormLayout(self)
        form.addRow(t(language, "resolution_label") + ":", self.value_spin)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel | QDialogButtonBox.Reset)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        buttons.button(QDialogButtonBox.Reset).clicked.connect(self.clear_value)
        buttons.button(QDialogButtonBox.Reset).setText(t(language, "clear_resolution"))
        form.addRow(buttons)
        self.setWindowTitle(t(language, "resolution_title"))

    def clear_value(self):
        self.cleared = True
        self.accept()

    def resolution(self) -> Optional[float]:
        return None if self.cleared else float(self.value_spin.value())


class MainWindow(QMainWindow):
    MAX_HISTORY = 30

    def __init__(self):
        super().__init__()
        self.language = "zh"
        self.resize(1500, 920)

        self.current_image_path: Optional[Path] = None
        self.image_files: list[Path] = []
        self.current_index = -1
        self.image_folder: Optional[Path] = None
        self.mask_folder: Optional[Path] = None
        self.yolo_label_folder: Optional[Path] = None
        self.mask_dirty = False
        self.yolo_dirty = False
        self.resolution_um_per_pixel: Optional[float] = None
        self.mask_visible = True
        self.history_masks: list[np.ndarray] = []
        self.history_names: list[str] = []
        self.history_index = -1
        self.restoring_history = False
        self.measurements_by_image: dict[str, list[dict]] = {}
        self.measurement_author = ""

        self.view = AnnotationView(self)
        self.area_window = AreaStatisticsWindow(self.language, self)

        self._build_actions()
        self._build_toolbar()
        self._build_central_ui()
        self._build_statusbar()
        self._connect_signals()
        self.retranslate_ui()
        self._update_navigation_state()
        self._update_label_swatch()

    def tr(self, key: str, **kwargs):
        return t(self.language, key, **kwargs)

    def _build_actions(self):
        self.open_image_action = QAction(self)
        self.open_image_action.setShortcut(QKeySequence.Open)
        self.open_folder_action = QAction(self)
        self.open_folder_action.setShortcut("Ctrl+Alt+O")
        self.open_mask_folder_action = QAction(self)
        self.open_mask_folder_action.setShortcut("Ctrl+Alt+M")
        self.open_yolo_folder_action = QAction(self)
        self.open_yolo_folder_action.setShortcut("Ctrl+Alt+Y")
        self.previous_action = QAction(self)
        self.previous_action.setShortcut("A")
        self.next_action = QAction(self)
        self.next_action.setShortcut("D")
        self.delete_sample_action = QAction(self)
        self.delete_sample_action.setShortcut("Ctrl+Delete")
        self.open_mask_action = QAction(self)
        self.open_mask_action.setShortcut("Ctrl+Shift+O")
        self.save_mask_action = QAction(self)
        self.save_mask_action.setShortcut(QKeySequence.Save)
        self.save_blend_action = QAction(self)
        self.save_blend_action.setShortcut("Ctrl+Shift+S")
        self.open_yolo_action = QAction(self)
        self.open_yolo_action.setShortcut("Ctrl+Shift+Y")
        self.save_yolo_action = QAction(self)
        self.save_yolo_action.setShortcut("Ctrl+Alt+S")
        self.area_action = QAction(self)
        self.area_action.setShortcut("Ctrl+I")
        self.toggle_mask_action = QAction(self)
        self.toggle_mask_action.setShortcut("M")
        self.resolution_action = QAction(self)
        self.resolution_action.setShortcut("Ctrl+R")
        self.clear_measurements_action = QAction(self)
        self.clear_measurements_action.setShortcut("Ctrl+L")
        self.clear_action = QAction(self)
        self.clear_action.setShortcut("Delete")

        self.paint_action = QAction(self); self.paint_action.setCheckable(True); self.paint_action.setShortcut("B")
        self.erase_action = QAction(self); self.erase_action.setCheckable(True); self.erase_action.setShortcut("E")
        self.pan_action = QAction(self); self.pan_action.setCheckable(True); self.pan_action.setShortcut("H")
        self.measure_action = QAction(self); self.measure_action.setCheckable(True); self.measure_action.setShortcut("L")
        self.bbox_action = QAction(self); self.bbox_action.setCheckable(True); self.bbox_action.setShortcut("R")
        self.obb_action = QAction(self); self.obb_action.setCheckable(True); self.obb_action.setShortcut("O")
        group = QActionGroup(self); group.setExclusive(True)
        for action in (self.paint_action, self.erase_action, self.pan_action, self.measure_action, self.bbox_action, self.obb_action):
            group.addAction(action)
        self.paint_action.setChecked(True)

        self.delete_yolo_action = QAction(self); self.delete_yolo_action.setShortcut("Backspace")
        self.rotate_obb_left_action = QAction(self); self.rotate_obb_left_action.setShortcut("[")
        self.rotate_obb_right_action = QAction(self); self.rotate_obb_right_action.setShortcut("]")
        self.rotate_obb_left_fast_action = QAction(self); self.rotate_obb_left_fast_action.setShortcut("Shift+[")
        self.rotate_obb_right_fast_action = QAction(self); self.rotate_obb_right_fast_action.setShortcut("Shift+]")
        self.zoom_in_action = QAction(self); self.zoom_in_action.setShortcut(QKeySequence.ZoomIn)
        self.zoom_out_action = QAction(self); self.zoom_out_action.setShortcut(QKeySequence.ZoomOut)
        self.fit_action = QAction(self); self.fit_action.setShortcut("F")
        self.actual_action = QAction(self); self.actual_action.setShortcut("1")

    def _build_toolbar(self):
        self.toolbar = QToolBar(self)
        self.toolbar.setMovable(False)
        self.addToolBar(self.toolbar)
        for action in (self.open_image_action, self.open_folder_action, self.previous_action, self.next_action):
            self.toolbar.addAction(action)
        self.toolbar.addSeparator()
        self.toolbar.addAction(self.zoom_out_action); self.toolbar.addAction(self.zoom_in_action)
        self.toolbar.addAction(self.fit_action); self.toolbar.addAction(self.actual_action)
        self.toolbar.addSeparator()
        self.toolbar.addAction(self.delete_sample_action)
        self.toolbar.addSeparator()
        self.language_label = QLabel()
        self.toolbar.addWidget(self.language_label)
        self.language_combo = QComboBox()
        for code in LANGUAGES:
            self.language_combo.addItem(LANGUAGE_NAMES[code], code)
        self.language_combo.setCurrentIndex(LANGUAGES.index(self.language))
        self.toolbar.addWidget(self.language_combo)

    def _group(self, title: str = "") -> tuple[QGroupBox, QVBoxLayout]:
        box = QGroupBox(title)
        layout = QVBoxLayout(box)
        layout.setContentsMargins(8, 12, 8, 8)
        layout.setSpacing(6)
        return box, layout

    def _button_for_action(self, action: QAction) -> QToolButton:
        """Create a compact button backed directly by a QAction.

        QToolButton supports setDefaultAction(); QPushButton does not. Using
        the QAction directly also keeps text, enabled state and checked state
        synchronized when the UI language or mode changes.
        """
        button = QToolButton()
        button.setDefaultAction(action)
        button.setToolButtonStyle(Qt.ToolButtonTextOnly)
        button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        return button

    def _build_central_ui(self):
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.view)
        self.side_tabs = QTabWidget()
        self.side_tabs.setMinimumWidth(330)
        self.side_tabs.setMaximumWidth(430)
        splitter.addWidget(self.side_tabs)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        self.setCentralWidget(splitter)

        # Mask tab
        self.mask_tab = QWidget(); mask_layout = QVBoxLayout(self.mask_tab)
        self.mask_group, g = self._group()
        row = QHBoxLayout(); row.addWidget(self._button_for_action(self.paint_action)); row.addWidget(self._button_for_action(self.erase_action)); row.addWidget(self._button_for_action(self.pan_action)); g.addLayout(row)
        grid = QGridLayout()
        self.mask_label_caption = QLabel(); grid.addWidget(self.mask_label_caption, 0, 0)
        self.label_spin = QSpinBox(); self.label_spin.setRange(1, 255); self.label_spin.setValue(1); grid.addWidget(self.label_spin, 0, 1)
        self.color_swatch = QLabel(); self.color_swatch.setFixedSize(28, 24); grid.addWidget(self.color_swatch, 0, 2)
        self.brush_caption = QLabel(); grid.addWidget(self.brush_caption, 1, 0)
        self.brush_spin = QSpinBox(); self.brush_spin.setRange(1, 200); self.brush_spin.setValue(self.view.brush_size); self.brush_spin.setSuffix(" px"); grid.addWidget(self.brush_spin, 1, 1, 1, 2)
        self.opacity_caption = QLabel(); grid.addWidget(self.opacity_caption, 2, 0)
        self.alpha_slider = QSlider(Qt.Horizontal); self.alpha_slider.setRange(0, 100); self.alpha_slider.setValue(int(self.view.overlay_alpha * 100)); grid.addWidget(self.alpha_slider, 2, 1, 1, 2)
        g.addLayout(grid)
        row = QHBoxLayout(); row.addWidget(self._button_for_action(self.open_mask_action)); row.addWidget(self._button_for_action(self.save_mask_action)); g.addLayout(row)
        row = QHBoxLayout(); row.addWidget(self._button_for_action(self.save_blend_action)); row.addWidget(self._button_for_action(self.area_action)); g.addLayout(row)
        row = QHBoxLayout(); row.addWidget(self._button_for_action(self.toggle_mask_action)); row.addWidget(self._button_for_action(self.clear_action)); g.addLayout(row)
        self.mask_folder_button = QPushButton(); self.mask_folder_button.clicked.connect(self.open_mask_folder); g.addWidget(self.mask_folder_button)
        mask_layout.addWidget(self.mask_group); mask_layout.addStretch(1)
        self.side_tabs.addTab(self.mask_tab, "")

        # YOLO tab
        self.yolo_tab = QWidget(); yolo_layout = QVBoxLayout(self.yolo_tab)
        self.yolo_group, g = self._group()
        row = QHBoxLayout(); row.addWidget(self._button_for_action(self.bbox_action)); row.addWidget(self._button_for_action(self.obb_action)); g.addLayout(row)
        grid = QGridLayout(); self.class_caption = QLabel(); grid.addWidget(self.class_caption, 0, 0)
        self.yolo_class_spin = QSpinBox(); self.yolo_class_spin.setRange(0, 999); grid.addWidget(self.yolo_class_spin, 0, 1); g.addLayout(grid)
        row = QHBoxLayout(); row.addWidget(self._button_for_action(self.rotate_obb_left_action)); row.addWidget(self._button_for_action(self.rotate_obb_right_action)); g.addLayout(row)
        row = QHBoxLayout(); row.addWidget(self._button_for_action(self.rotate_obb_left_fast_action)); row.addWidget(self._button_for_action(self.rotate_obb_right_fast_action)); g.addLayout(row)
        g.addWidget(self._button_for_action(self.delete_yolo_action))
        row = QHBoxLayout(); row.addWidget(self._button_for_action(self.open_yolo_action)); row.addWidget(self._button_for_action(self.save_yolo_action)); g.addLayout(row)
        self.yolo_folder_button = QPushButton(); self.yolo_folder_button.clicked.connect(self.open_yolo_folder); g.addWidget(self.yolo_folder_button)
        self.yolo_selection_label = QLabel(); self.yolo_selection_label.setWordWrap(True); g.addWidget(self.yolo_selection_label)
        self.yolo_help_label = QLabel(); self.yolo_help_label.setWordWrap(True); g.addWidget(self.yolo_help_label)
        yolo_layout.addWidget(self.yolo_group); yolo_layout.addStretch(1)
        self.side_tabs.addTab(self.yolo_tab, "")

        # Measurement tab
        self.measure_tab = QWidget(); ml = QVBoxLayout(self.measure_tab)
        self.measure_group, g = self._group()
        row = QHBoxLayout(); row.addWidget(self._button_for_action(self.measure_action)); row.addWidget(self._button_for_action(self.resolution_action)); g.addLayout(row)
        self.resolution_value_label = QLabel(); g.addWidget(self.resolution_value_label)
        author_row = QHBoxLayout()
        self.author_caption = QLabel(); author_row.addWidget(self.author_caption)
        self.author_edit = QLineEdit(); self.author_edit.textChanged.connect(self._author_changed); author_row.addWidget(self.author_edit, 1)
        g.addLayout(author_row)
        self.measure_help_label = QLabel(); self.measure_help_label.setWordWrap(True); g.addWidget(self.measure_help_label)
        ml.addWidget(self.measure_group)
        self.measure_list_group, lg = self._group()
        self.measurement_list = QListWidget(); lg.addWidget(self.measurement_list)
        row = QHBoxLayout(); self.delete_measure_button = QPushButton(); self.delete_measure_button.clicked.connect(self.delete_selected_measurement); row.addWidget(self.delete_measure_button)
        self.clear_measure_button = QPushButton(); self.clear_measure_button.clicked.connect(self.clear_current_measurements); row.addWidget(self.clear_measure_button); lg.addLayout(row)
        row = QHBoxLayout(); self.csv_button = QPushButton(); self.csv_button.clicked.connect(self.export_measurements_csv); row.addWidget(self.csv_button)
        self.xml_save_button = QPushButton(); self.xml_save_button.clicked.connect(self.save_measurements_xml); row.addWidget(self.xml_save_button); lg.addLayout(row)
        row = QHBoxLayout(); self.xml_load_button = QPushButton(); self.xml_load_button.clicked.connect(self.load_measurements_xml); row.addWidget(self.xml_load_button)
        self.measure_images_button = QPushButton(); self.measure_images_button.clicked.connect(self.export_measurement_images); row.addWidget(self.measure_images_button); lg.addLayout(row)
        ml.addWidget(self.measure_list_group, 1)
        self.side_tabs.addTab(self.measure_tab, "")

        # History tab
        self.history_tab = QWidget(); hl = QVBoxLayout(self.history_tab)
        self.history_group, hg = self._group()
        self.history_list = QListWidget(); hg.addWidget(self.history_list)
        hl.addWidget(self.history_group)
        self.side_tabs.addTab(self.history_tab, "")

    def _build_statusbar(self):
        status = QStatusBar(self); self.setStatusBar(status)
        self.file_label = QLabel(); self.position_label = QLabel(); self.zoom_label = QLabel(); self.dirty_label = QLabel()
        status.addWidget(self.file_label, 1); status.addPermanentWidget(self.position_label); status.addPermanentWidget(self.zoom_label); status.addPermanentWidget(self.dirty_label)

    def _connect_signals(self):
        self.open_image_action.triggered.connect(self.open_image)
        self.open_folder_action.triggered.connect(self.open_folder)
        self.previous_action.triggered.connect(lambda: self.navigate(-1)); self.next_action.triggered.connect(lambda: self.navigate(1))
        self.delete_sample_action.triggered.connect(self.delete_current_sample)
        self.open_mask_action.triggered.connect(self.open_mask); self.save_mask_action.triggered.connect(self.save_mask); self.save_blend_action.triggered.connect(self.save_blend)
        self.open_yolo_action.triggered.connect(self.open_yolo_labels); self.save_yolo_action.triggered.connect(self.save_yolo_labels)
        self.area_action.triggered.connect(self.show_area_statistics); self.toggle_mask_action.triggered.connect(self.toggle_mask_visibility); self.resolution_action.triggered.connect(self.set_resolution); self.clear_action.triggered.connect(self.clear_mask)
        self.paint_action.triggered.connect(lambda: self.view.set_mode("paint")); self.erase_action.triggered.connect(lambda: self.view.set_mode("erase")); self.pan_action.triggered.connect(lambda: self.view.set_mode("pan")); self.measure_action.triggered.connect(lambda: self.view.set_mode("measure")); self.bbox_action.triggered.connect(lambda: self.view.set_mode("bbox")); self.obb_action.triggered.connect(lambda: self.view.set_mode("obb"))
        self.delete_yolo_action.triggered.connect(self.delete_selected_yolo)
        self.rotate_obb_left_action.triggered.connect(lambda: self.rotate_selected_obb(-1.0)); self.rotate_obb_right_action.triggered.connect(lambda: self.rotate_selected_obb(1.0)); self.rotate_obb_left_fast_action.triggered.connect(lambda: self.rotate_selected_obb(-5.0)); self.rotate_obb_right_fast_action.triggered.connect(lambda: self.rotate_selected_obb(5.0))
        self.zoom_in_action.triggered.connect(lambda: self.view.zoom_by(1.25)); self.zoom_out_action.triggered.connect(lambda: self.view.zoom_by(0.8)); self.fit_action.triggered.connect(self.view.fit_image); self.actual_action.triggered.connect(self.view.actual_size)
        self.label_spin.valueChanged.connect(self.view.set_current_label); self.label_spin.valueChanged.connect(self._update_label_swatch)
        self.yolo_class_spin.valueChanged.connect(self.view.set_current_yolo_class)
        self.brush_spin.valueChanged.connect(self.view.set_brush_size); self.alpha_slider.valueChanged.connect(self.view.set_overlay_alpha)
        self.language_combo.currentIndexChanged.connect(self._language_changed)
        self.measurement_list.currentRowChanged.connect(self.view.set_selected_measurement)
        self.view.maskChanged.connect(self.mark_dirty); self.view.maskChanged.connect(lambda: self.area_window.update_from_mask(self.view.mask))
        self.view.cursorPositionChanged.connect(self._position_changed); self.view.zoomChanged.connect(self._zoom_changed)
        self.view.labelPicked.connect(self.pick_label_from_mask); self.view.brushSizeRequested.connect(self.adjust_brush_size); self.view.labelValueRequested.connect(self.adjust_label_value)
        self.view.measurementCreated.connect(self.on_measurement_created); self.view.measurementsChanged.connect(self.on_measurements_changed); self.view.measurementUndoPerformed.connect(self.on_measurement_undo); self.view.fillCompleted.connect(self.on_fill_completed); self.view.historyStepRequested.connect(self.record_history_step)
        self.view.yoloAnnotationsChanged.connect(self.mark_yolo_dirty); self.view.yoloSelectionChanged.connect(self.on_yolo_selection_changed)
        self.history_list.currentRowChanged.connect(self.restore_history_step)

    def _language_changed(self):
        code = self.language_combo.currentData()
        if code in LANGUAGES:
            self.language = code
            self.retranslate_ui()

    def retranslate_ui(self):
        self.setWindowTitle(self.tr("app_title") if self.current_image_path is None else f"{self.tr('app_title')} - {self.current_image_path.name}")
        action_texts = {
            self.open_image_action:"open_image", self.open_folder_action:"open_folder", self.previous_action:"previous", self.next_action:"next", self.delete_sample_action:"delete_sample",
            self.open_mask_action:"open_mask", self.save_mask_action:"save_mask", self.save_blend_action:"save_blend", self.open_yolo_action:"open_yolo", self.save_yolo_action:"save_yolo",
            self.area_action:"area", self.toggle_mask_action:"toggle_mask", self.resolution_action:"resolution", self.clear_action:"clear_mask",
            self.paint_action:"paint", self.erase_action:"erase", self.pan_action:"pan", self.measure_action:"measure", self.bbox_action:"rectangle", self.obb_action:"obb",
            self.delete_yolo_action:"delete_annotation", self.rotate_obb_left_action:"rotate_left", self.rotate_obb_right_action:"rotate_right", self.rotate_obb_left_fast_action:"rotate_left_fast", self.rotate_obb_right_fast_action:"rotate_right_fast",
            self.zoom_in_action:"zoom_in", self.zoom_out_action:"zoom_out", self.fit_action:"fit", self.actual_action:"actual",
        }
        for action, key in action_texts.items(): action.setText(self.tr(key))
        self.language_label.setText(self.tr("language") + ": ")
        self.side_tabs.setTabText(0, self.tr("tab_mask")); self.side_tabs.setTabText(1, self.tr("tab_yolo")); self.side_tabs.setTabText(2, self.tr("tab_measure")); self.side_tabs.setTabText(3, self.tr("tab_history"))
        self.mask_group.setTitle(self.tr("mask_tools")); self.yolo_group.setTitle(self.tr("yolo_tools")); self.measure_group.setTitle(self.tr("measure_tools")); self.measure_list_group.setTitle(self.tr("measurement_list")); self.history_group.setTitle(self.tr("history"))
        self.mask_label_caption.setText(self.tr("mask_label") + ":"); self.brush_caption.setText(self.tr("brush") + ":"); self.opacity_caption.setText(self.tr("opacity") + ":")
        self.class_caption.setText(self.tr("class_id") + ":"); self.mask_folder_button.setText(self.tr("mask_folder")); self.yolo_folder_button.setText(self.tr("yolo_folder"))
        self.yolo_help_label.setText(self.tr("yolo_help")); self.author_caption.setText(self.tr("author") + ":"); self.author_edit.setPlaceholderText(self.tr("author_placeholder")); self.measure_help_label.setText(self.tr("measure_help"))
        self.delete_measure_button.setText(self.tr("delete_measure")); self.clear_measure_button.setText(self.tr("clear_measure")); self.csv_button.setText(self.tr("save_csv")); self.xml_save_button.setText(self.tr("save_xml")); self.xml_load_button.setText(self.tr("load_xml")); self.measure_images_button.setText(self.tr("export_images"))
        self.area_window.retranslate(self.language)
        self._update_resolution_label(); self._update_measurement_list(); self._update_dirty_state(); self._refresh_file_label()
        if self.view.selected_yolo_index < 0: self.yolo_selection_label.setText(self.tr("no_selected_yolo"))

    def _position_changed(self, x: int, y: int, value: int):
        if self.language == "en": self.position_label.setText(f"x: {x}, y: {y}, value: {value}")
        elif self.language == "ja": self.position_label.setText(f"x: {x}, y: {y}, 値: {value}")
        else: self.position_label.setText(f"x: {x}, y: {y}, 值: {value}")

    def _zoom_changed(self, value: float):
        prefix = "Zoom" if self.language == "en" else ("ズーム" if self.language == "ja" else "缩放")
        self.zoom_label.setText(f"{prefix}: {value * 100:.0f}%")

    def _key(self, path: Path) -> str:
        try: return str(path.resolve())
        except OSError: return str(path)

    def _author_changed(self, text: str):
        self.measurement_author = text.strip()

    def on_measurements_changed(self):
        self._sync_current_measurements()
        self._update_measurement_list()

    def on_measurement_undo(self, action: str):
        if action == "cancel_pending":
            self.statusBar().showMessage(self.tr("measurement_undo_pending"), 2500)
        elif action == "undo_endpoint":
            self._sync_current_measurements()
            self._update_measurement_list()
            self.statusBar().showMessage(self.tr("measurement_undo_endpoint"), 3000)
        else:
            self.statusBar().showMessage(self.tr("measurement_undo_none"), 1800)

    def _sync_current_measurements(self):
        if self.current_image_path is not None:
            self.measurements_by_image[self._key(self.current_image_path)] = self.view.get_measurements()

    def _update_measurement_list(self):
        selected = self.view.selected_measurement_index
        self.measurement_list.blockSignals(True)
        self.measurement_list.clear()
        if self.current_image_path is None:
            self.measurement_list.blockSignals(False)
            self.view.set_selected_measurement(-1)
            return
        records = self.measurements_by_image.get(
            self._key(self.current_image_path), self.view.get_measurements()
        )
        for i, rec in enumerate(records, start=1):
            px = math.hypot(rec["x2"] - rec["x1"], rec["y2"] - rec["y1"])
            physical = (
                f" | {px * self.resolution_um_per_pixel:.4f} μm"
                if self.resolution_um_per_pixel is not None else ""
            )
            self.measurement_list.addItem(
                self.tr("measurement_item", id=i, px=px, physical=physical)
            )
        if 0 <= selected < self.measurement_list.count():
            self.measurement_list.setCurrentRow(selected)
        self.measurement_list.blockSignals(False)
        self.view.set_selected_measurement(selected)

    def _update_resolution_label(self):
        if self.resolution_um_per_pixel is None:
            self.resolution_value_label.setText(self.tr("resolution") + ": " + self.tr("not_set"))
        else:
            self.resolution_value_label.setText(self.tr("resolution_set", value=self.resolution_um_per_pixel))

    def reset_history(self, name: Optional[str] = None):
        name = name or self.tr("history_initial")
        self.history_masks=[]; self.history_names=[]; self.history_index=-1
        self.history_list.blockSignals(True); self.history_list.clear()
        if self.view.mask is not None:
            self.history_masks.append(self.view.mask.copy()); self.history_names.append(name); self.history_index=0
            self.history_list.addItem(f"0 · {name}"); self.history_list.setCurrentRow(0)
        self.history_list.blockSignals(False)

    def record_history_step(self, name: str):
        if self.restoring_history or self.view.mask is None: return
        mapped = {"画笔": self.tr("history_paint"), "橡皮擦": self.tr("history_erase"), "区域填充": self.tr("history_fill")}.get(name, name)
        if self.history_index < len(self.history_masks)-1:
            self.history_masks=self.history_masks[:self.history_index+1]; self.history_names=self.history_names[:self.history_index+1]
        if self.history_masks and np.array_equal(self.history_masks[-1], self.view.mask): return
        self.history_masks.append(self.view.mask.copy()); self.history_names.append(mapped)
        if len(self.history_masks) > self.MAX_HISTORY:
            self.history_masks.pop(0); self.history_names.pop(0)
        self.history_index=len(self.history_masks)-1
        self.history_list.blockSignals(True); self.history_list.clear()
        for i,n in enumerate(self.history_names): self.history_list.addItem(f"{i} · {n}")
        self.history_list.setCurrentRow(self.history_index); self.history_list.blockSignals(False)

    def restore_history_step(self, row: int):
        if row < 0 or row >= len(self.history_masks) or self.view.mask is None or row == self.history_index: return
        self.restoring_history=True; self.view.mask=self.history_masks[row].copy(); self.view.refresh(); self.history_index=row; self.mask_dirty=True; self._update_dirty_state(); self.area_window.update_from_mask(self.view.mask); self.restoring_history=False

    def on_fill_completed(self, filled_pixels: int):
        self.statusBar().showMessage(f"{filled_pixels:,} px", 2500)

    def on_measurement_created(self, length_pixels: float):
        self._sync_current_measurements(); self._update_measurement_list()
        physical = f" | {length_pixels * self.resolution_um_per_pixel:.4f} μm" if self.resolution_um_per_pixel is not None else ""
        self.statusBar().showMessage(self.tr("measurement_added", px=length_pixels, physical=physical), 3500)

    def delete_selected_measurement(self):
        row = self.measurement_list.currentRow()
        if self.current_image_path is None or row < 0: return
        key=self._key(self.current_image_path); records=self.measurements_by_image.get(key, self.view.get_measurements())
        if 0 <= row < len(records): records.pop(row)
        self.measurements_by_image[key]=records; self.view.set_measurements(records); self._update_measurement_list()

    def clear_current_measurements(self):
        if self.current_image_path is None: return
        if QMessageBox.question(self, self.tr("warning"), self.tr("confirm_clear_measure"), QMessageBox.Yes|QMessageBox.No, QMessageBox.No) != QMessageBox.Yes: return
        self.measurements_by_image[self._key(self.current_image_path)] = []; self.view.set_measurements([]); self._update_measurement_list()

    def set_resolution(self):
        dialog=ResolutionDialog(self.resolution_um_per_pixel, self.language, self)
        if dialog.exec()!=QDialog.Accepted: return
        self.resolution_um_per_pixel=dialog.resolution(); self.area_window.set_resolution(self.resolution_um_per_pixel); self.area_window.update_from_mask(self.view.mask); self.view.set_resolution(self.resolution_um_per_pixel); self._update_resolution_label(); self._update_measurement_list()
        self.statusBar().showMessage(self.tr("resolution_cleared") if self.resolution_um_per_pixel is None else self.tr("resolution_set", value=self.resolution_um_per_pixel), 3000)

    def export_measurements_csv(self):
        self._sync_current_measurements()
        if not any(self.measurements_by_image.values()): QMessageBox.information(self,self.tr("info"),self.tr("measurement_none")); return
        default=f"Measured-{datetime.now().strftime('%Y-%m-%d-%H-%M')}.csv"
        path,_=QFileDialog.getSaveFileName(self,self.tr("save_csv"),default,"CSV (*.csv)")
        if not path: return
        with open(path,'w',newline='',encoding='utf-8-sig') as f:
            w=csv.writer(f); w.writerow(["Author","Image Name","Stoma ID","x1","y1","x2","y2","Pixel Distance","Physical Distance (um)","Resolution (um/pixel)"])
            for image in self.image_files or ([self.current_image_path] if self.current_image_path else []):
                if image is None: continue
                for i,rec in enumerate(self.measurements_by_image.get(self._key(image),[]),start=1):
                    px=math.hypot(rec['x2']-rec['x1'],rec['y2']-rec['y1']); physical=px*self.resolution_um_per_pixel if self.resolution_um_per_pixel is not None else ""
                    w.writerow([self.measurement_author,image.name,i,rec['x1'],rec['y1'],rec['x2'],rec['y2'],f"{px:.4f}",f"{physical:.6f}" if physical!="" else "",self.resolution_um_per_pixel if self.resolution_um_per_pixel is not None else ""])
        self.statusBar().showMessage(self.tr("csv_saved",path=path),4000)

    def save_measurements_xml(self):
        self._sync_current_measurements()
        if not any(self.measurements_by_image.values()): QMessageBox.information(self,self.tr("info"),self.tr("measurement_none")); return
        default=f"Measured-{datetime.now().strftime('%Y-%m-%d-%H-%M')}.xml"
        path,_=QFileDialog.getSaveFileName(self,self.tr("save_xml"),default,"XML (*.xml)")
        if not path: return
        root=ET.Element("data")
        ET.SubElement(root,"Author").text=self.measurement_author
        images=self.image_files or ([self.current_image_path] if self.current_image_path else [])
        for image in images:
            if image is None: continue
            ET.SubElement(root,"imagePath").text=str(image)
            ET.SubElement(root,"Resolution").text=str(self.resolution_um_per_pixel if self.resolution_um_per_pixel is not None else "")
            for i,rec in enumerate(self.measurements_by_image.get(self._key(image),[]),start=1):
                m=ET.SubElement(root,"Measure"); m.set("Measure",str(i)); item=ET.SubElement(m,"item")
                ET.SubElement(item,"ltx").text=str(rec['x1']); ET.SubElement(item,"lty").text=str(rec['y1']); ET.SubElement(item,"rbx").text=str(rec['x2']); ET.SubElement(item,"rby").text=str(rec['y2'])
        ET.ElementTree(root).write(path,encoding='utf-8',xml_declaration=True)
        self.statusBar().showMessage(self.tr("xml_saved",path=path),4000)

    def load_measurements_xml(self):
        path,_=QFileDialog.getOpenFileName(self,self.tr("load_xml"),"","XML (*.xml);;All Files (*)")
        if not path: return
        try:
            root=ET.parse(path).getroot(); current: Optional[Path]=None; loaded={}; resolutions=[]
            loaded_author = None
            for child in root:
                if child.tag == "Author":
                    loaded_author = (child.text or "").strip()
                elif child.tag=="imagePath":
                    current=Path(child.text or ""); loaded.setdefault(self._key(current),[])
                elif child.tag=="Resolution":
                    try: resolutions.append(float((child.text or '').strip()))
                    except Exception: pass
                elif child.tag=="Measure" and current is not None:
                    item=child.find("item")
                    if item is None: continue
                    def num(name): return int(round(float(item.findtext(name,"0"))))
                    loaded[self._key(current)].append({"x1":num("ltx"),"y1":num("lty"),"x2":num("rbx"),"y2":num("rby")})
            self.measurements_by_image.update(loaded)
            if loaded_author is not None:
                self.measurement_author = loaded_author
                self.author_edit.blockSignals(True)
                self.author_edit.setText(loaded_author)
                self.author_edit.blockSignals(False)
            if resolutions and all(abs(resolutions[0]-r)<1e-12 for r in resolutions):
                self.resolution_um_per_pixel=resolutions[0]; self.view.set_resolution(self.resolution_um_per_pixel); self.area_window.set_resolution(self.resolution_um_per_pixel)
            if self.current_image_path is not None:
                self.view.set_measurements(self.measurements_by_image.get(self._key(self.current_image_path),[]))
            self._update_resolution_label(); self._update_measurement_list()
            QMessageBox.information(self,self.tr("info"),self.tr("xml_loaded",count=len(loaded),measures=sum(len(v) for v in loaded.values())))
        except Exception as exc:
            QMessageBox.critical(self,self.tr("error"),self.tr("xml_invalid",error=exc))

    def export_measurement_images(self):
        """导出测量结果图：原图区域只保留测量线和编号，右侧显示详细数值。"""
        self._sync_current_measurements()
        folder = QFileDialog.getExistingDirectory(self, self.tr("export_images"))
        if not folder:
            return

        output_dir = Path(folder)
        count = 0
        images = self.image_files or ([self.current_image_path] if self.current_image_path else [])

        # 与界面中的测量颜色保持一致：QColor(0, 255, 255) = cyan。
        measurement_bgr = (255, 255, 0)

        for image_path in images:
            if image_path is None:
                continue
            image = read_image(image_path, cv2.IMREAD_COLOR)
            if image is None:
                continue

            records = self.measurements_by_image.get(self._key(image_path), [])
            h, w = image.shape[:2]

            # 右侧详情区。若测量很多，允许画布向下扩展，避免内容被截断。
            panel_width = max(380, min(560, int(w * 0.38)))
            header_height = 112
            row_height = 34
            needed_height = header_height + max(1, len(records)) * row_height + 24
            canvas_h = max(h, needed_height)

            canvas = np.full((canvas_h, w + panel_width, 3), 255, dtype=np.uint8)
            canvas[:h, :w] = image

            # 原图上只画测量线和编号，不再覆盖 px / μm 详情。
            for i, rec in enumerate(records, start=1):
                p1 = (int(rec["x1"]), int(rec["y1"]))
                p2 = (int(rec["x2"]), int(rec["y2"]))
                cv2.line(canvas, p1, p2, measurement_bgr, 2, cv2.LINE_AA)

                mx = int((p1[0] + p2[0]) / 2)
                my = int((p1[1] + p2[1]) / 2)
                label = str(i)
                # 黑色描边 + 青色编号，可读但不会用色块遮挡图像。
                cv2.putText(
                    canvas, label, (mx + 5, my - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 0, 0), 4, cv2.LINE_AA
                )
                cv2.putText(
                    canvas, label, (mx + 5, my - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.68, measurement_bgr, 2, cv2.LINE_AA
                )

            # Qt 绘制右侧文字，系统字体可正确显示中文 / 日文 / 英文。
            qimage = numpy_to_qimage(canvas)
            painter = QPainter(qimage)
            painter.setRenderHint(QPainter.Antialiasing, True)

            separator_pen = QPen(QColor(190, 190, 190))
            separator_pen.setWidth(1)
            painter.setPen(separator_pen)
            painter.drawLine(w, 0, w, canvas_h)

            left = w + 24
            painter.setPen(QColor(25, 25, 25))
            title_font = QFont()
            title_font.setPointSize(15)
            title_font.setBold(True)
            painter.setFont(title_font)
            painter.drawText(left, 34, self.tr("export_detail_title"))

            info_font = QFont()
            info_font.setPointSize(10)
            painter.setFont(info_font)
            painter.setPen(QColor(85, 85, 85))
            if self.resolution_um_per_pixel is None:
                resolution_text = f'{self.tr("export_detail_resolution")}: -'
            else:
                resolution_text = (
                    f'{self.tr("export_detail_resolution")}: '
                    f'{self.resolution_um_per_pixel:g} μm/pixel'
                )
            author_text = f'{self.tr("export_detail_author")}: {self.measurement_author or "-"}'
            painter.drawText(left, 58, author_text)
            painter.drawText(left, 82, resolution_text)

            row_font = QFont()
            row_font.setPointSize(11)
            painter.setFont(row_font)
            y = header_height
            for i, rec in enumerate(records, start=1):
                px = math.hypot(
                    float(rec["x2"]) - float(rec["x1"]),
                    float(rec["y2"]) - float(rec["y1"]),
                )
                if self.resolution_um_per_pixel is None:
                    detail = f"ID {i}    {px:.2f} px"
                else:
                    physical = px * self.resolution_um_per_pixel
                    detail = f"ID {i}    {px:.2f} px    {physical:.4f} μm"

                painter.setPen(QColor(20, 20, 20))
                painter.drawText(left, y, detail)
                painter.setPen(QColor(225, 225, 225))
                painter.drawLine(left, y + 9, w + panel_width - 22, y + 9)
                y += row_height

            painter.end()

            target = output_dir / image_path.name
            if qimage.save(str(target)):
                count += 1

        QMessageBox.information(
            self, self.tr("info"),
            self.tr("images_exported", count=count, folder=folder)
        )

    def _confirm_discard(self) -> bool:
        if not self.mask_dirty and not self.yolo_dirty: return True
        return QMessageBox.question(self,self.tr("warning"),self.tr("confirm_unsaved"),QMessageBox.Yes|QMessageBox.No,QMessageBox.No)==QMessageBox.Yes

    def open_image(self):
        if not self._confirm_discard(): return
        path,_=QFileDialog.getOpenFileName(self,self.tr("open_image"),"","Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All Files (*)")
        if not path: return
        p=Path(path); self.image_folder=p.parent; self.mask_folder=None; self.yolo_label_folder=None; self.image_files=[p]; self.current_index=0; self._load_image_path(p)

    def open_folder(self):
        if not self._confirm_discard(): return
        folder=QFileDialog.getExistingDirectory(self,self.tr("open_folder"))
        if not folder: return
        p=Path(folder); files=sorted(x for x in p.iterdir() if x.is_file() and x.suffix.lower() in IMAGE_EXTENSIONS and not x.stem.lower().endswith(("_mask","_blend")))
        if not files: QMessageBox.information(self,self.tr("info"),self.tr("no_supported_images")); return
        self.image_folder=p; self.image_files=files; self.current_index=0; self._load_image_path(files[0]); self.statusBar().showMessage(self.tr("folder_status",path=p),4000)

    def open_mask_folder(self):
        folder=QFileDialog.getExistingDirectory(self,self.tr("mask_folder"))
        if not folder:return
        self.mask_folder=Path(folder)
        if self.current_image_path is not None: self._load_mask_for_image(self.current_image_path)
        self.statusBar().showMessage(self.tr("mask_folder_status",path=self.mask_folder),4000)

    def open_yolo_folder(self):
        folder=QFileDialog.getExistingDirectory(self,self.tr("yolo_folder"))
        if not folder:return
        self.yolo_label_folder=Path(folder)
        if self.current_image_path is not None: self._load_yolo_for_image(self.current_image_path)
        self.statusBar().showMessage(self.tr("yolo_folder_status",path=self.yolo_label_folder),4000)

    def _find_matching_mask(self,image_path:Path)->Optional[Path]:
        if self.mask_folder is not None and self.mask_folder.exists():
            for suffix in (image_path.suffix,".png",".bmp",".tif",".tiff",".jpg",".jpeg"):
                c=self.mask_folder/f"{image_path.stem}{suffix}"
                if c.exists() and c.is_file(): return c
                c=self.mask_folder/f"{image_path.stem}_mask{suffix}"
                if c.exists() and c.is_file(): return c
        for suffix in (".png",".bmp",".tif",".tiff"):
            c=image_path.with_name(f"{image_path.stem}_mask{suffix}")
            if c.exists() and c.is_file(): return c
        return None

    def _load_mask_for_image(self,image_path:Path):
        matched=self._find_matching_mask(image_path)
        if matched:
            mask=read_image(matched,cv2.IMREAD_UNCHANGED)
            if mask is not None:self.view.set_mask(mask,emit_change=False)
        self.mask_dirty=False; self._update_dirty_state(); self.area_window.update_from_mask(self.view.mask); self._refresh_file_label()

    def _yolo_label_path(self,image_path:Path)->Path:
        base=self.yolo_label_folder if self.yolo_label_folder is not None else image_path.parent
        return base/f"{image_path.stem}.txt"

    def _parse_yolo_label_file(self,label_path:Path)->tuple[list[dict],int]:
        annotations=[]; ignored=0
        if self.view.image_bgr is None:return annotations,ignored
        h,w=self.view.image_bgr.shape[:2]
        try: lines=label_path.read_text(encoding='utf-8').splitlines()
        except OSError:return annotations,ignored
        for line in lines:
            parts=line.strip().split()
            if not parts:continue
            try:
                cls=int(float(parts[0])); vals=[float(x) for x in parts[1:]]
                if len(vals)==4:
                    xc,yc,bw,bh=vals; x1=(xc-bw/2)*w; x2=(xc+bw/2)*w; y1=(yc-bh/2)*h; y2=(yc+bh/2)*h
                    points=np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]],dtype=float); typ='rect'
                elif len(vals)==8:
                    points=np.array([[vals[i]*w,vals[i+1]*h] for i in range(0,8,2)],dtype=float); typ='obb'
                else: ignored+=1; continue
                if not np.all(np.isfinite(points)): ignored+=1; continue
                annotations.append({'type':typ,'class_id':max(0,cls),'points':points})
            except Exception: ignored+=1
        return annotations,ignored

    def _load_yolo_for_image(self,image_path:Path):
        p=self._yolo_label_path(image_path)
        if p.exists():
            ann,ignored=self._parse_yolo_label_file(p); self.view.set_yolo_annotations(ann)
            if ignored:self.statusBar().showMessage(self.tr("yolo_ignored",count=ignored),3500)
        else:self.view.set_yolo_annotations([])
        self.yolo_dirty=False; self._update_dirty_state(); self._refresh_file_label()

    def _write_yolo_labels(self,output:Path)->bool:
        if self.view.image_bgr is None:return False
        h,w=self.view.image_bgr.shape[:2]; lines=[]
        for ann in self.view.get_yolo_annotations():
            pts=np.asarray(ann['points'],dtype=float); cls=int(ann['class_id'])
            if ann['type']=='rect':
                xmin,xmax=float(pts[:,0].min()),float(pts[:,0].max()); ymin,ymax=float(pts[:,1].min()),float(pts[:,1].max())
                lines.append(f"{cls} {((xmin+xmax)/2)/w:.6f} {((ymin+ymax)/2)/h:.6f} {(xmax-xmin)/w:.6f} {(ymax-ymin)/h:.6f}")
            else:
                vals=[]
                for x,y in pts: vals.extend([np.clip(x/w,0,1),np.clip(y/h,0,1)])
                lines.append(str(cls)+" "+" ".join(f"{v:.6f}" for v in vals))
        try: output.parent.mkdir(parents=True,exist_ok=True); output.write_text("\n".join(lines)+("\n" if lines else ""),encoding='utf-8'); return True
        except OSError:return False

    def open_yolo_labels(self):
        if not self.view.has_image():QMessageBox.information(self,self.tr("info"),self.tr("select_image_first"));return
        path,_=QFileDialog.getOpenFileName(self,self.tr("open_yolo"),"","YOLO (*.txt);;All Files (*)")
        if not path:return
        ann,ignored=self._parse_yolo_label_file(Path(path));self.view.set_yolo_annotations(ann);self.yolo_dirty=False;self._update_dirty_state();self._refresh_file_label()
        if ignored:self.statusBar().showMessage(self.tr("yolo_ignored",count=ignored),3500)

    def save_yolo_labels(self):
        if self.current_image_path is None:return
        output=self._yolo_label_path(self.current_image_path)
        if self._write_yolo_labels(output):self.yolo_dirty=False;self._update_dirty_state();self.statusBar().showMessage(self.tr("yolo_saved",path=output,count=len(self.view.yolo_annotations)),4000)
        else:QMessageBox.critical(self,self.tr("error"),self.tr("auto_save_failed",path=output))

    def _auto_save_current_yolo(self)->bool:
        if self.current_image_path is None:return True
        output=self._yolo_label_path(self.current_image_path)
        if not self._write_yolo_labels(output):QMessageBox.critical(self,self.tr("error"),self.tr("auto_save_failed",path=output));return False
        self.yolo_dirty=False;self._update_dirty_state();return True

    def mark_yolo_dirty(self):self.yolo_dirty=True;self._update_dirty_state();self._refresh_file_label()

    def on_yolo_selection_changed(self,index:int,class_id:int,annotation_type:str):
        self.yolo_class_spin.blockSignals(True);self.yolo_class_spin.setValue(max(0,int(class_id)));self.yolo_class_spin.blockSignals(False);self.view.current_yolo_class=max(0,int(class_id))
        self.yolo_selection_label.setText(self.tr("selected_yolo",type=annotation_type.upper(),index=index+1,class_id=class_id) if index>=0 else self.tr("no_selected_yolo"))

    def delete_selected_yolo(self):
        if self.view.delete_selected_yolo():self.statusBar().showMessage(self.tr("delete_annotation"),2000)

    def rotate_selected_obb(self,degrees:float):
        idx=self.view.selected_yolo_index
        if not (0<=idx<len(self.view.yolo_annotations)) or self.view.yolo_annotations[idx]['type']!='obb':self.statusBar().showMessage(self.tr("rotation_need_obb"),2500);return
        if not self.view.rotate_selected_obb(degrees):self.statusBar().showMessage(self.tr("rotation_outside"),2500)

    def navigate(self,step:int):
        if not self.image_files:return
        target=self.current_index+step
        if not 0<=target<len(self.image_files):return
        self._sync_current_measurements()
        if self.mask_dirty and not self._auto_save_current_mask():return
        if self.yolo_dirty and not self._auto_save_current_yolo():return
        self.current_index=target;self._load_image_path(self.image_files[target])

    def _auto_save_current_mask(self)->bool:
        if self.view.mask is None or self.current_image_path is None:return True
        output=(self.mask_folder/f"{self.current_image_path.stem}.png") if self.mask_folder is not None else self.current_image_path.with_name(f"{self.current_image_path.stem}_mask.png")
        output.parent.mkdir(parents=True,exist_ok=True)
        if not write_image(output,self.view.mask):QMessageBox.critical(self,self.tr("error"),self.tr("auto_save_failed",path=output));return False
        self.mask_dirty=False;self._update_dirty_state();return True

    def _related_sample_files(self,image_path:Path)->list[Path]:
        files=[image_path];m=self._find_matching_mask(image_path)
        if m:files.append(m)
        y=self._yolo_label_path(image_path)
        if y.exists():files.append(y)
        for suffix in IMAGE_EXTENSIONS:
            for ns in ("_mask","_blend"):
                c=image_path.with_name(f"{image_path.stem}{ns}{suffix}")
                if c.exists():files.append(c)
        unique=[];seen=set()
        for p in files:
            r=p.resolve()
            if r not in seen:seen.add(r);unique.append(p)
        return unique

    def delete_current_sample(self):
        if self.current_image_path is None:return
        image_path=self.current_image_path; related=self._related_sample_files(image_path); listing="\n".join(f"• {p}" for p in related)
        if QMessageBox.warning(self,self.tr("warning"),self.tr("confirm_delete_sample",files=listing),QMessageBox.Yes|QMessageBox.No,QMessageBox.No)!=QMessageBox.Yes:return
        failed=[]
        for p in related:
            try:
                if p.exists():p.unlink()
            except OSError:failed.append(p)
        if failed:QMessageBox.critical(self,self.tr("error"),self.tr("delete_failed",files="\n".join(map(str,failed))));return
        old=self.current_index; self.image_files=[p for p in self.image_files if p.resolve()!=image_path.resolve()]; self.measurements_by_image.pop(self._key(image_path),None); self.mask_dirty=False;self.yolo_dirty=False
        if not self.image_files:
            self.current_index=-1;self.current_image_path=None;self.view.image_bgr=None;self.view.mask=None;self.view.yolo_annotations=[];self.view.set_measurements([]);self.view.pixmap_item.setPixmap(QPixmap());self.reset_history();self._update_navigation_state();self._refresh_file_label();return
        self.current_index=min(old,len(self.image_files)-1);self._load_image_path(self.image_files[self.current_index]);self.statusBar().showMessage(self.tr("deleted",name=image_path.name),3500)

    def _load_image_path(self,path:Path):
        self._sync_current_measurements()
        image=read_image(path,cv2.IMREAD_COLOR)
        if image is None:QMessageBox.critical(self,self.tr("error"),self.tr("invalid_image",path=path));return
        self.view.set_image(image);self.current_image_path=path
        self._load_mask_for_image(path);self._load_yolo_for_image(path)
        self.view.set_resolution(self.resolution_um_per_pixel);self.view.set_measurements(self.measurements_by_image.get(self._key(path),[]));self._update_measurement_list()
        self.mask_dirty=False;self.yolo_dirty=False;self._update_dirty_state();self._update_navigation_state();self.area_window.update_from_mask(self.view.mask);self.reset_history();self._refresh_file_label();self.setWindowTitle(f"{self.tr('app_title')} - {path.name}")

    def _refresh_file_label(self):
        if self.current_image_path is None:self.file_label.setText(self.tr("no_image"));return
        path=self.current_image_path; m=self._find_matching_mask(path); y=self._yolo_label_path(path); count=len(self.measurements_by_image.get(self._key(path),self.view.get_measurements()))
        idx=f" ({self.current_index+1}/{len(self.image_files)})" if self.image_files and self.current_index>=0 else ""
        self.file_label.setText(f"{path.name}{idx} | {self.tr('mask')}: {m.name if m else self.tr('mask_not_found')} | {self.tr('yolo')}: {len(self.view.yolo_annotations)} | {self.tr('measurements')}: {count}")

    def _update_navigation_state(self):
        has=bool(self.image_files) and self.current_index>=0;self.previous_action.setEnabled(has and self.current_index>0);self.next_action.setEnabled(has and self.current_index<len(self.image_files)-1);self.delete_sample_action.setEnabled(has)

    def open_mask(self):
        if not self.view.has_image():QMessageBox.information(self,self.tr("info"),self.tr("select_image_first"));return
        path,_=QFileDialog.getOpenFileName(self,self.tr("open_mask"),"","Mask Images (*.png *.bmp *.tif *.tiff);;All Files (*)")
        if not path:return
        mask=read_image(Path(path),cv2.IMREAD_UNCHANGED)
        if mask is None:QMessageBox.critical(self,self.tr("error"),self.tr("invalid_image",path=path));return
        self.view.set_mask(mask,emit_change=False);self.mask_dirty=False;self._update_dirty_state();self.area_window.update_from_mask(self.view.mask);self.reset_history(self.tr("mask_loaded"))

    def _default_output(self,suffix:str)->str:
        if self.current_image_path is None:return ""
        if suffix=="_mask" and self.mask_folder is not None:return str(self.mask_folder/f"{self.current_image_path.stem}.png")
        return str(self.current_image_path.with_name(f"{self.current_image_path.stem}{suffix}.png"))

    def save_mask(self):
        if self.view.mask is None:QMessageBox.information(self,self.tr("info"),self.tr("select_image_first"));return
        path,_=QFileDialog.getSaveFileName(self,self.tr("save_mask"),self._default_output("_mask"),"PNG (*.png);;TIFF (*.tif *.tiff);;Bitmap (*.bmp)")
        if not path:return
        output=Path(path);output=output if output.suffix else output.with_suffix('.png')
        if not write_image(output,self.view.mask):QMessageBox.critical(self,self.tr("error"),self.tr("auto_save_failed",path=output));return
        self.mask_dirty=False;self._update_dirty_state();self.statusBar().showMessage(self.tr("mask_saved",path=output),4000)

    def save_blend(self):
        blend=self.view.create_blend()
        if blend is None:return
        path,_=QFileDialog.getSaveFileName(self,self.tr("save_blend"),self._default_output("_blend"),"PNG (*.png);;JPEG (*.jpg *.jpeg);;TIFF (*.tif *.tiff)")
        if not path:return
        output=Path(path);output=output if output.suffix else output.with_suffix('.png')
        if not write_image(output,blend):QMessageBox.critical(self,self.tr("error"),self.tr("auto_save_failed",path=output));return
        self.statusBar().showMessage(self.tr("blend_saved",path=output),4000)

    def show_area_statistics(self):
        self.area_window.set_resolution(self.resolution_um_per_pixel);self.area_window.update_from_mask(self.view.mask);self.area_window.show();self.area_window.raise_();self.area_window.activateWindow()

    def clear_mask(self):
        if self.view.mask is None:return
        if QMessageBox.question(self,self.tr("warning"),self.tr("confirm_clear_mask"),QMessageBox.Yes|QMessageBox.No,QMessageBox.No)==QMessageBox.Yes:self.view.clear_mask();self.record_history_step(self.tr("history_clear"))

    def pick_label_from_mask(self,value:int):
        if value==0:self.erase_action.setChecked(True);self.view.set_mode("erase")
        else:self.label_spin.setValue(value);self.paint_action.setChecked(True);self.view.set_mode("paint")

    def adjust_brush_size(self,direction:int):self.brush_spin.setValue(self.brush_spin.value()+direction)
    def adjust_label_value(self,direction:int):self.label_spin.setValue(self.label_spin.value()+direction);self.paint_action.setChecked(True);self.view.set_mode("paint")
    def toggle_mask_visibility(self):self.mask_visible=not self.mask_visible;self.view.set_overlay_visible(self.mask_visible)

    def _update_label_swatch(self):
        color=label_color(self.label_spin.value());self.color_swatch.setStyleSheet(f"background: rgb({color.red()}, {color.green()}, {color.blue()}); border: 1px solid #777; border-radius: 3px;")

    def mark_dirty(self):self.mask_dirty=True;self._update_dirty_state()
    def _update_dirty_state(self):
        parts=[]
        if self.mask_dirty:parts.append("Mask")
        if self.yolo_dirty:parts.append("YOLO")
        self.dirty_label.setText((self.tr("unsaved")+": "+"/".join(parts)) if parts else "")

    def closeEvent(self,event):
        self._sync_current_measurements()
        if self._confirm_discard():event.accept()
        else:event.ignore()


def main():
    app=QApplication(sys.argv)
    app.setApplicationName("SMART Annotator")
    window=MainWindow();window.show();sys.exit(app.exec())


if __name__ == "__main__":
    main()
