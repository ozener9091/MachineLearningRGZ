"""Общая обработка полос стали для Colab и локального интерфейса."""

from __future__ import annotations

import cv2
import numpy as np

TARGET_HEIGHT = 256
TILE_WIDTH = 640
TILE_STEP = 480
MODEL_HEIGHT = 512
MODEL_WIDTH = 640


def tile_starts(width: int, tile_width: int = TILE_WIDTH, step: int = TILE_STEP) -> list[int]:
    if width <= tile_width:
        return [0]
    starts = list(range(0, width - tile_width + 1, step))
    last = width - tile_width
    if starts[-1] != last:
        starts.append(last)
    return starts


def decode_rle(rle: str, height: int, width: int) -> np.ndarray:
    """Severstal RLE: индексы с 1, порядок пикселей по столбцам."""
    flat = np.zeros(height * width, dtype=np.uint8)
    values = np.fromstring(str(rle), sep=" ", dtype=np.int64)
    if len(values) % 2:
        raise ValueError("Некорректная RLE-разметка")
    for start, length in values.reshape(-1, 2):
        begin = int(start) - 1
        end = begin + int(length)
        if begin < 0 or end > flat.size:
            raise ValueError("RLE выходит за границы изображения")
        flat[begin:end] = 1
    return flat.reshape((height, width), order="F")


def mask_boxes(mask: np.ndarray, class_id: int, min_area: int = 10) -> list[tuple[int, int, int, int, int]]:
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    boxes = []
    for i in range(1, n):
        x, y, w, h, area = map(int, stats[i])
        if area >= min_area:
            boxes.append((class_id, x, y, x + w, y + h))
    return boxes


def yolo_lines_for_tile(
    masks: dict[int, np.ndarray], start: int, actual_width: int,
    height: int, tile_width: int = TILE_WIDTH, min_area: int = 10,
) -> list[str]:
    lines = []
    for class_id, mask in masks.items():
        crop = mask[:, start:start + actual_width]
        for _, x1, y1, x2, y2 in mask_boxes(crop, class_id, min_area):
            lines.append(
                f"{class_id} {(x1+x2)/(2*tile_width):.6f} "
                f"{(y1+y2)/(2*height):.6f} "
                f"{(x2-x1)/tile_width:.6f} {(y2-y1)/height:.6f}"
            )
    return lines


def prepare_model_tile(frame_bgr: np.ndarray, start: int) -> np.ndarray:
    """Вход высотой 256; результат 512x640 для YOLO."""
    height, width = frame_bgr.shape[:2]
    if height != TARGET_HEIGHT:
        raise ValueError(f"Ожидается высота {TARGET_HEIGHT}, получена {height}")
    tile = np.zeros((TARGET_HEIGHT, TILE_WIDTH, 3), dtype=np.uint8)
    crop = frame_bgr[:, start:start + TILE_WIDTH]
    tile[:, :crop.shape[1]] = crop
    return cv2.resize(tile, (MODEL_WIDTH, MODEL_HEIGHT), interpolation=cv2.INTER_LINEAR)


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    intersection_wh = np.maximum(0.0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2]))
    intersection = float(intersection_wh[0] * intersection_wh[1])
    area_a = float(np.prod(np.maximum(0.0, a[2:] - a[:2])))
    area_b = float(np.prod(np.maximum(0.0, b[2:] - b[:2])))
    return intersection / max(area_a + area_b - intersection, 1e-9)


def class_nms(detections: list[tuple], threshold: float = 0.50) -> list[tuple]:
    """Строка: (class_id, confidence, x1, y1, x2, y2)."""
    chosen = []
    for det in sorted(detections, key=lambda item: -item[1]):
        box = np.asarray(det[2:], dtype=float)
        if all(det[0] != kept[0] or iou_xyxy(box, np.asarray(kept[2:], dtype=float)) < threshold
               for kept in chosen):
            chosen.append(det)
    return chosen


def predict_tiled(model, image_bgr: np.ndarray, *, conf: float = 0.15,
                  device: str | int = "cpu", batch: int = 8) -> list[tuple]:
    """Предсказать на перекрывающихся фрагментах, вернуть рамки исходного изображения."""
    original_height, original_width = image_bgr.shape[:2]
    if original_height == 0 or original_width == 0:
        return []
    scaled_width = max(1, round(original_width * TARGET_HEIGHT / original_height))
    if original_height == TARGET_HEIGHT:
        scaled = image_bgr
    else:
        scaled = cv2.resize(image_bgr, (scaled_width, TARGET_HEIGHT), interpolation=cv2.INTER_AREA)
    starts = tile_starts(scaled_width)
    detections = []
    for offset in range(0, len(starts), batch):
        batch_starts = starts[offset:offset + batch]
        images = [prepare_model_tile(scaled, start) for start in batch_starts]
        results = model.predict(images, conf=conf, imgsz=640, device=device, verbose=False)
        for start, result in zip(batch_starts, results):
            if result.boxes is None or len(result.boxes) == 0:
                continue
            classes = result.boxes.cls.cpu().numpy().astype(int)
            scores = result.boxes.conf.cpu().numpy()
            boxes = result.boxes.xyxy.cpu().numpy()
            for class_id, score, (x1, y1, x2, y2) in zip(classes, scores, boxes):
                # Модель видит tile 640x512; реальные координаты — 640x256.
                x1 = max(0.0, min(float(x1 + start), float(scaled_width)))
                x2 = max(0.0, min(float(x2 + start), float(scaled_width)))
                y1 = max(0.0, min(float(y1 / 2), float(TARGET_HEIGHT)))
                y2 = max(0.0, min(float(y2 / 2), float(TARGET_HEIGHT)))
                if x2 > x1 and y2 > y1:
                    sx, sy = original_width / scaled_width, original_height / TARGET_HEIGHT
                    detections.append((int(class_id), float(score),
                                       x1 * sx, y1 * sy, x2 * sx, y2 * sy))
    return class_nms(detections)


def draw_detections(image_bgr: np.ndarray, detections: list[tuple],
                    names: dict[int, str], conf: float = 0.25) -> np.ndarray:
    output = image_bgr.copy()
    colors = [(50, 210, 50), (220, 150, 60), (90, 80, 230), (190, 60, 190)]
    for class_id, score, x1, y1, x2, y2 in detections:
        if score < conf:
            continue
        color = colors[class_id % len(colors)]
        p1, p2 = (round(x1), round(y1)), (round(x2), round(y2))
        cv2.rectangle(output, p1, p2, color, 2)
        # cv2 не умеет кириллицу: класс указываем номером.
        label = f"{class_id + 1}: {score:.2f}"
        cv2.putText(output, label, (p1[0], max(16, p1[1] - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return output
