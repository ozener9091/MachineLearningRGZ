"""Локальный интерфейс для обученной в Colab модели Severstal YOLO."""

from __future__ import annotations

import argparse
import json
import tempfile
from collections import Counter
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import torch
from ultralytics import YOLO
from steel_pipeline import draw_detections, predict_tiled


def build_app(model_path: Path, device: str) -> gr.Blocks:
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Не найден файл модели: {model_path}\n"
            "Скачайте best.pt из Colab и положите его рядом с local_app.py "
            "или передайте путь через --model."
        )

    model = YOLO(str(model_path))
    names = model.names
    config_path = (model_path.with_name("model_config.json")
                   if model_path.name == "best.pt" else model_path.with_suffix(".json"))
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    tiled = config.get("inference") == "tiled_640x256"
    default_confidence = float(config.get("confidence", 0.25))

    def detect(frame_bgr: np.ndarray, confidence: float):
        if tiled:
            detections = predict_tiled(model, frame_bgr, conf=min(confidence, 0.05), device=device)
            annotated = draw_detections(frame_bgr, detections, names, conf=confidence)
            classes = [class_id for class_id, score, *_ in detections if score >= confidence]
            return annotated, classes
        result = model.predict(
            frame_bgr, imgsz=640, conf=confidence, device=device, verbose=False
        )[0]
        return result.plot(), result.boxes.cls.cpu().numpy().astype(int).tolist()

    def predict(frame_rgb: np.ndarray, confidence: float):
        if frame_rgb is None:
            return None, "Загрузите изображение."
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        annotated_bgr, classes = detect(frame_bgr, confidence)
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        counts = Counter(classes)
        summary = ", ".join(f"{names[class_id]}: {count}" for class_id, count in sorted(counts.items()))
        return annotated_rgb, summary or "Дефекты не обнаружены при выбранном пороге."

    def predict_video(video_path: str | None, confidence: float, max_frames: int):
        if not video_path:
            return None, "Загрузите видео."
        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            return None, "Не удалось открыть видеофайл."

        fps = capture.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps) or fps <= 0:
            fps = 25.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width <= 0 or height <= 0:
            capture.release()
            return None, "Неверные размеры видео."

        output = Path(tempfile.mkdtemp(prefix="steel_yolo_")) / "result.mp4"
        writer = cv2.VideoWriter(
            str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        if not writer.isOpened():
            capture.release()
            return None, "Не удалось создать выходной MP4-файл."

        processed = found = 0
        try:
            while processed < int(max_frames):
                ok, frame = capture.read()
                if not ok:
                    break
                annotated, classes = detect(frame, confidence)
                writer.write(annotated)
                processed += 1
                found += len(classes)
        finally:
            capture.release()
            writer.release()

        if processed == 0:
            output.unlink(missing_ok=True)
            return None, "Видео не содержит доступных кадров."
        return str(output), f"Обработано кадров: {processed}; найдено рамок: {found}"

    def webcam(frame_rgb: np.ndarray, confidence: float):
        image, _ = predict(frame_rgb, confidence)
        return image

    with gr.Blocks(title="Дефекты стали — локальная YOLO") as app:
        gr.Markdown(
            "# Обнаружение дефектов стали\n"
            f"Модель: `{model_path.name}` · Устройство: `{device}` · "
            f"Режим: `{'фрагменты' if tiled else 'целое изображение'}`. "
            "Четыре класса исходного датасета Severstal."
        )
        confidence = gr.Slider(
            minimum=0.05, maximum=0.95, value=default_confidence, step=0.05,
            label="Порог уверенности",
        )

        with gr.Tab("Изображение"):
            input_image = gr.Image(type="numpy", label="Загрузить изображение")
            image_button = gr.Button("Найти дефекты", variant="primary")
            output_image = gr.Image(label="Результат")
            image_summary = gr.Textbox(label="Найденные дефекты")
            image_button.click(
                predict, [input_image, confidence], [output_image, image_summary]
            )

        with gr.Tab("Видео"):
            input_video = gr.Video(sources=["upload", "webcam"], label="Файл или запись с камеры")
            max_frames = gr.Slider(
                minimum=30, maximum=3000, value=300, step=30,
                label="Максимум кадров",
            )
            video_button = gr.Button("Обработать видео", variant="primary")
            output_video = gr.Video(format="mp4", label="Результат")
            video_summary = gr.Textbox(label="Сводка")
            video_button.click(
                predict_video,
                [input_video, confidence, max_frames],
                [output_video, video_summary],
            )

        with gr.Tab("Камера онлайн"):
            gr.Markdown("Разрешите браузеру доступ к камере. Скорость зависит от процессора или GPU.")
            live_input = gr.Image(sources=["webcam"], type="numpy", label="Камера")
            live_output = gr.Image(label="Распознавание", streaming=True)
            live_input.stream(
                webcam, [live_input, confidence], [live_output],
                time_limit=60, stream_every=0.3, concurrency_limit=1,
            )

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path(__file__).with_name("best.pt"))
    parser.add_argument("--device", default="auto", help="auto, cpu или cuda:0")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    device = ("cuda:0" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    app = build_app(args.model.expanduser().resolve(), device)
    app.launch(server_name="127.0.0.1", server_port=args.port, share=False, inbrowser=True)


if __name__ == "__main__":
    main()
