import numpy as np

from steel_pipeline import (
    class_nms, decode_rle, mask_boxes, predict_tiled, tile_starts,
    yolo_lines_for_tile,
)


class FakeTensor:
    def __init__(self, array):
        self.array = np.asarray(array)

    def cpu(self):
        return self

    def numpy(self):
        return self.array


class FakeBoxes:
    def __init__(self):
        self.cls = FakeTensor([1])
        self.conf = FakeTensor([0.8])
        self.xyxy = FakeTensor([[100, 100, 200, 200]])

    def __len__(self):
        return 1


class FakeResult:
    boxes = FakeBoxes()


class FakeModel:
    def predict(self, images, **kwargs):
        return [FakeResult() for _ in images]


def test_geometry():
    assert tile_starts(1600) == [0, 480, 960]
    mask = decode_rle("2 2", 4, 3)
    assert mask_boxes(mask, 0, min_area=1) == [(0, 0, 1, 1, 3)]
    lines = yolo_lines_for_tile({0: mask}, 0, 3, 4, min_area=1)
    assert len(lines) == 1 and lines[0].startswith("0 ")

    image = np.zeros((256, 1600, 3), dtype=np.uint8)
    detections = predict_tiled(FakeModel(), image, conf=0.1, batch=8)
    assert len(detections) == 3
    assert np.allclose(detections[0][2:], [100, 50, 200, 100])
    assert np.allclose(detections[1][2:], [580, 50, 680, 100])
    assert np.allclose(detections[2][2:], [1060, 50, 1160, 100])

    duplicates = [(0, 0.9, 1, 1, 11, 11), (0, 0.8, 1, 1, 11, 11)]
    assert len(class_nms(duplicates)) == 1


if __name__ == "__main__":
    test_geometry()
    print("Geometry tests passed")
