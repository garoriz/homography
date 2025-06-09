import cv2
import numpy as np

class CornerSelector:
    def __init__(self, image, window_name="Выберите 4 угла"):
        self.image = image.copy()
        self.display_image = image.copy()
        self.window_name = window_name
        self.corners = []
        self.is_completed = False

    def select_corners(self):
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

        while True:
            cv2.imshow(self.window_name, self.display_image)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or self.is_completed:
                break

        cv2.destroyWindow(self.window_name)
        if not self.is_completed:
            return None

        return np.array(self.corners, dtype=np.float32)

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.corners) < 4:
            self.corners.append((x, y))
            print(f"Точка {len(self.corners)}: ({x}, {y})")

            self.display_image = self.image.copy()
            for i, (cx, cy) in enumerate(self.corners, 1):
                cv2.circle(self.display_image, (cx, cy), 10, (0, 0, 255), -1)
                cv2.putText(self.display_image, str(i), (cx + 10, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            if len(self.corners) == 4:
                self.is_completed = True
                print("Выбраны все 4 угла!")

if __name__ == "__main__":
    image = cv2.imread("document.png")
    if image is None:
        print("Ошибка: изображение не найдено.")
        exit(1)

    selector = CornerSelector(image)
    src_corner_points = selector.select_corners()

    if src_corner_points is not None:
        print("\nsrc_corner_points:")
        print(src_corner_points)

        a4_aspect_ratio = 297 / 210
        dst_width = 600
        dst_height = int(dst_width * a4_aspect_ratio)

        dst_corner_points = np.array([
            [0, 0],
            [dst_width - 1, 0],
            [dst_width - 1, dst_height - 1],
            [0, dst_height - 1]
        ], dtype=np.float32)

        H = cv2.getPerspectiveTransform(src_corner_points, dst_corner_points)
        corrected_image = cv2.warpPerspective(image, H, (dst_width, dst_height))

        h1, w1 = image.shape[:2]
        h2, w2 = corrected_image.shape[:2]

        target_height = max(h1, h2)


        def resize_to_height(img, target_h):
            h, w = img.shape[:2]
            scale = target_h / h
            new_w = int(w * scale)
            return cv2.resize(img, (new_w, target_h))


        image_resized = resize_to_height(image, target_height)
        corrected_resized = resize_to_height(corrected_image, target_height)

        separator = 255 * np.ones((target_height, 32, 3), dtype=np.uint8)

        side_by_side = np.hstack([image_resized, separator, corrected_resized])

        cv2.imshow("Сравнение: исходное и исправленное", side_by_side)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Процесс выбора углов не был завершен.")
