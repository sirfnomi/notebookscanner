"""
Next-Gen AI Document & Book Scanner Engine - Core Pipeline
Powered by Deep Learning:
1. Deep Learning (MobileNetV3) Document Corner Keypoint Detector & Perspective Rectification
2. Auto-Orientation & Deskew Normalization
3. Finger & Thumb Inpainting
4. vFlat-Style Illumination Regularization & Paper Whitening
5. Multi-Page PDF Assembly
"""

import os
import time
import glob
import urllib.request
import cv2
import numpy as np
import img2pdf
import onnxruntime as ort

MODEL_URL = "https://huggingface.co/spaces/KennethTM/document_corner_detector/resolve/main/models/timm-mobilenetv3_small_100.onnx"
MODEL_PATH = "models/timm-mobilenetv3_small_100.onnx"


def ensure_ai_model():
    """Ensures the Deep Learning corner detection model is downloaded."""
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1000000:
        print("Downloading Deep Learning Document Corner Detector (13.7 MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded successfully!")


class AIDocumentCornerDetector:
    """Deep Learning Neural Network for 4-Corner Document Detection & Perspective Warp"""
    _session = None

    @classmethod
    def get_session(cls):
        if cls._session is None:
            ensure_ai_model()
            cls._session = ort.InferenceSession(MODEL_PATH)
        return cls._session

    @staticmethod
    def normalize_image(image, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        image = (image / 255.0).astype("float32")
        image[:, :, 0] = (image[:, :, 0] - mean[0]) / std[0]
        image[:, :, 1] = (image[:, :, 1] - mean[1]) / std[1]
        image[:, :, 2] = (image[:, :, 2] - mean[2]) / std[2]
        return image

    @staticmethod
    def resize_longest_max_size(image, max_size=224):
        height, width = image.shape[:2]
        ratio = max_size / max(width, height)
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def pad_if_needed(image, target_size=224):
        height, width, _ = image.shape
        y0 = abs((height - target_size) // 2)
        x0 = abs((width - target_size) // 2)
        background = np.zeros((target_size, target_size, 3), dtype="uint8")
        background[y0:(y0 + height), x0:(x0 + width), :] = image
        return background

    @staticmethod
    def heatmap2keypoints(heatmap: np.ndarray, img_size: int = 224) -> list:
        indx = heatmap.reshape(-1, img_size * img_size).argmax(axis=1)
        row = indx // img_size
        col = indx % img_size
        return np.stack((col, row), axis=1).tolist()

    @staticmethod
    def centercrop_keypoints(keypoints, crop_height, crop_width, img_size=224):
        y_diff = (img_size - crop_height) // 2
        x_diff = (img_size - crop_width) // 2
        return [[x - x_diff, y - y_diff] for x, y in keypoints]

    @staticmethod
    def resize_keypoints(keypoints, current_height, current_width, target_height, target_width):
        return [[int((x / current_width) * target_width), int((y / current_height) * target_height)] for x, y in keypoints]

    @classmethod
    def predict_corners(cls, image_bgr: np.ndarray) -> np.ndarray:
        session = cls.get_session()
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w, _ = image_rgb.shape

        image_resize = cls.resize_longest_max_size(image_rgb)
        h_small, w_small, _ = image_resize.shape
        image_pad = cls.pad_if_needed(image_resize, target_size=224)
        image_norm = cls.normalize_image(image_pad)
        image_array = np.transpose(image_norm, (2, 0, 1))
        image_array = np.expand_dims(image_array, axis=0)

        output = session.run([output_name], {input_name: image_array})
        output_keypoints = cls.heatmap2keypoints(output[0].squeeze())
        crop_keypoints = cls.centercrop_keypoints(output_keypoints, h_small, w_small, 224)
        large_keypoints = cls.resize_keypoints(crop_keypoints, h_small, w_small, h, w)
        return np.float32(large_keypoints)

    @classmethod
    def warp_perspective(cls, image_bgr: np.ndarray) -> np.ndarray:
        pts = cls.predict_corners(image_bgr)
        # pts order: TL, TR, BR, BL
        w_top = np.linalg.norm(pts[1] - pts[0])
        w_bot = np.linalg.norm(pts[2] - pts[3])
        target_w = int(max(w_top, w_bot))

        h_left = np.linalg.norm(pts[3] - pts[0])
        h_right = np.linalg.norm(pts[2] - pts[1])
        target_h = int(max(h_left, h_right))

        if target_w < 100 or target_h < 100:
            return image_bgr

        target_pts = np.float32([
            [0, 0],
            [target_w - 1, 0],
            [target_w - 1, target_h - 1],
            [0, target_h - 1]
        ])

        M = cv2.getPerspectiveTransform(pts, target_pts)
        return cv2.warpPerspective(image_bgr, M, (target_w, target_h), flags=cv2.INTER_CUBIC)


class DocumentPreprocessingEngine:
    """Orientation and subtle rotation normalization"""
    @staticmethod
    def detect_and_fix_orientation(image: np.ndarray) -> np.ndarray:
        try:
            import pytesseract
            small = cv2.resize(image, (640, int(640 * image.shape[0] / image.shape[1])))
            osd = pytesseract.image_to_osd(small, output_type=pytesseract.Output.DICT)
            angle = osd.get('rotate', 0)
            if angle == 90:
                return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            elif angle == 180:
                return cv2.rotate(image, cv2.ROTATE_180)
            elif angle == 270:
                return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        except Exception:
            pass
        return image


class OcclusionRemovalEngine:
    """Finger / Thumb Segmentation & Clean Margin Inpainting"""
    @staticmethod
    def detect_finger_mask(image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        border_mask = np.zeros((h, w), dtype=np.uint8)
        border_w = int(w * 0.08)
        border_h = int(h * 0.08)
        border_mask[:border_h, :] = 255
        border_mask[-border_h:, :] = 255
        border_mask[:, :border_w] = 255
        border_mask[:, -border_w:] = 255

        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        skin_ycrcb = cv2.inRange(ycrcb, np.array([0, 133, 77]), np.array([255, 173, 127]))
        skin_hsv = cv2.inRange(hsv, np.array([0, 25, 50]), np.array([30, 220, 255]))

        combined_skin = cv2.bitwise_and(skin_ycrcb, skin_hsv)
        candidate_mask = cv2.bitwise_and(combined_skin, combined_skin, mask=border_mask)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        candidate_mask = cv2.morphologyEx(candidate_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(candidate_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            if cv2.contourArea(c) > (w * h * 0.002):
                cv2.drawContours(mask, [c], -1, 255, -1)

        return mask

    @staticmethod
    def inpaint_fingers(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if np.count_nonzero(mask) == 0:
            return image
        dilated_mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), iterations=2)
        return cv2.inpaint(image, dilated_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)


class IlluminationRegularizationEngine:
    """vFlat / CamScanner-Style Illumination Regularization & Paper Whitening"""
    @staticmethod
    def whiten_paper_vflat_style(image: np.ndarray, whiteness_gain: float = 1.15) -> np.ndarray:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        h, w = l.shape
        scale = 800.0 / max(h, w)
        sw, sh = int(w * scale), int(h * scale)
        l_small = cv2.resize(l, (sw, sh))

        k_size = 51
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
        bg_small = cv2.morphologyEx(l_small, cv2.MORPH_CLOSE, kernel)
        bg_small = cv2.GaussianBlur(bg_small, (51, 51), 0)

        bg = cv2.resize(bg_small, (w, h))

        l_float = l.astype(np.float32)
        bg_float = np.maximum(bg.astype(np.float32), 1.0)
        l_norm = (l_float / bg_float) * 235.0

        l_white = np.clip(l_norm * whiteness_gain, 0, 255)
        paper_mask = l_white > 220
        l_white[paper_mask] = 220 + (l_white[paper_mask] - 220) * 1.0
        l_final = np.clip(l_white, 0, 255).astype(np.uint8)

        lab_clean = cv2.merge([l_final, a, b])
        result = cv2.cvtColor(lab_clean, cv2.COLOR_LAB2BGR)

        blur = cv2.GaussianBlur(result, (0, 0), 3.0)
        enhanced = cv2.addWeighted(result, 1.2, blur, -0.2, 0)
        return np.clip(enhanced, 0, 255).astype(np.uint8)


class ScannerPipelineOrchestrator:
    """End-to-End Deep Learning Batch Orchestrator and PDF Assembly"""
    def __init__(self, enable_orientation: bool = True, enable_ai_warp: bool = True,
                 enable_finger_removal: bool = True, enable_whitening: bool = True):
        self.enable_orientation = enable_orientation
        self.enable_ai_warp = enable_ai_warp
        self.enable_finger_removal = enable_finger_removal
        self.enable_whitening = enable_whitening

    def process_frame(self, image: np.ndarray) -> dict:
        timings = {}
        stages = {'0_raw': image.copy()}
        current = image.copy()

        # 1. Orientation
        t0 = time.time()
        if self.enable_orientation:
            current = DocumentPreprocessingEngine.detect_and_fix_orientation(current)
        timings['orientation_ms'] = round((time.time() - t0) * 1000, 1)
        stages['1_oriented'] = current.copy()

        # 2. Deep Learning 4-Corner Detection & Perspective Rectification
        t0 = time.time()
        if self.enable_ai_warp:
            current = AIDocumentCornerDetector.warp_perspective(current)
        timings['ai_perspective_warp_ms'] = round((time.time() - t0) * 1000, 1)
        stages['2_ai_warped'] = current.copy()

        # 3. Finger Removal
        t0 = time.time()
        if self.enable_finger_removal:
            mask = OcclusionRemovalEngine.detect_finger_mask(current)
            stages['finger_mask'] = mask.copy()
            current = OcclusionRemovalEngine.inpaint_fingers(current, mask)
        timings['finger_removal_ms'] = round((time.time() - t0) * 1000, 1)
        stages['3_inpainted'] = current.copy()

        # 4. Paper Whitening & Illumination Regularization
        t0 = time.time()
        if self.enable_whitening:
            current = IlluminationRegularizationEngine.whiten_paper_vflat_style(current)
        timings['whitening_ms'] = round((time.time() - t0) * 1000, 1)
        stages['4_whitened_final'] = current.copy()

        timings['total_pipeline_ms'] = round(sum(timings.values()), 1)
        return {'final': current, 'stages': stages, 'timings': timings}

    @staticmethod
    def compile_batch_to_pdf(processed_image_paths: list, output_pdf_path: str = "scanned_document.pdf") -> str:
        if not processed_image_paths:
            raise ValueError("No processed images provided for PDF compilation.")

        a4_in_pt = (img2pdf.mm_to_pt(210), img2pdf.mm_to_pt(297))
        layout_fun = img2pdf.get_layout_fun(pagesize=a4_in_pt, fit=img2pdf.FitMode.into)

        with open(output_pdf_path, "wb") as f:
            f.write(img2pdf.convert(processed_image_paths, layout_fun=layout_fun))

        return output_pdf_path


def main():
    import argparse
    parser = argparse.ArgumentParser(description="AI Book & Document Scanner Engine (Deep Learning)")
    parser.add_argument("--input", "-i", type=str, default="input_images", help="Input directory of image photos")
    parser.add_argument("--output", "-o", type=str, default="scanned_book.pdf", help="Output PDF file path")
    args = parser.parse_args()

    extensions = ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG')
    images = []
    for ext in extensions:
        images.extend(glob.glob(os.path.join(args.input, ext)))
    images = sorted(list(set(images)))

    if not images:
        print(f"No image files found in {args.input}.")
        return

    print(f"Found {len(images)} images in {args.input}. Processing with Deep Learning...")
    orchestrator = ScannerPipelineOrchestrator()

    temp_out_dir = os.path.join(args.input, "_processed_temp")
    os.makedirs(temp_out_dir, exist_ok=True)
    processed_paths = []

    for idx, img_path in enumerate(images):
        name = os.path.basename(img_path)
        print(f"[{idx+1}/{len(images)}] Processing {name}...", end=" ")
        raw = cv2.imread(img_path)
        if raw is None:
            print("Failed to read image.")
            continue
        res = orchestrator.process_frame(raw)
        out_page = os.path.join(temp_out_dir, f"page_{idx+1:04d}.jpg")
        cv2.imwrite(out_page, res['final'])
        processed_paths.append(out_page)
        print(f"Done in {res['timings']['total_pipeline_ms']} ms")

    if processed_paths:
        print(f"\nAssembling {len(processed_paths)} pages into PDF: {args.output}...")
        ScannerPipelineOrchestrator.compile_batch_to_pdf(processed_paths, args.output)
        print(f"SUCCESS! Output saved to: {args.output}")


if __name__ == "__main__":
    main()
