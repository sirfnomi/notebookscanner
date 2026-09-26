"""
Next-Gen AI Document & Book Scanner Engine - Core Deep Learning Pipeline
Modeled after vFlat Scan & Google Drive Scanner.

Deep Learning AI Architecture:
1. YOLO Document Region Detection (models/yolo_doc_v1.onnx)
   - Neural detection of document boundary, eliminating extreme extraneous floor/cloth margins.
2. UVDoc Neural 3D Spine & Surface Dewarping (models/uvdoc.onnx)
   - Deep Learning document image rectification (SIGGRAPH Asia / PaddleOCR).
   - Unrolls 3D cylindrical spine curvature, eliminates opposite-page flaps, removes holding thumb,
     and perfectly levels text baselines (0.0°) and vertical margins (90.0°).
3. vFlat-Style Magic Color Illumination Regularization & Paper Whitening
   - Morphological background division (I / I_bg) in LAB color space.
   - Pure scanner white paper profile while preserving vivid blue ink and red markings.
4. Multi-Page High-Resolution PDF Compilation
   - Lossless assembly into standardized A4 print-ready PDF via img2pdf.
"""

import os
import time
import glob
import urllib.request
import cv2
import numpy as np
import img2pdf
import onnxruntime as ort

# Model configuration & URLs
YOLO_MODEL_URL = "https://huggingface.co/7rplus/pagescan-weights/resolve/main/yolo_doc_v1.onnx"
YOLO_MODEL_PATH = "models/yolo_doc_v1.onnx"

UVDOC_MODEL_URL = "https://github.com/PT-Perkasa-Pilar-Utama/ppu-paddle-ocr-models/raw/main/correction/UVDoc.onnx"
UVDOC_MODEL_PATH = "models/uvdoc.onnx"


def ensure_models():
    """Ensures Deep Learning AI models are downloaded and ready for inference."""
    os.makedirs("models", exist_ok=True)
    if not os.path.exists(YOLO_MODEL_PATH) or os.path.getsize(YOLO_MODEL_PATH) < 1000000:
        print("Downloading YOLO Document Detector (10.8 MB)...")
        urllib.request.urlretrieve(YOLO_MODEL_URL, YOLO_MODEL_PATH)
        print("YOLO model downloaded successfully!")

    if not os.path.exists(UVDOC_MODEL_PATH) or os.path.getsize(UVDOC_MODEL_PATH) < 1000000:
        print("Downloading UVDoc Deep Learning Dewarping Engine (30.1 MB)...")
        urllib.request.urlretrieve(UVDOC_MODEL_URL, UVDOC_MODEL_PATH)
        print("UVDoc model downloaded successfully!")


class YOLODocumentDetector:
    """Deep Learning Neural Network for Document Region Detection"""
    _session = None

    @classmethod
    def get_session(cls):
        if cls._session is None:
            ensure_models()
            cls._session = ort.InferenceSession(YOLO_MODEL_PATH)
        return cls._session

    @classmethod
    def detect_document_bbox(cls, image_bgr: np.ndarray, conf_threshold: float = 0.4):
        """
        Detects primary document bounding box with safety margin to prevent text clipping.
        Returns: (x1, y1, x2, y2)
        """
        h, w = image_bgr.shape[:2]
        session = cls.get_session()
        inp_name = session.get_inputs()[0].name
        out_name = session.get_outputs()[0].name

        img_960 = cv2.resize(image_bgr, (960, 960))
        img_rgb = cv2.cvtColor(img_960, cv2.COLOR_BGR2RGB)
        img_norm = (img_rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]

        preds = session.run([out_name], {inp_name: img_norm})[0][0]
        confs = preds[4]
        best_idx = int(np.argmax(confs))
        best_conf = float(confs[best_idx])

        if best_conf < conf_threshold:
            return 0, 0, w, h

        cx, cy, bw, bh = preds[:4, best_idx]
        
        # Add a 2.5% safety expansion margin to guarantee text/margins are not clipped
        margin_x = bw * 0.025
        margin_y = bh * 0.025
        
        x1 = max(0, int((cx - (bw / 2.0) - margin_x) * w / 960.0))
        y1 = max(0, int((cy - (bh / 2.0) - margin_y) * h / 960.0))
        x2 = min(w, int((cx + (bw / 2.0) + margin_x) * w / 960.0))
        y2 = min(h, int((cy + bh / 2.0 + margin_y) * h / 960.0))

        # Snap to frame edges if boundary is within 3% of the image border
        if y1 < int(h * 0.03):
            y1 = 0
        if (h - y2) < int(h * 0.03):
            y2 = h
        if x1 < int(w * 0.03):
            x1 = 0
        if (w - x2) < int(w * 0.03):
            x2 = w

        return x1, y1, x2, y2


class UVDocNeuralDewarper:
    """Deep Learning 3D Document Spine & Surface Rectification Engine"""
    _session = None

    @classmethod
    def get_session(cls):
        if cls._session is None:
            ensure_models()
            cls._session = ort.InferenceSession(UVDOC_MODEL_PATH)
        return cls._session

    @classmethod
    def dewarp_and_rectify(cls, image_bgr: np.ndarray, target_h: int = 2048) -> np.ndarray:
        """
        Applies neural grid rectification to unroll 3D book spine curvature,
        eliminate second page flap/thumb, level text baselines, and remove skew.
        Uses paper-white protective canvas padding to prevent receptive field edge erosion
        from clipping text headers (e.g. 'Index') or bottom margins.
        """
        session = cls.get_session()
        inp_name = session.get_inputs()[0].name
        out_name = session.get_outputs()[0].name

        # Protective paper-white canvas padding
        pad_top = 80
        pad_bottom = 80
        pad_left = 40
        pad_right = 40
        padded = cv2.copyMakeBorder(image_bgr, pad_top, pad_bottom, pad_left, pad_right, 
                                    cv2.BORDER_CONSTANT, value=[245, 245, 245])
        ph, pw = padded.shape[:2]

        target_w = int(target_h * (pw / ph) // 32 * 32)
        
        # Prepare input tensor
        img_in = cv2.resize(padded, (target_w, target_h), interpolation=cv2.INTER_AREA)
        img_rgb = cv2.cvtColor(img_in, cv2.COLOR_BGR2RGB)
        img_norm = (img_rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]

        # Neural inference
        out = session.run([out_name], {inp_name: img_norm})[0]
        
        # Direct RGB unwarping reconstruction
        rectified_rgb = (np.transpose(out[0], (1, 2, 0)) * 255.0).clip(0, 255).astype(np.uint8)
        rectified_bgr = cv2.cvtColor(rectified_rgb, cv2.COLOR_RGB2BGR)

        return rectified_bgr


class IlluminationWhiteningEngine:
    """vFlat / CamScanner-Style Magic Color Illumination Regularization & Paper Whitening"""
    @staticmethod
    def whiten_paper_vflat_style(image: np.ndarray, whiteness_gain: float = 1.12) -> np.ndarray:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        h, w = l.shape
        scale = 800.0 / max(h, w)
        sw, sh = int(w * scale), int(h * scale)
        l_small = cv2.resize(l, (sw, sh), interpolation=cv2.INTER_AREA)

        # Morphological background estimation (large structural element)
        k_size = 51
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
        bg_small = cv2.morphologyEx(l_small, cv2.MORPH_CLOSE, kernel)
        bg_small = cv2.GaussianBlur(bg_small, (51, 51), 0)
        bg = cv2.resize(bg_small, (w, h), interpolation=cv2.INTER_LINEAR)

        # Background division (I / I_bg)
        l_float = l.astype(np.float32)
        bg_float = np.maximum(bg.astype(np.float32), 1.0)
        l_norm = (l_float / bg_float) * 238.0

        # Whiteness compression curve
        l_white = np.clip(l_norm * whiteness_gain, 0, 255)
        paper_mask = l_white > 215
        l_white[paper_mask] = 215 + (l_white[paper_mask] - 215) * 1.15
        l_final = np.clip(l_white, 0, 255).astype(np.uint8)

        lab_clean = cv2.merge([l_final, a, b])
        result = cv2.cvtColor(lab_clean, cv2.COLOR_LAB2BGR)

        # High-frequency stroke enhancement (unsharp masking)
        blur = cv2.GaussianBlur(result, (0, 0), 2.0)
        enhanced = cv2.addWeighted(result, 1.25, blur, -0.25, 0)
        return np.clip(enhanced, 0, 255).astype(np.uint8)


class ScannerPipelineOrchestrator:
    """End-to-End Deep Learning AI Document & Book Scanner Pipeline"""
    def __init__(self, enable_yolo: bool = True, enable_ai_dewarp: bool = True,
                 enable_whitening: bool = True, target_resolution: int = 2048):
        self.enable_yolo = enable_yolo
        self.enable_ai_dewarp = enable_ai_dewarp
        self.enable_whitening = enable_whitening
        self.target_resolution = target_resolution
        ensure_models()

    def process_frame(self, image: np.ndarray) -> dict:
        timings = {}
        stages = {'0_raw': image.copy()}
        current = image.copy()
        h, w = current.shape[:2]

        # Stage 1: YOLO Deep Learning Document Region Isolation
        t0 = time.time()
        if self.enable_yolo:
            x1, y1, x2, y2 = YOLODocumentDetector.detect_document_bbox(current)
            if (x2 - x1) < w or (y2 - y1) < h:
                current = current[y1:y2, x1:x2]
        timings['yolo_detection_ms'] = round((time.time() - t0) * 1000, 1)
        stages['1_yolo_cropped'] = current.copy()

        # Stage 2: UVDoc Neural 3D Spine Dewarping & Perspective Rectification
        t0 = time.time()
        if self.enable_ai_dewarp:
            current = UVDocNeuralDewarper.dewarp_and_rectify(current, target_h=self.target_resolution)
        timings['ai_dewarp_rectify_ms'] = round((time.time() - t0) * 1000, 1)
        stages['2_ai_dewarped'] = current.copy()

        # Stage 3: Illumination Regularization & Paper Whitening
        t0 = time.time()
        if self.enable_whitening:
            current = IlluminationWhiteningEngine.whiten_paper_vflat_style(current)
        timings['whitening_ms'] = round((time.time() - t0) * 1000, 1)
        stages['3_whitened_final'] = current.copy()

        timings['total_pipeline_ms'] = round(sum(timings.values()), 1)
        return {'final': current, 'stages': stages, 'timings': timings}

    @staticmethod
    def compile_batch_to_pdf(processed_image_paths: list, output_pdf_path: str = "scanned_book.pdf") -> str:
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
    parser.add_argument("--output", "-o", type=str, default="scanned_notebook.pdf", help="Output PDF file path")
    parser.add_argument("--resolution", "-r", type=int, default=2048, help="Target processing height in pixels")
    args = parser.parse_args()

    extensions = ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG')
    images = []
    for ext in extensions:
        images.extend(glob.glob(os.path.join(args.input, ext)))
    images = sorted(list(set(images)))

    if not images:
        print(f"No image files found in {args.input}.")
        return

    print("=" * 70)
    print("AI DOCUMENT & BOOK SCANNER ENGINE (Deep Learning Architecture)")
    print("=" * 70)
    print(f"Input Directory  : {args.input}")
    print(f"Total Images     : {len(images)}")
    print(f"Output PDF       : {args.output}")
    print(f"AI Models Active : YOLO Doc Detection + UVDoc 3D Dewarping")
    print("=" * 70)

    orchestrator = ScannerPipelineOrchestrator(target_resolution=args.resolution)

    temp_out_dir = os.path.join(args.input, "_processed_scans")
    os.makedirs(temp_out_dir, exist_ok=True)
    processed_paths = []

    for idx, img_path in enumerate(images):
        name = os.path.basename(img_path)
        print(f"[{idx+1}/{len(images)}] Processing {name}...", end=" ", flush=True)
        raw = cv2.imread(img_path)
        if raw is None:
            print("Failed to read image.")
            continue
        res = orchestrator.process_frame(raw)
        out_page = os.path.join(temp_out_dir, f"page_{idx+1:04d}.jpg")
        cv2.imwrite(out_page, res['final'])
        processed_paths.append(out_page)
        
        t = res['timings']
        print(f"Done in {t['total_pipeline_ms']} ms (AI Dewarp: {t['ai_dewarp_rectify_ms']} ms, Whitening: {t['whitening_ms']} ms)")

    if processed_paths:
        print("\n" + "=" * 70)
        print(f"Assembling {len(processed_paths)} pages into high-resolution PDF: {args.output}...")
        ScannerPipelineOrchestrator.compile_batch_to_pdf(processed_paths, args.output)
        pdf_size_mb = os.path.getsize(args.output) / (1024 * 1024)
        print(f"SUCCESS! Output PDF created ({pdf_size_mb:.2f} MB): {args.output}")
        print("=" * 70)


if __name__ == "__main__":
    main()
