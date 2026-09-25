"""
Next-Gen AI Document & Book Scanner Engine - Core Pipeline
Implements:
1. Auto-Orientation & Fine Deskew
2. 4-Corner Document Boundary Detection & Perspective Rectification (Homography)
3. Non-Rigid 3D Surface Dewarping
4. Finger & Thumb Margin Inpainting
5. CamScanner/vFlat-Style Illumination Regularization & Paper Whitening
6. Multi-Page PDF Assembly
"""

import os
import time
import glob
import cv2
import numpy as np
import img2pdf


def order_points(pts: np.ndarray) -> np.ndarray:
    """Orders 4 points as: top-left, top-right, bottom-right, bottom-left"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]      # top-left
    rect[2] = pts[np.argmax(s)]      # bottom-right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]   # top-right
    rect[3] = pts[np.argmax(diff)]   # bottom-left
    return rect


class DocumentPreprocessingEngine:
    """Engine 1: Auto-Orientation & Deskew Normalization"""
    @staticmethod
    def detect_and_fix_orientation(image: np.ndarray) -> np.ndarray:
        """Detect 0, 90, 180, 270 degree rotation and rotate upright."""
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

    @staticmethod
    def deskew(image: np.ndarray, max_angle: float = 15.0) -> np.ndarray:
        """Correct subtle skew angles within +/- 15 degrees."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)
        if lines is None:
            return image

        lines = lines.reshape(-1, 4)
        angles = []
        for x1, y1, x2, y2 in lines:
            theta = np.degrees(np.arctan2(float(y2 - y1), float(x2 - x1)))
            if abs(theta) <= max_angle:
                angles.append(theta)
            elif abs(abs(theta) - 90) <= max_angle:
                angles.append(theta - 90 if theta > 0 else theta + 90)

        if not angles:
            return image

        median_angle = float(np.median(angles))
        if abs(median_angle) < 0.2:
            return image

        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        rot_mat = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        return cv2.warpAffine(image, rot_mat, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


class DocumentGeometryEngine:
    """Engine 2 & 3: 4-Corner Quad Perspective Rectification & 3D Dewarping"""
    @staticmethod
    def detect_and_warp_quad(image: np.ndarray) -> np.ndarray:
        """
        Detects 4 outer page corners using multi-scale bilateral contour analysis.
        Warps perspective to an un-slanted, flat rectangular page.
        """
        orig = image.copy()
        h, w = image.shape[:2]

        scale = 1000.0 / max(h, w)
        small_h, small_w = int(h * scale), int(w * scale)
        small = cv2.resize(image, (small_w, small_h), interpolation=cv2.INTER_AREA)

        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        blurred = cv2.bilateralFilter(gray, 9, 75, 75)
        edges = cv2.Canny(blurred, 30, 120)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=3)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_quad = None
        max_area = 0
        total_area = small_w * small_h

        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
        for c in sorted_contours:
            area = cv2.contourArea(c)
            if area < total_area * 0.40:
                continue

            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)

            if len(approx) == 4 and area > max_area:
                best_quad = approx.reshape(4, 2)
                max_area = area
                break

        if best_quad is None and sorted_contours:
            largest = sorted_contours[0]
            if cv2.contourArea(largest) > total_area * 0.40:
                hull = cv2.convexHull(largest)
                peri = cv2.arcLength(hull, True)
                approx = cv2.approxPolyDP(hull, 0.03 * peri, True)
                if len(approx) == 4:
                    best_quad = approx.reshape(4, 2)

        if best_quad is not None:
            pts = best_quad / scale
            rect = order_points(pts)
            (tl, tr, br, bl) = rect

            width_a = np.linalg.norm(br - bl)
            width_b = np.linalg.norm(tr - tl)
            max_w = max(int(width_a), int(width_b))

            height_a = np.linalg.norm(tr - br)
            height_b = np.linalg.norm(tl - bl)
            max_h = max(int(height_a), int(height_b))

            dst = np.array([
                [0, 0],
                [max_w - 1, 0],
                [max_w - 1, max_h - 1],
                [0, max_h - 1]
            ], dtype="float32")

            M = cv2.getPerspectiveTransform(rect, dst)
            return cv2.warpPerspective(orig, M, (max_w, max_h), flags=cv2.INTER_CUBIC)

        return orig

    @staticmethod
    def dewarp_3d_surface(image: np.ndarray, strength: float = 0.0) -> np.ndarray:
        """
        Adaptive 3D curvature unrolling. Strength 0.0 preserves planar lines.
        """
        if strength <= 0.0:
            return image

        h, w = image.shape[:2]
        grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))

        curve_profile = np.sin(np.linspace(0, np.pi, h))[:, None]
        horizontal_decay = np.exp(-((grid_x - (w * 0.15)) / (w * 0.25)) ** 2)
        displacement_x = curve_profile * horizontal_decay * (w * strength)

        map_x = np.clip(grid_x - displacement_x, 0, w - 1).astype(np.float32)
        map_y = grid_y.astype(np.float32)

        return cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


class OcclusionRemovalEngine:
    """Engine 4: Finger / Thumb Segmentation & Clean Margin Inpainting"""
    @staticmethod
    def detect_finger_mask(image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        border_mask = np.zeros((h, w), dtype=np.uint8)
        border_w = int(w * 0.10)
        border_h = int(h * 0.10)
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
    """Engine 5: Paper Whitening & Spine Shadow Elimination (vFlat / CamScanner Magic Color)"""
    @staticmethod
    def whiten_paper_vflat_style(image: np.ndarray, whiteness_gain: float = 1.15) -> np.ndarray:
        """
        1. Estimates smooth background illumination in LAB luminance space.
        2. Neutralizes shadows and maps paper to clean white.
        3. Preserves blue ink, red margin lines, and pen strokes with high contrast.
        """
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        h, w = l.shape
        scale = 800.0 / max(h, w)
        sw, sh = int(w * scale), int(h * scale)
        l_small = cv2.resize(l, (sw, sh), interpolation=cv2.INTER_AREA)

        k_size = 51
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
        bg_small = cv2.morphologyEx(l_small, cv2.MORPH_CLOSE, kernel)
        bg_small = cv2.GaussianBlur(bg_small, (51, 51), 0)

        bg = cv2.resize(bg_small, (w, h), interpolation=cv2.INTER_CUBIC)

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
    """End-to-End Batch Orchestrator and PDF Assembly"""
    def __init__(self, enable_orientation: bool = True, enable_deskew: bool = True,
                 enable_crop: bool = True, enable_dewarp: bool = False,
                 enable_finger_removal: bool = True, enable_whitening: bool = True):
        self.enable_orientation = enable_orientation
        self.enable_deskew = enable_deskew
        self.enable_crop = enable_crop
        self.enable_dewarp = enable_dewarp
        self.enable_finger_removal = enable_finger_removal
        self.enable_whitening = enable_whitening

    def process_frame(self, image: np.ndarray) -> dict:
        timings = {}
        stages = {'0_raw': image.copy()}
        current = image.copy()

        # 1. Orientation & Deskew
        t0 = time.time()
        if self.enable_orientation:
            current = DocumentPreprocessingEngine.detect_and_fix_orientation(current)
        if self.enable_deskew:
            current = DocumentPreprocessingEngine.deskew(current)
        timings['orientation_deskew_ms'] = round((time.time() - t0) * 1000, 1)
        stages['1_oriented'] = current.copy()

        # 2. 4-Corner Quad Perspective Warp & Isolation
        t0 = time.time()
        if self.enable_crop:
            current = DocumentGeometryEngine.detect_and_warp_quad(current)
        timings['crop_perspective_ms'] = round((time.time() - t0) * 1000, 1)
        stages['2_cropped_quad'] = current.copy()

        # 3. 3D Dewarp (Adaptive)
        t0 = time.time()
        if self.enable_dewarp:
            current = DocumentGeometryEngine.dewarp_3d_surface(current, strength=0.02)
        timings['dewarp_3d_ms'] = round((time.time() - t0) * 1000, 1)
        stages['3_dewarped'] = current.copy()

        # 4. Finger Removal
        t0 = time.time()
        if self.enable_finger_removal:
            mask = OcclusionRemovalEngine.detect_finger_mask(current)
            stages['finger_mask'] = mask.copy()
            current = OcclusionRemovalEngine.inpaint_fingers(current, mask)
        timings['finger_removal_ms'] = round((time.time() - t0) * 1000, 1)
        stages['4_inpainted'] = current.copy()

        # 5. Paper Whitening & Illumination Regularization
        t0 = time.time()
        if self.enable_whitening:
            current = IlluminationRegularizationEngine.whiten_paper_vflat_style(current)
        timings['whitening_ms'] = round((time.time() - t0) * 1000, 1)
        stages['5_whitened_final'] = current.copy()

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
    parser = argparse.ArgumentParser(description="AI Book & Document Scanner Engine")
    parser.add_argument("--input", "-i", type=str, default="input_images", help="Input directory of image photos")
    parser.add_argument("--output", "-o", type=str, default="scanned_book.pdf", help="Output PDF file path")
    parser.add_argument("--dewarp", action="store_true", help="Enable 3D spine dewarping")
    args = parser.parse_args()

    extensions = ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG')
    images = []
    for ext in extensions:
        images.extend(glob.glob(os.path.join(args.input, ext)))
    images = sorted(list(set(images)))

    if not images:
        print(f"No image files found in {args.input}.")
        return

    print(f"Found {len(images)} images in {args.input}. Processing...")
    orchestrator = ScannerPipelineOrchestrator(enable_dewarp=args.dewarp)

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
