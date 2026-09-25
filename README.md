# 📖 AI Document & Book Scanner Engine

A high-performance scanning pipeline designed to transform raw photographs of flat sheets, curved notebooks, and bound book spines into clean, professionally flattened, digitized multi-page PDF documents.

---

## 🚀 How to Test Your Own Images

You have two convenient ways to test with your own photos:

### Option 1: On Google Colab (Recommended — GPU Accelerated)

1. Open **[Google Colab](https://colab.research.google.com/)** and click **Upload** ➔ Select [`book_scanner_pipeline.ipynb`](file:///f:/Projects/Test/scanner/book_scanner_pipeline.ipynb).
2. Set Runtime to GPU: **Runtime ➔ Change runtime type ➔ T4 GPU**.
3. Run Step 1 through Step 4 (or simply click **Runtime ➔ Run all**).
4. Run **Step 5 ("Upload Your Own Images & Generate PDF")**:
   * An interactive **"Choose Files"** button will appear.
   * Select your book photos from your computer.
   * The notebook processes each photo, displays side-by-side before/after previews, and automatically downloads the compiled **`My_Scanned_Book.pdf`** to your computer!
5. Or run **Step 6 ("Gradio Web App")** for an interactive playground with sliders for paper whiteness and dewarping toggles.

---

### Option 2: Running Locally from Terminal

If you have installed dependencies (`pip install opencv-python-headless img2pdf pytesseract`):

1. Drop your raw photos into the [`input_images/`](file:///f:/Projects/Test/scanner/input_images/) folder.
2. Run the command:
   ```bash
   python pipeline.py --input input_images --output my_book.pdf
   ```
3. The engine processes all images sequentially and compiles them into `my_book.pdf`.

---

## 🛠️ Pipeline Architecture

```
[ Raw Camera Photograph ]
          │
          ▼
1. Auto-Orientation & Deskew      ──► Detects 0°/90°/180°/270° & corrects subtle angular skew
          │
          ▼
2. Boundary Crop & Margins        ──► Isolates page boundaries with safety padding
          │
          ▼
3. 3D Spine Surface Dewarping     ──► Flattens acute curvature along the book spine
          │
          ▼
4. Finger & Thumb Inpainting      ──► Detects holding thumbs and reconstructs page texture
          │
          ▼
5. Illumination & Paper Whitening ──► Eliminates spine gradient shadows & normalizes background to paper-white
          │
          ▼
6. Multi-Page PDF Assembly        ──► Assembles all pages into high-resolution standardized PDF (300 DPI)
```

---

## 📦 Workspace File Structure

* [`prd.md`](file:///f:/Projects/Test/scanner/prd.md) — Product Requirements Document.
* [`book_scanner_pipeline.ipynb`](file:///f:/Projects/Test/scanner/book_scanner_pipeline.ipynb) — Ready-to-run Google Colab notebook with direct file upload widget and Gradio UI.
* [`pipeline.py`](file:///f:/Projects/Test/scanner/pipeline.py) — Standalone Python engine script with CLI batch processing.
* [`input_images/`](file:///f:/Projects/Test/scanner/input_images/) — Local directory for dropping test photos.
