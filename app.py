# app.py (continuous camera capture + upload)
import streamlit as st
import sqlite3, json, io, csv, inspect
from omr import (
    load_image_bytes,
    warp_document,
    detect_bubbles_and_answers,
    score_answers
)
from utils import is_blurry, brightness_score, estimate_skew_angle
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
import time

DB_PATH = "omr_system.db"

# ---------------- DB HELPERS ----------------
def get_exam_ids_from_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS answer_keys (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            exam_id TEXT,
            set_name TEXT,
            answers_json TEXT,
            uploaded_on TEXT
        )
    """)
    cur.execute("SELECT DISTINCT exam_id FROM answer_keys")
    rows = [r[0] for r in cur.fetchall()]
    conn.close()
    return rows


def load_keys_from_db(exam_id):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT set_name, answers_json FROM answer_keys WHERE exam_id=?", (exam_id,))
    rows = cur.fetchall()
    conn.close()
    return {set_name: json.loads(ans_json) for set_name, ans_json in rows}

# ---------------- OVERLAY ANNOTATOR (unchanged) ----------------
def annotate_overlay_with_scores(warped, scores, show_correct=False, debug=False, offset_x=-30, offset_y=0):
    overlay = Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(overlay)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except Exception:
        font = ImageFont.load_default()

    H, W = warped.shape[:2]
    margin_x = 120
    margin_y = 200
    footer_gap = 200
    grid_w = W - 2 * margin_x
    grid_h = H - margin_y - footer_gap
    rows, blocks, choices = 20, 5, 4
    cell_h = grid_h / rows
    block_w = grid_w / blocks

    for b in range(blocks):
        block_left = int(margin_x + b * block_w)
        side_x = max(8, block_left - 36 + offset_x)
        for r in range(rows):
            qnum = b * 20 + r + 1
            qstr = str(qnum)
            detail = scores.get("details", {}).get(qstr, {})
            chosen = detail.get("chosen", "")
            ambiguous = detail.get("ambiguous", False)
            is_correct = detail.get("is_correct", False)
            centers = detail.get("centers", [])

            row_top = margin_y + r * cell_h
            marker_y = int(row_top + cell_h * 0.5 + offset_y)

            if chosen:
                if ambiguous:
                    color, text = (255, 165, 0), "?"
                else:
                    color = (0, 200, 0) if is_correct else (220, 40, 40)
                    text = chosen
                bg_r = 14
                draw.ellipse(
                    [(side_x - bg_r, marker_y - bg_r), (side_x + bg_r, marker_y + bg_r)],
                    fill=(255, 255, 255), outline=color, width=2
                )
                try:
                    bbox = draw.textbbox((0, 0), text, font=font)
                    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                except Exception:
                    try:
                        tw, th = font.getsize(text)
                    except Exception:
                        tw, th = (12, 12)
                draw.text((side_x - tw // 2, marker_y - th // 2), text, fill=color, font=font)
            else:
                dot_r = 3
                draw.ellipse(
                    [(side_x - dot_r, marker_y - dot_r), (side_x + dot_r, marker_y + dot_r)],
                    outline=(200, 200, 200), width=1
                )

            if show_correct and centers and chosen:
                try:
                    idx = ['A', 'B', 'C', 'D'].index(chosen)
                except:
                    idx = None
                if idx is not None and idx < len(centers):
                    c = centers[idx]
                    cx, cy, rad = c['x'], c['y'], c['r']
                    bubble_color = (0, 200, 0) if is_correct else (255, 165, 0) if ambiguous else (220, 40, 40)
                    draw.ellipse([(cx - rad, cy - rad), (cx + rad, cy + rad)], outline=bubble_color, width=3)

    return overlay

# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="Evaluator Dashboard", layout="wide")
st.title("Evaluator Dashboard — OMR Evaluation")

# Sidebar tuning controls
st.sidebar.header("Settings / Detection tuning")
blur_thresh = st.sidebar.slider('Blur threshold', 10, 300, 100)
brightness_low = st.sidebar.slider('Min brightness', 30, 220, 70)
conf_threshold = st.sidebar.slider('Confidence threshold', 1, 100, 25) / 100.0
fill_thresh = st.sidebar.slider('Fill threshold', 0.05, 0.40, 0.18, step=0.01)
ambig_low = st.sidebar.slider('Ambiguous lower bound', 0.01, 0.30, 0.08, step=0.01)
adaptive_bs = st.sidebar.slider('Adaptive blocksize', 11, 51, 25, step=2)
adaptive_C = st.sidebar.slider('Adaptive C', -10, 20, 10)
min_radius_ratio = st.sidebar.slider('Min radius ratio', 0.004, 0.02, 0.009, step=0.001)
max_radius_ratio = st.sidebar.slider('Max radius ratio', 0.02, 0.06, 0.035, step=0.001)
search_radius = st.sidebar.slider('Search radius (px)', 8, 48, 18)
morph_iter = st.sidebar.slider('Morph iterations', 0, 4, 2)
cc_area_thresh = st.sidebar.slider('Connected-component area thresh', 0.02, 0.5, 0.12, step=0.01)
off_x = st.sidebar.slider("Marker horizontal offset (per-block)", -60, 60, -30, step=1)
off_y = st.sidebar.slider("Marker vertical offset (px)", -80, 120, 0, step=1)
show_correct = st.sidebar.checkbox("Outline chosen bubble (green=correct/red=wrong/orange=ambiguous)", value=False)
debug_mode = st.sidebar.checkbox("Detect debug mode (draw debug info)", value=False)

# Load exam ids
exam_ids = get_exam_ids_from_db()
if not exam_ids:
    st.warning("⚠️ No answer keys found in the database. Please upload keys in the Admin Dashboard first.")
    st.stop()

exam_id = st.selectbox("Select Exam/Week", exam_ids)
keys = load_keys_from_db(exam_id)
if not keys:
    st.error(f"No sets found for {exam_id}. Please check Admin Dashboard.")
    st.stop()
set_choice = st.selectbox("Select Set", options=list(keys.keys()))
key = keys[set_choice]

# ---------------- Upload or Camera option ----------------
mode = st.radio("Choose Input Mode", ["📂 Upload from files", "📸 Continuous camera capture"])

# storage for camera captures across reruns
if "camera_batch" not in st.session_state:
    st.session_state.camera_batch = []  # list of dicts {name, bytes}
if "process_now" not in st.session_state:
    st.session_state.process_now = False

uploaded_files = []
if mode == "📂 Upload from files":
    uploaded_files = st.file_uploader("Upload OMR sheets (multiple allowed)", type=["png","jpg","jpeg"], accept_multiple_files=True)

elif mode == "📸 Continuous camera capture":
    st.markdown("**Camera capture mode** — take one photo at a time and click **Add capture** to append to the batch.")
    cam_photo = st.camera_input("Take a photo of the OMR sheet")
    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        if st.button("Add capture to batch"):
            if cam_photo is None:
                st.warning("No camera photo to add. Take a photo first.")
            else:
                # read bytes and store with timestamp name
                b = cam_photo.read()
                name = f"camera_{len(st.session_state.camera_batch)+1}_{int(time.time())}.jpg"
                st.session_state.camera_batch.append({"name": name, "bytes": b})
                st.success(f"Added capture ({name}) — {len(st.session_state.camera_batch)} in batch.")
    with col2:
        if st.button("Clear batch"):
            st.session_state.camera_batch = []
            st.info("Cleared camera batch.")
    with col3:
        if st.button("Process all captures now"):
            # trigger processing below
            st.session_state.process_now = True

    # show thumbnails / list with remove option
    if st.session_state.camera_batch:
        st.write(f"Batch contains {len(st.session_state.camera_batch)} captures:")
        thumbs = st.session_state.camera_batch
        cols = st.columns(min(4, len(thumbs)))
        for i, thumb in enumerate(thumbs):
            c = cols[i % len(cols)]
            with c:
                st.image(thumb["bytes"], width=150)
                if st.button(f"Remove #{i+1}", key=f"rm_{i}"):
                    st.session_state.camera_batch.pop(i)
                    st.experimental_rerun()
        # when processing, build uploaded_files from session_state only if user pressed Process
        if st.session_state.process_now:
            uploaded_files = []
            for t in st.session_state.camera_batch:
                # create an UploadedFile-like object with read() & name attr used later
                class _SimpleFile:
                    def __init__(self, name, b):
                        self.name = name
                        self._b = b
                    def read(self):
                        return self._b
                uploaded_files.append(_SimpleFile(t["name"], t["bytes"]))
            # reset flag so we don't reprocess on every rerun automatically
            st.session_state.process_now = False

# ---------------- Processing ----------------
all_scores = []
if uploaded_files:
    for uploaded in uploaded_files:
        display_name = getattr(uploaded, "name", "camera_capture")
        st.subheader(f"📄 Evaluating: {display_name}")
        img_bytes = uploaded.read()
        img = load_image_bytes(img_bytes)
        if img is None:
            st.error(f"Unable to read {display_name}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurry, var = is_blurry(gray, threshold=blur_thresh)
        bright = brightness_score(gray)
        skew = estimate_skew_angle(gray)

        st.write(f"Blur variance: {var:.2f} → {'BLURRY' if blurry else 'OK'}")
        st.write(f"Brightness: {bright:.1f} → {'LOW' if bright < brightness_low else 'OK'}")
        st.write(f"Skew angle: {skew:.1f}° → {'High skew' if abs(skew) > 15 else 'OK'}")

        warped, _ = warp_document(img)

        params = {
            "adaptive_blocksize": adaptive_bs,
            "adaptive_C": adaptive_C,
            "fill_thresh": fill_thresh,
            "ambig_low": ambig_low,
            "mean_thresh": 140,
            "min_radius_ratio": min_radius_ratio,
            "max_radius_ratio": max_radius_ratio,
            "search_radius": int(search_radius),
            "morph_iter": morph_iter,
            "cc_area_thresh": cc_area_thresh
        }

        # Try calling detect_bubbles_and_answers with params if supported, else without params
        extracted = None
        base_overlay = None
        try:
            # prefer signature with params if available
            sig = inspect.signature(detect_bubbles_and_answers)
            if "params" in sig.parameters:
                extracted, base_overlay = detect_bubbles_and_answers(warped, debug=debug_mode, params=params)
            else:
                extracted, base_overlay = detect_bubbles_and_answers(warped, debug=debug_mode)
        except TypeError:
            # fallback to no-params call
            extracted, base_overlay = detect_bubbles_and_answers(warped, debug=debug_mode)
        except Exception as e:
            st.error(f"Detection error: {e}")
            # show fallback overlay if available
            if base_overlay is not None:
                try:
                    buf = io.BytesIO(); base_overlay.save(buf, format="PNG"); st.image(buf.getvalue(), use_container_width=True)
                except Exception:
                    pass
            continue

        if not extracted:
            st.error(f"❌ Detection failed for {display_name}")
            if base_overlay is not None:
                try:
                    buf = io.BytesIO(); base_overlay.save(buf, format="PNG"); st.image(buf.getvalue(), use_container_width=True)
                except Exception:
                    pass
            continue

        # Score answers
        scores = score_answers(extracted, key)
        all_scores.append((display_name, scores))

        st.write(f"✅ Total: {scores['total']} / 100")
        for i, val in enumerate(scores['per_subject']):
            st.write(f"Subject {i+1}: {val}/20")

        # Annotate and show overlay (robust handling)
        annotated = annotate_overlay_with_scores(warped, scores, show_correct=show_correct, debug=debug_mode, offset_x=off_x, offset_y=off_y)
        try:
            # accept PIL Image, numpy arrays, or OpenCV images
            if annotated is None:
                raise RuntimeError("annotated image is None")
            if isinstance(annotated, np.ndarray):
                # if BGR (OpenCV) convert to RGB
                if annotated.ndim == 3 and annotated.shape[2] == 3:
                    annotated_pil = Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
                else:
                    annotated_pil = Image.fromarray(annotated)
            elif isinstance(annotated, Image.Image):
                annotated_pil = annotated
            else:
                # last resort: try to convert
                annotated_pil = Image.fromarray(np.array(annotated))

            buf = io.BytesIO()
            annotated_pil.save(buf, format="PNG")
            buf.seek(0)
            img_bytes = buf.read()
            if not img_bytes:
                raise RuntimeError("Annotated image buffer is empty")
            st.image(img_bytes, caption=f"Detected Marks for {display_name}", use_container_width=True)
        except Exception as e:
            st.error("Failed to render annotated image — see logs for details.")
            st.exception(e)

    # Export all results into one CSV
    def to_csv(all_scores):
        output = io.StringIO()
        w = csv.writer(output)
        w.writerow(["file","question","chosen","correct","confidence","is_correct","ambiguous"])
        for fname, scores in all_scores:
            for q, d in sorted(scores["details"].items(), key=lambda x:int(x[0])):
                w.writerow([
                    fname,
                    q,
                    d.get("chosen",""),
                    d.get("correct_set",""),
                    d.get("confidence",0),
                    d.get("is_correct", False),
                    d.get("ambiguous", False)
                ])
        return output.getvalue().encode("utf-8")

    st.download_button("📥 Download All Results (CSV)", data=to_csv(all_scores),
                    file_name="all_results.csv", mime="text/csv")
