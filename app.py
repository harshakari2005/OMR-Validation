# app.py (Evaluator + Admin; Firestore support + SQLite fallback)
import streamlit as st
import sqlite3, json, io, csv, inspect, datetime, time, os
from omr import (
    load_image_bytes,
    warp_document,
    detect_bubbles_and_answers,
    score_answers
)
from utils import is_blurry, brightness_score, estimate_skew_angle
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
import pandas as pd

# Firestore imports (optional)
firestore_client = None
firebase_available = False
try:
    import firebase_admin
    from firebase_admin import credentials, firestore
    firebase_available = True
except Exception:
    firebase_available = False

DB_PATH = "omr_system.db"

# ---------------- FIREBASE INIT ----------------
def init_firestore():
    global firestore_client, firebase_available
    if not firebase_available:
        return None

    # Try multiple ways to get credentials:
    # 1) Streamlit secrets: st.secrets["firebase"]["service_account"] (string or dict)
    # 2) Path from env FIREBASE_CRED_PATH
    # 3) Local file 'omr-validation-firebase-adminsdk.json' if present
    try:
        if "firebase" in st.secrets and "service_account" in st.secrets.get("firebase", {}):
            # secrets can store JSON string or dict
            svc = st.secrets["firebase"]["service_account"]
            if isinstance(svc, str):
                cred_dict = json.loads(svc)
            else:
                cred_dict = svc
            cred = credentials.Certificate(cred_dict)
            firebase_admin.initialize_app(cred)
            firestore_client = firestore.client()
            return firestore_client
    except Exception as e:
        # continue to next method
        st.experimental_set_query_params(_fb_err=str(e)) if hasattr(st, "experimental_set_query_params") else None

    # env var path
    try:
        env_path = os.environ.get("FIREBASE_CRED_PATH")
        if env_path and os.path.exists(env_path):
            cred = credentials.Certificate(env_path)
            firebase_admin.initialize_app(cred)
            firestore_client = firestore.client()
            return firestore_client
    except Exception:
        pass

    # local file in project root (uploaded file)
    try:
        possible = [
            "/workdir/omr-validation-firebase-adminsdk.json",  # fallback possibilities
            "omr-validation-firebase-adminsdk.json",
            "/mnt/data/omr-validation-firebase-adminsdk-fbsvc-a3c0b2924e.json",
            os.path.join(os.getcwd(), "omr-validation-firebase-adminsdk-fbsvc-a3c0b2924e.json")
        ]
        for p in possible:
            if p and os.path.exists(p):
                cred = credentials.Certificate(p)
                firebase_admin.initialize_app(cred)
                firestore_client = firestore.client()
                return firestore_client
    except Exception:
        pass

    # If none worked:
    return None

# attempt to init once at import
if firebase_available:
    try:
        init_firestore()
    except Exception:
        firestore_client = None

# ---------------- DB HELPERS (Firestore first, SQLite fallback) ----------------
def ensure_tables():
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
    conn.commit()
    conn.close()

# ---------- Firestore wrappers ----------
def firestore_get_exam_ids():
    if firestore_client is None:
        return []
    docs = firestore_client.collection("answer_keys").stream()
    exam_ids = set()
    for d in docs:
        data = d.to_dict()
        if data and "exam_id" in data:
            exam_ids.add(data["exam_id"])
    return sorted(list(exam_ids))

def firestore_load_keys(exam_id):
    if firestore_client is None:
        return {}
    docs = firestore_client.collection("answer_keys").where("exam_id", "==", exam_id).stream()
    out = {}
    for d in docs:
        data = d.to_dict()
        set_name = data.get("set_name")
        answers = data.get("answers_dict") or {}
        if set_name:
            out[set_name] = answers
    return out

def firestore_insert_key(exam_id, set_name, answers_dict):
    if firestore_client is None:
        raise RuntimeError("Firestore not initialized")
    # document id = examid__setname__timestamp for uniqueness
    doc_id = f"{exam_id}__{set_name}__{int(time.time())}"
    firestore_client.collection("answer_keys").document(doc_id).set({
        "exam_id": exam_id,
        "set_name": set_name,
        "answers_dict": answers_dict,
        "uploaded_on": datetime.datetime.utcnow().isoformat()
    })

def firestore_delete_key(exam_id, set_name):
    if firestore_client is None:
        raise RuntimeError("Firestore not initialized")
    # find docs with exam_id and set_name and delete
    docs = firestore_client.collection("answer_keys").where("exam_id", "==", exam_id).where("set_name", "==", set_name).stream()
    for d in docs:
        firestore_client.collection("answer_keys").document(d.id).delete()

# ---------- SQLite wrappers ----------
def get_exam_ids_from_db_sqlite():
    ensure_tables()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT exam_id FROM answer_keys")
    rows = [r[0] for r in cur.fetchall()]
    conn.close()
    return rows

def load_keys_from_db_sqlite(exam_id):
    ensure_tables()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT set_name, answers_json FROM answer_keys WHERE exam_id=?", (exam_id,))
    rows = cur.fetchall()
    conn.close()
    return {set_name: json.loads(ans_json) for set_name, ans_json in rows}

def insert_key_into_db_sqlite(exam_id, set_name, answers_dict):
    ensure_tables()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO answer_keys (exam_id, set_name, answers_json, uploaded_on) VALUES (?,?,?,?)",
        (exam_id, set_name, json.dumps(answers_dict), datetime.datetime.now().isoformat())
    )
    conn.commit()
    conn.close()

def delete_key_from_db_sqlite(exam_id, set_name):
    ensure_tables()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("DELETE FROM answer_keys WHERE exam_id=? AND set_name=?", (exam_id, set_name))
    conn.commit()
    conn.close()

# ---------- Unified helpers that prefer Firestore ----------
def get_exam_ids():
    # prefer Firestore
    if firestore_client:
        try:
            ids = firestore_get_exam_ids()
            if ids:
                return ids
        except Exception:
            pass
    return get_exam_ids_from_db_sqlite()

def load_keys_from_db(exam_id):
    if firestore_client:
        try:
            keys = firestore_load_keys(exam_id)
            if keys:
                return keys
        except Exception:
            pass
    return load_keys_from_db_sqlite(exam_id)

def insert_key_into_db(exam_id, set_name, answers_dict):
    # write to both (Firestore preferred); if firestore fails, still write sqlite
    sqlite_ok = False
    try:
        insert_key_into_db_sqlite(exam_id, set_name, answers_dict)
        sqlite_ok = True
    except Exception:
        sqlite_ok = False

    if firestore_client:
        try:
            firestore_insert_key(exam_id, set_name, answers_dict)
            return True
        except Exception as e:
            # log to Streamlit app, but keep sqlite version
            st.warning(f"Warning: Firestore write failed: {e}")
            return sqlite_ok
    return sqlite_ok

def delete_key_from_db(exam_id, set_name):
    sqlite_ok = False
    try:
        delete_key_from_db_sqlite(exam_id, set_name)
        sqlite_ok = True
    except Exception:
        sqlite_ok = False

    if firestore_client:
        try:
            firestore_delete_key(exam_id, set_name)
            return True
        except Exception as e:
            st.warning(f"Warning: Firestore delete failed: {e}")
            return sqlite_ok
    return sqlite_ok

# ---------------- OVERLAY ANNOTATOR (returns PIL Image) ----------------
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

# ---------------- IMAGE DISPLAY HELPER ----------------
def show_image(image_bytes_or_pil, caption=None):
    try:
        st.image(image_bytes_or_pil, caption=caption, use_container_width=True)
    except TypeError:
        try:
            st.image(image_bytes_or_pil, caption=caption, use_column_width=True)
        except TypeError:
            st.image(image_bytes_or_pil, caption=caption)

# ---------------- AUTH HELPERS ----------------
def is_admin_authenticated():
    return st.session_state.get("admin_authenticated", False)

def admin_login_widget():
    if "admin_authenticated" not in st.session_state:
        st.session_state.admin_authenticated = False

    try:
        admin_pw = st.secrets["auth"]["admin_password"]
    except Exception:
        st.warning("Admin password not configured in Secrets. On Streamlit Cloud set `auth.admin_password`.")
        admin_pw = None

    if is_admin_authenticated():
        cols = st.columns([1, 8])
        with cols[0]:
            if st.button("🔒 Logout (Admin)"):
                st.session_state.admin_authenticated = False
                st.success("Logged out.")
                st.experimental_rerun()
        with cols[1]:
            st.info("You are authenticated as Admin.")
        return True

    pw = st.text_input("🔑 Admin password", type="password")
    if st.button("Login as Admin"):
        if admin_pw is None:
            st.error("Admin password not configured in secrets.")
            return False
        if pw == admin_pw:
            st.session_state.admin_authenticated = True
            st.success("Admin login successful.")
            return True
        else:
            st.error("Wrong password.")
            return False
    return False

# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="OMR — Evaluator & Admin", layout="wide")
st.title("OMR — Evaluator & Admin Dashboard")

# top-level mode selector
mode_top = st.sidebar.radio("App Mode", ["Evaluator", "Admin"])

# Common sliders for evaluator (only when in Evaluator mode)
if mode_top == "Evaluator":
    st.sidebar.header("Detection tuning")
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

# ---------------- Admin Mode ----------------
if mode_top == "Admin":
    st.header("🔑 Admin — Manage Answer Keys")

    # show Firebase status
    if firestore_client:
        st.success("Firestore connected ✅")
    else:
        if firebase_available:
            st.warning("Firestore not initialized (check credentials). Uploaded keys will be saved to local SQLite.")
        else:
            st.info("firebase-admin not installed — using local SQLite only.")

    # login widget
    if not is_admin_authenticated():
        with st.expander("Admin login"):
            admin_login_widget()
        st.stop()

    # authenticated admin flow
    ensure_tables()

    with st.expander("Upload new answer key"):
        exam_id = st.text_input("Exam ID (e.g., Week1, Test2025)", key="adm_exam")
        set_name = st.text_input("Set Name (e.g., SetA, SetB)", key="adm_set")
        uploaded_key = st.file_uploader(
            "Upload Answer Key (CSV or Excel: question,choice) — two columns",
            type=["csv", "xlsx", "xls"],
            key="adm_up"
        )
        if st.button("Save Answer Key"):
            if not exam_id or not set_name:
                st.warning("Please provide Exam ID and Set Name.")
            elif uploaded_key is None:
                st.warning("Upload a CSV or Excel file.")
            else:
                try:
                    if uploaded_key.name.lower().endswith(".csv"):
                        df = pd.read_csv(uploaded_key)
                    else:
                        df = pd.read_excel(uploaded_key)

                    answers = {}
                    for idx, row in df.iterrows():
                        try:
                            q = str(row.iloc[0]).strip()
                            a = str(row.iloc[1]).strip()
                        except Exception:
                            continue
                        if q and a and q.lower() not in ("nan", ""):
                            answers[q] = a

                    if not answers:
                        st.warning("No valid rows parsed from file.")
                    else:
                        ok = insert_key_into_db(exam_id, set_name, answers)
                        if ok:
                            st.success(f"Uploaded key: {exam_id} / {set_name}")
                        else:
                            st.error("Failed to save key to any DB.")
                except Exception as e:
                    st.error(f"Error parsing/saving key: {e}")

    with st.expander("Existing keys"):
        exam_ids = get_exam_ids()
        if not exam_ids:
            st.info("No keys found yet.")
        else:
            sel_exam = st.selectbox("Select exam to view sets", ["-- select --"] + exam_ids)
            if sel_exam and sel_exam != "-- select --":
                keys = load_keys_from_db(sel_exam)
                if not keys:
                    st.info("No sets for this exam.")
                else:
                    for set_name, answers_json in keys.items():
                        with st.container():
                            st.write(f"**{sel_exam} — {set_name}**  — {len(answers_json)} items")
                            cols = st.columns([1,1,1,4])
                            if cols[0].button("View", key=f"view_{sel_exam}_{set_name}"):
                                rows = sorted(
                                    [(int(k) if str(k).isdigit() else k, v) for k,v in answers_json.items()],
                                    key=lambda x: (x[0] if isinstance(x[0], int) else float("inf"))
                                )
                                preview = "\n".join([f"{k},{v}" for k,v in rows[:200]])
                                st.text(preview)
                            if cols[1].button("Download CSV", key=f"dl_{sel_exam}_{set_name}"):
                                csv_text = "question,answer\n" + "\n".join([f"{q},{a}" for q,a in answers_json.items()])
                                st.download_button(f"Download {sel_exam}_{set_name}.csv", data=csv_text.encode("utf-8"),
                                                   file_name=f"{sel_exam}_{set_name}.csv", mime="text/csv")
                            if cols[2].button("Delete", key=f"del_{sel_exam}_{set_name}"):
                                delete_key_from_db(sel_exam, set_name)
                                st.success("Deleted. Refreshing...")
                                st.experimental_rerun()

    st.stop()  # don't run evaluator UI when in admin mode

# ---------------- Evaluator Mode (default flow below) ----------------
exam_ids = get_exam_ids()
if not exam_ids:
    st.warning("⚠️ No answer keys found in the database. Please upload keys in the Admin dashboard first.")
    st.stop()

exam_id = st.selectbox("Select Exam/Week", exam_ids)
keys = load_keys_from_db(exam_id)
if not keys:
    st.error(f"No sets found for {exam_id}. Please check Admin Dashboard.")
    st.stop()
set_choice = st.selectbox("Select Set", options=list(keys.keys()))
key = keys[set_choice]

# (rest of evaluator flow remains same as your working code)
# ... (file upload, camera, detection, scoring, overlay, CSV export)
# I omitted the evaluator internals here for brevity — keep them the same as you had.
