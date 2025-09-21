# omr.py
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Standard warp size (pixels)
WARP_W = 1200
WARP_H = 1600

def load_image_bytes(file_bytes):
    """Load image from uploaded bytes into OpenCV BGR ndarray."""
    arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img

# ---------------- document warp helpers ----------------
def find_document_contour(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    best = None
    for c in contours:
        area = cv2.contourArea(c)
        if area < 10000:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if area > max_area and len(approx) == 4:
            best = approx
            max_area = area
    return best

def order_points(pts):
    rect = np.zeros((4,2), dtype='float32')
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def warp_document(img):
    """
    Hybrid warp:
    - Try to detect a 4-point contour (page border).
    - If found and large enough, use it to perspective-correct.
    - Otherwise fallback to using full image corners (keeps entire sheet).
    """
    doc = find_document_contour(img)
    h, w = img.shape[:2]

    # fraction of image area a contour must cover to be trusted
    area_ratio_threshold = 0.45
    use_fallback = True
    if doc is not None:
        contour_area = cv2.contourArea(doc)
        img_area = float(h * w)
        if contour_area / img_area >= area_ratio_threshold:
            use_fallback = False

    if use_fallback:
        pts = np.array([[0,0],[w,0],[w,h],[0,h]], dtype='float32')
    else:
        pts = doc.reshape(4,2).astype('float32')
        pts = order_points(pts)

    dst = np.array([[0,0],[WARP_W-1,0],[WARP_W-1,WARP_H-1],[0,WARP_H-1]], dtype='float32')
    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(img, M, (WARP_W, WARP_H))
    return warped, M

# ---------------- bubble detection & mapping ----------------
def detect_bubbles_and_answers(warped, debug=False, params=None):
    """
    Detector that accepts 'params' dictionary to tune behavior.

    Returns: answers (dict), overlay (PIL Image)
    answers[q] includes:
      { "choice": "A"/"B"/"C"/"D" or "", "confidence": float, "ambiguous": bool,
        "centers": [ {x,y,r,fill}, ... ] }
    """
    # default parameters
    if params is None:
        params = {}
    adaptive_blocksize = int(params.get("adaptive_blocksize", 25))  # odd
    adaptive_C = int(params.get("adaptive_C", 10))
    fill_thresh = float(params.get("fill_thresh", 0.18))
    ambig_low = float(params.get("ambig_low", 0.08))
    mean_thresh = float(params.get("mean_thresh", 140))
    min_radius_ratio = float(params.get("min_radius_ratio", 0.009))
    max_radius_ratio = float(params.get("max_radius_ratio", 0.035))
    search_radius = int(params.get("search_radius", 18))
    morph_iter = int(params.get("morph_iter", 2))
    cc_area_thresh = float(params.get("cc_area_thresh", 0.12))

    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # adaptive threshold (blocksize must be odd and >=3)
    bs = adaptive_blocksize if adaptive_blocksize % 2 == 1 else adaptive_blocksize + 1
    bs = max(3, bs)
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, bs, adaptive_C)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=morph_iter)
    th = cv2.medianBlur(th, 3)
    thc = th  # closed threshold for fill checks

    # hough circle candidate radii scaled
    minR = max(6, int(min(w,h) * min_radius_ratio))
    maxR = max(10, int(min(w,h) * max_radius_ratio))

    centers = []
    try:
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1.2,
                                   minDist=int(minR*1.5),
                                   param1=60, param2=32,
                                   minRadius=minR, maxRadius=maxR)
        if circles is not None:
            circles = np.round(circles[0]).astype(int)
            for (cx, cy, r) in circles:
                if cy < int(0.10*h) or cy > int(0.96*h):
                    continue
                centers.append((int(cx), int(cy), int(r)))
    except Exception:
        centers = []

    # contour fallback if hough returned too few centers
    if len(centers) < 220:
        centers = []
        contours, _ = cv2.findContours(thc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            if area < 80 or area > 7000:
                continue
            x,y,wc,hc = cv2.boundingRect(c)
            ar = (wc / float(hc)) if hc>0 else 0
            if ar < 0.6 or ar > 1.6:
                continue
            cx = x + wc//2
            cy = y + hc//2
            r = max(wc, hc)//2
            if cy < int(0.10*h) or cy > int(0.96*h):
                continue
            centers.append((int(cx), int(cy), int(r)))

    # deduplicate
    dedup = []
    for (cx,cy,r) in centers:
        found = False
        for (ux,uy,ur) in dedup:
            if abs(ux-cx) <= 6 and abs(uy-cy) <= 6:
                found = True
                break
        if not found:
            dedup.append((cx,cy,r))
    centers = dedup

    # if still too few centers, use fallback grid
    if len(centers) < 160:
        return _fallback_grid_detection(warped, thc, debug=debug, params=params)

    # Build rows using 1D y-binning
    y_coords = np.array([c[1] for c in centers])
    ymin = np.percentile(y_coords, 2)
    ymax = np.percentile(y_coords, 98)
    rows = 20
    row_centers = np.linspace(ymin, ymax, rows)

    centers_arr = np.array(centers)
    yy = centers_arr[:,1][:,None]
    dist = np.abs(yy - row_centers[None,:])
    row_idx = np.argmin(dist, axis=1)

    rows_lists = [[] for _ in range(rows)]
    for i, idx in enumerate(row_idx):
        rows_lists[idx].append(tuple(centers_arr[i].tolist()))

    blocks = 5
    choices = 4
    answers = {}
    overlay = Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(overlay)

    # process each row
    for r in range(rows):
        row_list = rows_lists[r]
        row_sorted = sorted(row_list, key=lambda t: t[0])
        if len(row_sorted) >= blocks * choices:
            row_sorted = row_sorted[:blocks*choices]

        # split into 5 groups by x; if not enough centers, we'll estimate positions
        if len(row_sorted) >= blocks * choices:
            groups = [row_sorted[b*choices:(b+1)*choices] for b in range(blocks)]
        elif len(row_sorted) >= blocks:
            indices = np.linspace(0, len(row_sorted), blocks+1, dtype=int)
            groups = []
            for i0,i1 in zip(indices[:-1], indices[1:]):
                groups.append(row_sorted[i0:i1])
            # ensure we have exactly 'blocks' groups
            while len(groups) < blocks:
                groups.append([])
        else:
            groups = [[] for _ in range(blocks)]

        for b, block_centers in enumerate(groups):
            # compute final question number (block-major ordering)
            final_q = str(b*rows + r + 1)

            # prepare 4 candidate positions for A-D
            centers_for_choices = []
            if len(block_centers) >= choices:
                chosen_chunk = sorted(block_centers, key=lambda t: t[0])[:choices]
                for (cx,cy,rad) in chosen_chunk:
                    centers_for_choices.append({'x':int(cx),'y':int(cy),'r':int(rad)})
            else:
                # estimate using coarse grid positions inside this block
                margin_x = 120
                margin_y = 200
                grid_w = WARP_W - 2*margin_x
                block_w = grid_w / blocks
                block_left = margin_x + b*block_w
                for ci in range(choices):
                    est_x = int(block_left + (ci + 0.5)*(block_w/choices))
                    est_y = int(row_centers[r])
                    est_r = max(8, int(minR*1.2))
                    centers_for_choices.append({'x':est_x,'y':est_y,'r':est_r})

            # evaluate fill for each candidate
            choice_infos = []
            for cidx, cinfo in enumerate(centers_for_choices):
                cx = int(cinfo['x']); cy = int(cinfo['y']); rad = int(cinfo['r'])
                rr = max(8, int(rad*0.9))
                x1 = max(0, cx-rr); y1 = max(0, cy-rr)
                x2 = min(w, cx+rr); y2 = min(h, cy+rr)
                patch = thc[y1:y2, x1:x2]
                if patch.size == 0:
                    fill = 0.0
                else:
                    fill = float(np.count_nonzero(patch)) / float(patch.size)
                choice_infos.append({'x':cx,'y':cy,'r':rad,'fill':fill})

            fills = [ci['fill'] for ci in choice_infos]
            max_idx = int(np.argmax(fills))
            max_val = fills[max_idx]

            marked = False
            ambiguous = False
            if max_val >= fill_thresh:
                marked = True
            elif ambig_low <= max_val < fill_thresh:
                ambiguous = True

            answer_choice = ''
            if marked or ambiguous:
                answer_choice = ['A','B','C','D'][max_idx]

            out_centers = [{'x':ci['x'],'y':ci['y'],'r':ci['r'],'fill':round(ci['fill'],3)} for ci in choice_infos]

            answers[final_q] = {
                'choice': answer_choice,
                'confidence': round(max_val,3),
                'ambiguous': ambiguous,
                'centers': out_centers
            }

            if debug:
                # draw candidate circles and fill labels
                for idx_ch, ci in enumerate(choice_infos):
                    cx,cy,rad = ci['x'],ci['y'],ci['r']
                    fill = ci['fill']
                    col = (0,255,0) if idx_ch==max_idx and marked and not ambiguous else (255,165,0) if idx_ch==max_idx and ambiguous else (200,200,200)
                    draw.ellipse([(cx-rad,cy-rad),(cx+rad,cy+rad)], outline=col, width=2)
                    try:
                        font = ImageFont.truetype("arial.ttf", 12)
                    except Exception:
                        font = ImageFont.load_default()
                    draw.text((cx-rad, cy+rad-10), f"{fill:.2f}", fill=(0,0,0), font=font)
                if answer_choice:
                    try:
                        font = ImageFont.truetype("arial.ttf", 18)
                    except Exception:
                        font = ImageFont.load_default()
                    txt_col = (0,200,0) if marked and not ambiguous else (255,165,0) if ambiguous else (220,40,40)
                    draw.text((out_centers[max_idx]['x']-8, out_centers[max_idx]['y']-out_centers[max_idx]['r']-20),
                              answer_choice, fill=txt_col, font=font)

    return answers, overlay

# ---------------- fallback grid detection (previous approach) ----------------
def _fallback_grid_detection(warped, thc, debug=False, params=None):
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    margin_x = 120
    margin_y = 200
    grid_w = WARP_W - 2*margin_x
    grid_h = WARP_H - margin_y - 200
    rows = 20
    blocks = 5
    choices = 4
    cell_h = grid_h / rows
    block_w = grid_w / blocks

    # params fallback
    if params is None: params = {}
    fill_thresh = float(params.get("fill_thresh", 0.18))
    ambig_low = float(params.get("ambig_low", 0.08))

    answers = {}
    overlay = Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(overlay)
    for b in range(blocks):
        for r in range(rows):
            qnum = b*rows + r + 1
            x0 = int(margin_x + b*block_w)
            y0 = int(margin_y + r*cell_h)
            choice_infos = []
            for c in range(choices):
                cx = int(x0 + (c + 0.5)*(block_w/choices))
                cy = int(y0 + cell_h*0.5)
                radius = int(max(8, min(block_w/choices, cell_h)*0.25))
                x1 = max(0, cx-radius); y1 = max(0, cy-radius)
                x2 = min(WARP_W, cx+radius); y2 = min(WARP_H, cy+radius)
                patch = thc[y1:y2, x1:x2]
                if patch.size == 0:
                    fill = 0.0
                else:
                    fill = float(np.count_nonzero(patch))/float(patch.size)
                choice_infos.append({'x':cx,'y':cy,'r':radius,'fill':round(fill,3)})
            fills = [ci['fill'] for ci in choice_infos]
            max_idx = int(np.argmax(fills)); max_val = fills[max_idx]
            marked = max_val > fill_thresh
            ambiguous = ambig_low <= max_val <= fill_thresh
            answer_choice = ''
            if marked or ambiguous:
                answer_choice = ['A','B','C','D'][max_idx]
            answers[str(qnum)] = {'choice': answer_choice, 'confidence': round(max_val,3), 'ambiguous': ambiguous, 'centers': choice_infos}
            if debug:
                for idx_ch, ci in enumerate(choice_infos):
                    cx,cy,rad = ci['x'],ci['y'],ci['r']
                    col = (0,255,0) if idx_ch==max_idx and marked and not ambiguous else (255,165,0) if ambiguous else (200,200,200)
                    draw.ellipse([(cx-rad,cy-rad),(cx+rad,cy+rad)], outline=col, width=2)
                if answer_choice:
                    try:
                        font = ImageFont.truetype("arial.ttf", 18)
                    except Exception:
                        font = ImageFont.load_default()
                    draw.text((choice_infos[max_idx]['x']-8, choice_infos[max_idx]['y']-choice_infos[max_idx]['r']-20),
                              answer_choice, fill=(255,0,0), font=font)
    return answers, overlay

# ---------------- scoring ----------------
def score_answers(extracted, key):
    """
    Robust scoring:
      - normalize key entries to a set of letters (handles "A", "A,B", "[A,B]" etc.)
      - if chosen is in key-set -> correct
    Returns {"total", "per_subject", "details"}
    """
    import re
    def parse_correct_value(v):
        if v is None:
            return set()
        s = str(v).strip()
        s = s.strip("[](){} ")
        letters = re.findall(r"[A-Za-z]", s)
        letters = [ch.upper() for ch in letters]
        return set(letters)

    if isinstance(key, dict) and "answers" in key and isinstance(key["answers"], dict):
        raw_map = key["answers"]
    else:
        raw_map = key if isinstance(key, dict) else {}

    norm_key = {}
    for k, v in raw_map.items():
        q = str(k).strip()
        norm_key[q] = parse_correct_value(v)

    total = 0
    per_subject = [0]*5
    details = {}
    for qstr, data in extracted.items():
        q = str(qstr).strip()
        chosen_raw = data.get("choice","")
        chosen = str(chosen_raw).strip().upper() if chosen_raw is not None else ""
        if len(chosen) > 1:
            import re
            found = re.findall(r"[A-Z]", chosen)
            chosen = found[0] if len(found) else ""
        correct_set = norm_key.get(q, set())
        is_correct = False
        if chosen != "" and correct_set:
            if chosen in correct_set:
                is_correct = True
        if is_correct:
            try:
                subject = (int(q)-1)//20
                if 0 <= subject < 5:
                    per_subject[subject] += 1
            except:
                pass
            total += 1
        details[q] = {
            "chosen": chosen,
            "correct_set": "".join(sorted(list(correct_set))) if correct_set else "",
            "confidence": data.get("confidence", 0),
            "ambiguous": data.get("ambiguous", False),
            "is_correct": is_correct,
            "centers": data.get("centers", [])
        }
    return {"total": total, "per_subject": per_subject, "details": details}
