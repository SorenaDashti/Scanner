# document_scanner.py

import cv2
import numpy as np
import os
from typing import Tuple, Optional, List, Dict, Any
import traceback
import math

# --- Helper Functions (بدون تغییر نسبت به نسخه قبلی با angle) ---

def order_points(pts: np.ndarray) -> np.ndarray:
    """مرتب‌سازی ۴ نقطه به ترتیب: بالا-چپ، بالا-راست، پایین-راست، پایین-چپ."""
    if not isinstance(pts, np.ndarray):
        try: pts = np.array(pts, dtype="float32")
        except ValueError: raise ValueError(f"Input pts type {type(pts)} cannot be converted.")
    # اصلاح شکل ورودی
    if pts.ndim == 1 and pts.size == 8: pts = pts.reshape((4, 2))
    elif pts.shape == (4, 1, 2): pts = pts.reshape((4, 2))
    if pts.shape != (4, 2): raise ValueError(f"order_points requires shape (4, 2), got {pts.shape}")

    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)] # بالا-چپ: کمترین مجموع
    rect[2] = pts[np.argmax(s)] # پایین-راست: بیشترین مجموع
    diff = np.diff(pts, axis=1) # تفاضل y-x
    rect[1] = pts[np.argmin(diff)] # بالا-راست: کمترین تفاضل
    rect[3] = pts[np.argmax(diff)] # پایین-چپ: بیشترین تفاضل
    return rect

def four_point_transform(image: np.ndarray, pts: np.ndarray) -> Optional[np.ndarray]:
    """اعمال تبدیل پرسپکتیو روی تصویر با استفاده از ۴ نقطه داده شده."""
    try:
        rect = order_points(pts)
        (tl, tr, br, bl) = rect
    except ValueError as e:
        print(f"[ERROR] TFM Order points failed: {e}")
        return None

    # محاسبه عرض و ارتفاع مقصد
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))

    # بررسی ابعاد نامعتبر
    if maxWidth <= 10 or maxHeight <= 10:
        print(f"[ERROR] TFM Invalid calculated dimensions: {maxWidth}x{maxHeight}")
        return None

    # نقاط مقصد
    dst = np.array([[0, 0],[maxWidth - 1, 0],[maxWidth - 1, maxHeight - 1],[0, maxHeight - 1]], dtype="float32")

    # ماتریس تبدیل
    M = cv2.getPerspectiveTransform(rect, dst)
    if M is None:
        print(f"[ERROR] TFM getPerspectiveTransform failed. Rect:\n{rect}")
        return None

    # اعمال تبدیل
    try:
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight), flags=cv2.INTER_LANCZOS4)
        return warped
    except cv2.error as e:
        print(f"[ERROR] TFM warpPerspective failed: {e}")
        return None

def calculate_angles(pts: np.ndarray) -> List[float]:
    """محاسبه ۴ زاویه داخلی یک چهارضلعی (بر حسب درجه)."""
    if pts.shape != (4, 2):
        return []
    try:
        ordered_pts = order_points(pts)
    except ValueError:
        return []

    angles = []
    for i in range(4):
        p1 = ordered_pts[i]
        p2 = ordered_pts[(i + 1) % 4]
        p0 = ordered_pts[(i - 1 + 4) % 4]
        v1 = p0 - p1
        v2 = p2 - p1
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        if norm_v1 == 0 or norm_v2 == 0:
            return [] # خطا: بردار صفر
        # محاسبه کسینوس زاویه و تبدیل به درجه
        cos_angle = np.clip(np.dot(v1, v2) / (norm_v1 * norm_v2), -1.0, 1.0)
        angle_deg = math.degrees(math.acos(cos_angle))
        angles.append(angle_deg)
    return angles

def check_angles(angles: List[float], min_angle: float = 65.0, max_angle: float = 115.0) -> bool:
    """بررسی اینکه آیا تمام زوایا در محدوده مشخصی هستند."""
    if not angles or len(angles) != 4:
        return False
    return all(min_angle <= angle <= max_angle for angle in angles)

def refine_corner(corner: Tuple[float, float], gray_img: np.ndarray, search_window_size: int = 11) -> Tuple[float, float]:
    """پالایش موقعیت یک گوشه با استفاده از cornerSubPix."""
    if search_window_size % 2 == 0: search_window_size += 1
    x_flt, y_flt = corner[0], corner[1]
    h, w = gray_img.shape
    # بررسی اولیه بودن نقطه در تصویر
    if not (0 <= x_flt < w and 0 <= y_flt < h): return corner

    win_size = (search_window_size // 2, search_window_size // 2)
    zero_zone = (-1, -1)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
    corners_to_refine = np.array([[[x_flt, y_flt]]], dtype=np.float32)
    try:
        cv2.cornerSubPix(gray_img, corners_to_refine, win_size, zero_zone, criteria)
        refined_corner = tuple(corners_to_refine[0][0])
        refined_x, refined_y = refined_corner
        # بررسی مجدد بودن نقطه پالایش شده در تصویر
        if not (0 <= refined_x < w and 0 <= refined_y < h): return corner
        return refined_corner
    except (cv2.error, Exception): # گرفتن خطاهای احتمالی cv2 یا غیره
        # print(f"[DEBUG] cornerSubPix failed for ({x_flt:.1f},{y_flt:.1f})")
        return corner

def refine_all_corners(quad: np.ndarray, gray_img: np.ndarray, search_window_size: int = 11) -> Optional[np.ndarray]:
    """پالایش تمام 4 گوشه چهارضلعی با cornerSubPix. None در صورت خطا برمیگرداند."""
    try:
        ordered_quad = order_points(quad)
    except ValueError as e:
        print(f"[WARNING] Could not order points before refinement: {e}.")
        # اگر شکل ورودی درست بود، با همان ترتیب ادامه بده
        if quad.shape == (4,2):
             ordered_quad = quad.astype("float32") # اطمینان از float32
        else:
             print(f"[ERROR] Invalid shape {quad.shape} for refinement & ordering failed.")
             return None # بازگشت None در صورت خطای جدی شکل

    refined_quad_list = []
    for i in range(4):
        refined_point = refine_corner(tuple(ordered_quad[i]), gray_img, search_window_size)
        # بررسی اینکه refine_corner نقطه معتبری برگردانده
        if refined_point == tuple(ordered_quad[i]): # اگر پالایش ناموفق بود
             print(f"[DEBUG] Refinement seemed to fail for point {i}")
             # میتوان تصمیم گرفت که کل پالایش ناموفق است یا با نقطه اصلی ادامه داد
             # فعلا با نقطه اصلی ادامه میدهیم
             refined_quad_list.append(list(ordered_quad[i]))
        else:
             refined_quad_list.append(list(refined_point))

    refined_quad = np.array(refined_quad_list, dtype="float32")

    # مرتب سازی نهایی پس از پالایش
    try:
        final_ordered_refined_quad = order_points(refined_quad)
        return final_ordered_refined_quad
    except ValueError as e:
        print(f"[ERROR] Could not order points *after* refinement: {e}. Returning un-ordered refined.")
        # اگر مرتب سازی نهایی هم شکست خورد، نقاط پالایش شده (ولی شاید نامرتب) را برگردان
        # یا None برگردانیم؟ فعلا نامرتب را برمی گردانیم
        return refined_quad

# --- تابع جستجوی چهارضلعی ایده آل ---
def find_document_quad_candidates(
    contours: List[np.ndarray],
    img_area: int,
    min_area_ratio: float = 0.05,
    max_area_ratio: float = 0.95,
    min_solidity: float = 0.85,
    aspect_ratio_range: Tuple[float, float] = (0.4, 2.5),
    num_contours_to_check: int = 20,
    epsilon_factor_range: Tuple[float, float] = (0.01, 0.06),
    epsilon_steps: int = 8,
    require_angle_check: bool = True,
    angle_tolerance_deg: float = 25.0
) -> List[Tuple[float, np.ndarray]]:
    """
    یافتن لیستی از کانتورهای چهارضلعی *محدب* کاندیدا.
    پارامترها انعطاف پذیری را کنترل می کنند.
    """
    candidates = []
    min_contour_area = img_area * min_area_ratio
    max_contour_area = img_area * max_area_ratio
    min_angle = 90.0 - angle_tolerance_deg
    max_angle = 90.0 + angle_tolerance_deg

    num_to_check = min(num_contours_to_check, len(contours))
    if num_to_check <= 0: return [] # اگر کانتوری نبود

    contours_to_process = sorted(contours, key=cv2.contourArea, reverse=True)[:num_to_check]

    for cnt in contours_to_process:
        cnt_area = cv2.contourArea(cnt)

        if not (min_contour_area <= cnt_area <= max_contour_area): continue
        peri = cv2.arcLength(cnt, True)
        if peri <= 0: continue

        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area > 0:
            solidity = cnt_area / float(hull_area)
            if solidity < min_solidity: continue
        else: continue

        best_approx_for_cnt = None
        max_approx_area = 0

        for eps_factor in np.linspace(epsilon_factor_range[0], epsilon_factor_range[1], epsilon_steps):
            epsilon = eps_factor * peri
            approx = cv2.approxPolyDP(cnt, epsilon, True)

            if len(approx) == 4:
                approx_area = cv2.contourArea(approx)
                if approx_area < min_contour_area * 0.7: continue
                if not cv2.isContourConvex(approx): continue

                aspect_ratio_ok = False
                current_aspect_ratio = 0
                try:
                    # ورودی minAreaRect باید float32 باشد
                    rect = cv2.minAreaRect(approx.astype(np.float32))
                    (w, h) = rect[1]
                    if w > 1 and h > 1:
                        current_aspect_ratio = max(w / h, h / w)
                        if aspect_ratio_range[0] <= current_aspect_ratio <= aspect_ratio_range[1]:
                            aspect_ratio_ok = True
                except (cv2.error, ZeroDivisionError): pass # خطا به معنی عدم موفقیت

                if not aspect_ratio_ok: continue

                angles_ok = True
                if require_angle_check:
                     angles = calculate_angles(approx.reshape(4, 2))
                     if not check_angles(angles, min_angle, max_angle):
                         angles_ok = False

                if angles_ok:
                    if approx_area > max_approx_area:
                        max_approx_area = approx_area
                        best_approx_for_cnt = approx

        if best_approx_for_cnt is not None:
            score = max_approx_area
            candidates.append((score, best_approx_for_cnt.reshape(4, 2).astype("float32")))

    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates


# --- Main Scanning Function (با ساختار اصلاح شده) ---
def scan_document(
    image_path: str,
    save_debug_steps: bool = False,
    output_dir_base: str = "scan_results_more_layers",
    interp_height: int = 1000,

    # --- پارامترهای پیش پردازش ---
    bilateral_d: int = 9,
    bilateral_sigma_color: int = 80,
    bilateral_sigma_space: int = 80,
    gaussian_blur_ksize: Tuple[int, int] = (5, 5), # دیگر کاما اضافی ندارد
    clahe_clip_limit: float = 2.0,
    clahe_tile_grid_size: Tuple[int, int] = (8, 8),

    # --- پارامترهای استخراج کانتور ---
    adaptive_thresh_block_size: int = 35,
    adaptive_thresh_C: int = 9,
    morph_kernel_size: int = 3,
    morph_close_iterations: int = 3,
    canny_low_thresh: int = 35,
    canny_high_thresh: int = 120,
    dilation_kernel_size: int = 3,
    dilation_iterations: int = 2,

    # === پارامترهای لایه های جستجو ===
    num_contours_to_check: int = 30,
    epsilon_range: Tuple[float, float] = (0.01, 0.08),
    epsilon_steps: int = 9,

    # لایه 1: دقیق (Strict)
    strict_area: Tuple[float, float] = (0.05, 0.95),
    strict_solidity: float = 0.90,
    strict_aspect_ratio: Tuple[float, float] = (1/2.2, 2.2),
    strict_angle_tol: float = 15.0,

    # لایه 2: متعادل (Balanced)
    balanced_area: Tuple[float, float] = (0.04, 0.96),
    balanced_solidity: float = 0.85,
    balanced_aspect_ratio: Tuple[float, float] = (1/2.8, 2.8),
    balanced_angle_tol: float = 25.0,

    # لایه 3: انعطاف پذیر (Loose)
    loose_area: Tuple[float, float] = (0.03, 0.97),
    loose_solidity: float = 0.78,
    loose_aspect_ratio: Tuple[float, float] = (1/3.8, 3.8),
    loose_angle_tol: float = 38.0,

    # لایه 4: بسیار انعطاف پذیر (Very Loose) - بدون بررسی زاویه
    very_loose_area: Tuple[float, float] = (0.02, 0.98),
    very_loose_solidity: float = 0.70,
    very_loose_aspect_ratio: Tuple[float, float] = (1/5.0, 5.0),

    # === پارامترهای Fallback ===
    fallback_min_area: float = 0.015,
    fallback_approx_eps: float = 0.025,
    fallback_min_rect_aspect_ratio: float = 1.2,

    # --- پارامترهای پالایش گوشه ---
    refine_corner_active: bool = True,
    refine_corner_window_size: int = 17,
    refine_gray_blur_ksize: Tuple[int, int] = (3, 3)

) -> Optional[np.ndarray]:
    """
    اسکن سند با رویکرد چند لایه گسترش یافته و پیش‌پردازش جایگزین.
    (توضیحات تابع مانند قبل)
    """
    # --- شروع بدنه تابع scan_document (تورفتگی صحیح) ---
    print(f"--- Processing: {os.path.basename(image_path)} ---")
    debug_data: Dict[str, Any] = {'images': {}, 'logs': []}
    debug_image_counter = 0 # برای شماره گذاری تصاویر دیباگ

    def save_debug_img(name: str, img: np.ndarray):
        nonlocal debug_image_counter
        if save_debug_steps and img is not None:
             # اضافه کردن شماره به نام فایل برای حفظ ترتیب
             debug_data['images'][f"{debug_image_counter:02d}_{name}"] = img.copy()
             debug_image_counter += 1

    def log_debug(message: str):
        print(message)
        debug_data['logs'].append(message)

    # --- ایجاد پوشه ها ---
    try:
        output_dir = os.path.join(output_dir_base, "scanned")
        debug_dir = os.path.join(output_dir_base, "debug_steps")
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        os.makedirs(output_dir, exist_ok=True)
        if save_debug_steps:
            os.makedirs(debug_dir, exist_ok=True)
    except OSError as e:
        log_debug(f"[ERROR] Cannot create directories: {e}")
        return None # خطا در ایجاد پوشه

    # --- بارگذاری و تغییر اندازه ---
    try:
        image = cv2.imread(image_path)
        if image is None: raise IOError("Image load failed")
        orig = image.copy()
        img_height, img_width = image.shape[:2]
        if img_height <= 0 or img_width <= 0: raise ValueError("Invalid image dims")
        ratio = img_height / float(interp_height)
        interp_width = int(img_width / ratio)
        if interp_width <= 0 : raise ValueError("Invalid interp_width")
        resized = cv2.resize(image, (interp_width, interp_height), interpolation=cv2.INTER_LANCZOS4)
        resized_area = interp_width * interp_height
        log_debug(f"Image loaded and resized to {interp_width}x{interp_height}")
        save_debug_img("Resized", resized)
    except Exception as e:
        log_debug(f"[ERROR] Loading/Resizing failed: {e}")
        traceback.print_exc()
        return None # خطا در بارگذاری

    # --- پیش‌پردازش ها ---
    gray_images = {}
    gray_for_refine = None
    try:
        # 1. استاندارد
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        gray_filtered = cv2.bilateralFilter(gray, bilateral_d, bilateral_sigma_color, bilateral_sigma_space)
        gray_blurred = cv2.GaussianBlur(gray_filtered, gaussian_blur_ksize, 0)
        gray_images['standard'] = gray_blurred
        save_debug_img("Gray_Standard_Blurred", gray_blurred)

        # 2. CLAHE
        clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_tile_grid_size)
        gray_clahe = clahe.apply(gray)
        gray_clahe_blurred = cv2.GaussianBlur(gray_clahe, (3,3), 0)
        gray_images['clahe'] = gray_clahe_blurred
        save_debug_img("Gray_CLAHE_Blurred", gray_clahe_blurred)

        # 3. برای پالایش
        # اطمینان از فرد بودن سایز کرنل
        refine_ksize = (max(1, refine_gray_blur_ksize[0] // 2 * 2 + 1), max(1, refine_gray_blur_ksize[1] // 2 * 2 + 1))
        if refine_ksize[0] > 0 and refine_ksize[1] > 0:
            gray_for_refine = cv2.GaussianBlur(gray, refine_ksize, 0)
        else:
            gray_for_refine = gray.copy() # کپی برای اطمینان
        save_debug_img("Gray_For_Refine", gray_for_refine)

    except Exception as e:
        log_debug(f"[ERROR] Preprocessing failed: {e}")
        traceback.print_exc()
        return None # خطا در پیش پردازش

    # --- استخراج کانتورها با روش های مختلف ---
    contour_sources = {}
    for pp_name, gray_img in gray_images.items():
        log_debug(f"\n--- Finding Contours using Preprocessing: {pp_name.upper()} ---")
        contours_adaptive_list = []
        contours_canny_list = []
        contours_adaptive_external = []
        contours_canny_external = []

        # اطمینان از فرد بودن بلاک سایز
        current_adaptive_block_size = max(3, adaptive_thresh_block_size // 2 * 2 + 1)

        try: # Adaptive Thresholding
            thresh = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, current_adaptive_block_size, adaptive_thresh_C)
            # اطمینان از فرد بودن کرنل مورفولوژی
            current_morph_ksize = max(1, morph_kernel_size // 2 * 2 + 1)
            kernel_morph = cv2.getStructuringElement(cv2.MORPH_RECT, (current_morph_ksize, current_morph_ksize))
            closed_adaptive = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_morph, iterations=morph_close_iterations)
            save_debug_img(f"{pp_name}_AdaptiveClosed", closed_adaptive)
            contours_adaptive_list, _ = cv2.findContours(closed_adaptive.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            contours_adaptive_external, _ = cv2.findContours(closed_adaptive.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            log_debug(f"Adaptive [{pp_name}]: Found {len(contours_adaptive_list)} (LIST), {len(contours_adaptive_external)} (EXTERNAL)")
        except Exception as e:
             log_debug(f"[ERROR] Adaptive contour finding failed [{pp_name}]: {e}")
             traceback.print_exc()

        try: # Canny
            edges = cv2.Canny(gray_img, canny_low_thresh, canny_high_thresh)
            # اطمینان از فرد بودن کرنل دیلیشن
            current_dilate_ksize = max(1, dilation_kernel_size // 2 * 2 + 1)
            kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (current_dilate_ksize, current_dilate_ksize))
            dilated_edges = cv2.dilate(edges, kernel_dilate, iterations=dilation_iterations)
            closed_canny = cv2.morphologyEx(dilated_edges, cv2.MORPH_CLOSE, kernel_dilate, iterations=1)
            save_debug_img(f"{pp_name}_CannyClosed", closed_canny)
            contours_canny_list, _ = cv2.findContours(closed_canny.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            contours_canny_external, _ = cv2.findContours(closed_canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            log_debug(f"Canny    [{pp_name}]: Found {len(contours_canny_list)} (LIST), {len(contours_canny_external)} (EXTERNAL)")
        except Exception as e:
            log_debug(f"[ERROR] Canny contour finding failed [{pp_name}]: {e}")
            traceback.print_exc()

        contour_sources[f'{pp_name}_list'] = contours_adaptive_list + contours_canny_list
        contour_sources[f'{pp_name}_external'] = contours_adaptive_external + contours_canny_external


    # --- جستجوی چند لایه ---
    best_quad_found = None
    source_description = "None"

    contour_source_priority = ['standard_list', 'standard_external', 'clahe_list', 'clahe_external']

    search_tiers = [
        {'name': 'Strict', 'params': {'min_area_ratio': strict_area[0], 'max_area_ratio': strict_area[1], 'min_solidity': strict_solidity, 'aspect_ratio_range': strict_aspect_ratio, 'require_angle_check': True, 'angle_tolerance_deg': strict_angle_tol}},
        {'name': 'Balanced', 'params': {'min_area_ratio': balanced_area[0], 'max_area_ratio': balanced_area[1], 'min_solidity': balanced_solidity, 'aspect_ratio_range': balanced_aspect_ratio, 'require_angle_check': True, 'angle_tolerance_deg': balanced_angle_tol}},
        {'name': 'Loose', 'params': {'min_area_ratio': loose_area[0], 'max_area_ratio': loose_area[1], 'min_solidity': loose_solidity, 'aspect_ratio_range': loose_aspect_ratio, 'require_angle_check': True, 'angle_tolerance_deg': loose_angle_tol}},
        {'name': 'VeryLoose', 'params': {'min_area_ratio': very_loose_area[0], 'max_area_ratio': very_loose_area[1], 'min_solidity': very_loose_solidity, 'aspect_ratio_range': very_loose_aspect_ratio, 'require_angle_check': False, 'angle_tolerance_deg': 90.0}},
    ]

    # حلقه اصلی جستجو
    for cs_key in contour_source_priority:
        contours = contour_sources.get(cs_key, [])
        if not contours:
            log_debug(f"\n--- Skipping Contour Source: {cs_key.upper()} (No contours) ---")
            continue

        log_debug(f"\n--- Searching using Contour Source: {cs_key.upper()} ({len(contours)} contours) ---")
        current_source_best_quad = None
        current_source_description = "None"

        for tier in search_tiers:
            tier_name = tier['name']
            tier_params = tier['params']
            log_debug(f"---> Trying Tier: {tier_name} [{cs_key}]")

            try:
                candidates = find_document_quad_candidates(
                    contours, resized_area,
                    num_contours_to_check=num_contours_to_check,
                    epsilon_factor_range=epsilon_range,
                    epsilon_steps=epsilon_steps,
                    **tier_params
                )
            except Exception as find_e:
                log_debug(f"[ERROR] Exception in find_document_quad_candidates ({tier_name}, {cs_key}): {find_e}")
                candidates = [] # ادامه با کاندیدای خالی

            if candidates:
                best_score, current_source_best_quad = candidates[0]
                current_source_description = f"Tier '{tier_name}' using Contours '{cs_key}'"
                log_debug(f"[SUCCESS] Found candidate in {current_source_description} (Score: {best_score:.0f})")
                if save_debug_steps:
                    img_tier_found = resized.copy()
                    cv2.drawContours(img_tier_found, [current_source_best_quad.astype(np.int32)], -1, (0, 255, 0), 3)
                    save_debug_img(f"{cs_key}_{tier_name}_Success", img_tier_found)
                break # خروج از حلقه لایه ها
            else:
                log_debug(f"Tier '{tier_name}' [{cs_key}] failed.")

        if current_source_best_quad is not None:
            best_quad_found = current_source_best_quad
            source_description = current_source_description
            break # خروج از حلقه منابع

    # =============================================
    # --- Fallback نهایی ---
    # =============================================
    if best_quad_found is None:
        log_debug("\n--- All Tiers Failed. Attempting FINAL FALLBACK ---")
        fallback_contours = []
        for cs_key in contour_source_priority:
            fallback_contours.extend(contour_sources.get(cs_key, []))

        if fallback_contours:
            fallback_contours = sorted(fallback_contours, key=cv2.contourArea, reverse=True)
            min_fb_area = resized_area * fallback_min_area
            largest_cnt = None
            log_debug(f"Fallback: Checking {len(fallback_contours)} combined contours (min area: {min_fb_area:.0f})")

            for cnt in fallback_contours:
                 current_area = cv2.contourArea(cnt)
                 if current_area < min_fb_area: break

                 try: # بررسی boundingRect ممکن است خطا دهد
                     x, y, w, h = cv2.boundingRect(cnt)
                 except cv2.error: continue

                 if w < 10 or h < 10: continue
                 aspect_ratio_bb = max(w / h, h / w)
                 if aspect_ratio_bb > 8.0: continue

                 largest_cnt = cnt
                 log_debug(f"Fallback: Using largest suitable contour (Area: {current_area:.0f}, BB AR: {aspect_ratio_bb:.1f})")
                 save_debug_img("Fallback_LargestContour", cv2.drawContours(resized.copy(), [cnt], -1, (0, 165, 255), 2))
                 break

            if largest_cnt is not None:
                 try:
                     peri = cv2.arcLength(largest_cnt, True)
                     if peri > 0:
                        # Fallback A: ApproxPolyDP
                        approx = cv2.approxPolyDP(largest_cnt, fallback_approx_eps * peri, True)
                        if len(approx) == 4 and cv2.isContourConvex(approx):
                            approx_area_fb = cv2.contourArea(approx)
                            if approx_area_fb > min_fb_area * 0.9: # مساحت تقریب هم باید معقول باشد
                                best_quad_found = approx.reshape(4, 2).astype("float32")
                                source_description = "Fallback-ApproxPolyDP"
                                log_debug("[SUCCESS] Fallback successful using ApproxPolyDP.")
                                save_debug_img("Fallback_Approx", cv2.drawContours(resized.copy(), [approx.astype(np.int32)], -1, (255, 255, 0), 2))
                            else: log_debug("[INFO] Fallback ApproxPolyDP area too small.")

                        # اگر approx موفق نبود یا مساحتش کم بود، سراغ MinAreaRect برو
                        if best_quad_found is None:
                            rect = cv2.minAreaRect(largest_cnt.astype(np.float32)) # نیاز به float32
                            box = cv2.boxPoints(rect)
                            if box is not None and box.shape == (4, 2):
                                box_area = cv2.contourArea(box)
                                if box_area > min_fb_area * 0.8:
                                    best_quad_found = box.astype("float32")
                                    source_description = "Fallback-MinAreaRect"
                                    log_debug(f"[SUCCESS] Fallback successful using MinAreaRect (Area: {box_area:.0f}).")
                                    save_debug_img("Fallback_MinAreaRect", cv2.drawContours(resized.copy(), [box.astype(np.int32)], -1, (255, 0, 255), 2))
                                else: log_debug("[WARNING] Fallback MinAreaRect result too small.")
                            else: log_debug("[WARNING] Fallback MinAreaRect failed.")
                     else: log_debug("[WARNING] Fallback failed: Largest contour zero perimeter.")
                 except Exception as e: log_debug(f"[ERROR] Exception during fallback: {e}")
            else: log_debug("[WARNING] Fallback failed: No suitable largest contour found.")
        else: log_debug("[WARNING] Fallback failed: No contours available.")

    # --- گزارش نهایی ---
    if best_quad_found is not None:
        log_debug(f"\n===> Final Quad Candidate Selected via: {source_description} <===")
    else:
        log_debug("\n===> ERROR: Document outline detection failed completely. <===")
        save_debug_img("Detection_Failed", resized)

    # =============================================
    # --- پالایش و تبدیل ---
    # =============================================
    final_quad = None
    if best_quad_found is not None:
        if refine_corner_active:
            log_debug("[INFO] Proceeding to Corner Refinement...")
            if gray_for_refine is None:
                 log_debug("[WARNING] Skipping refinement: gray_for_refine image is missing.")
                 try: # حتی اگر پالایش نشد، مرتب سازی انجام شود
                      final_quad = order_points(best_quad_found)
                 except ValueError: final_quad = None
            else:
                try:
                    refined_quad = refine_all_corners(best_quad_found, gray_for_refine, refine_corner_window_size)
                    # بررسی نتیجه پالایش
                    if refined_quad is not None and refined_quad.shape == (4, 2):
                        final_quad = refined_quad
                        log_debug("[INFO] Corner refinement successful.")
                        save_debug_img("Corner_Refinement", cv2.polylines(resized.copy(), [best_quad_found.astype(np.int32)], True, (0, 0, 255), 1) | cv2.polylines(resized.copy(), [final_quad.astype(np.int32)], True, (0, 255, 255), 2))
                    else:
                        log_debug("[WARNING] Refinement failed or returned invalid shape. Using unrefined (ordered).")
                        try: final_quad = order_points(best_quad_found)
                        except ValueError: final_quad = None
                except Exception as e:
                     log_debug(f"[ERROR] Refinement exception: {e}. Using unrefined (ordered).")
                     try: final_quad = order_points(best_quad_found)
                     except ValueError: final_quad = None
        else:
             log_debug("[INFO] Corner refinement disabled. Using found quad (ordered).")
             try: final_quad = order_points(best_quad_found)
             except ValueError as e_ord: log_debug(f"[ERROR] Ordering final quad failed: {e_ord}"); final_quad = None

    # --- تبدیل نهایی ---
    final_result = None
    if final_quad is not None and final_quad.shape == (4, 2):
        try:
            original_quad = final_quad * ratio
            log_debug("[INFO] Performing final perspective transform...")
            warped = four_point_transform(orig, original_quad)
            if warped is not None:
                 log_debug("[INFO] Perspective transformation successful.")
                 final_result = warped
                 save_debug_img("Final_Output", warped)
                 output_path = os.path.join(output_dir, f"{base_name}_scanned.jpg")
                 try:
                     cv2.imwrite(output_path, final_result, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                     log_debug(f"Successfully saved to: {output_path}")
                 except Exception as write_e: log_debug(f"[ERROR] Failed save result: {write_e}")
            else:
                 log_debug("[ERROR] Final transform returned None.")
                 save_debug_img("Transform_Failed_Final", cv2.drawContours(orig.copy(), [original_quad.astype(np.int32)], -1, (0,0,255), 3))
        except Exception as e:
            log_debug(f"[ERROR] Final transform step failed: {e}")
            traceback.print_exc()
            # سعی کن چهارضلعی مشکل ساز را ذخیره کنی
            if 'original_quad' in locals() and isinstance(original_quad, np.ndarray):
                save_debug_img("Transform_Exception_Quad", cv2.drawContours(orig.copy(), [original_quad.astype(np.int32)], -1, (0,0,255), 3))

    else:
        log_debug("[ERROR] Cannot perform final transform: No valid final quad.")

    # --- ذخیره دیباگ ---
    if save_debug_steps:
        log_debug(f"\nSaving {len(debug_data['images'])} debug images to: {debug_dir}")
        sorted_keys = sorted(debug_data['images'].keys())
        for name in sorted_keys:
            img = debug_data['images'][name]
            if img is None: continue
            try:
                debug_file_path = os.path.join(debug_dir, f"{base_name}_{name}.jpg")
                cv2.imwrite(debug_file_path, img)
            except Exception as write_e: print(f"[ERROR] Write debug img {name} failed: {write_e}")
        log_file_path = os.path.join(debug_dir, f"{base_name}_log.txt")
        try:
            with open(log_file_path, 'w', encoding='utf-8') as f: f.write('\n'.join(debug_data['logs']))
            log_debug(f"Debug log saved to: {log_file_path}")
        except Exception as log_write_e: print(f"[ERROR] Write debug log failed: {log_write_e}")

    log_debug(f"--- Processing Finished for {os.path.basename(image_path)} ---")
    return final_result # بازگرداندن نتیجه یا None


# --- Example Usage (بدون تغییر) ---
if __name__ == '__main__':
    test_image_dir = "test_images"
    # استفاده از نام پوشه جدید برای نتایج این نسخه
    output_base = "scan_results_corrected_layers"

    # ایجاد پوشه تست اگر وجود ندارد
    os.makedirs(test_image_dir, exist_ok=True)
    try:
        # یافتن فایل های تصویر در پوشه تست
        example_files = [f for f in os.listdir(test_image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'))]
    except FileNotFoundError:
        print(f"Error: Test image directory '{test_image_dir}' not found.")
        example_files = []

    if not example_files:
        print(f"Test image directory '{test_image_dir}' is empty or not found. Please add document images to test.")

    image_paths_to_process = [os.path.join(test_image_dir, f) for f in example_files]

    if not image_paths_to_process:
         print("\nNo images found to process. Exiting.")
    else:
        print(f"\nFound {len(image_paths_to_process)} images to process in '{test_image_dir}'.")

    # پردازش هر تصویر
    for img_path in image_paths_to_process:
        if not os.path.isfile(img_path):
            print(f"Warning: Skipping non-file path: {img_path}")
            continue

        print(f"\n===================================")
        print(f"Processing: {os.path.basename(img_path)}")
        print(f"===================================")

        # فراخوانی تابع اصلی اسکن
        scanned_image = scan_document(
            image_path=img_path,
            save_debug_steps=True,         # فعال بودن دیباگ بسیار مهم است
            output_dir_base=output_base,
            # --- پارامترهای دیگر را می توانید اینجا برای تست تغییر دهید ---
            # interp_height=1200, # افزایش دقت با هزینه سرعت
            # strict_solidity=0.92, # سخت گیرانه تر کردن لایه اول
            # loose_angle_tol=45.0, # تحمل زاویه بیشتر در لایه سوم
            # refine_corner_active=False, # غیرفعال کردن پالایش برای تست
        )

        # نمایش نتیجه
        if scanned_image is None:
            print(f"-----> Failed to scan document: {os.path.basename(img_path)}")
        else:
             print(f"-----> Successfully processed: {os.path.basename(img_path)}")
             # می توانید نتیجه را اینجا نمایش دهید (اختیاری)
             # try:
             #      h, w = scanned_image.shape[:2]
             #      display_h = 600
             #      display_w = int(w * (display_h / h))
             #      cv2.imshow("Scanned Result", cv2.resize(scanned_image, (display_w, display_h)))
             #      cv2.waitKey(0)
             # except Exception as display_e:
             #      print(f"Could not display result: {display_e}")

    if image_paths_to_process:
        print(f"\nProcessing complete. Check the '{output_base}' folder for results and debug steps.")

    # cv2.destroyAllWindows() # اگر از imshow استفاده می کنید