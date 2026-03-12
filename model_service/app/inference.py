import cv2
import base64
import numpy as np

from app.preprocess import preprocess_image
from app.postprocess import process_output


def encode_image_to_base64(image: np.ndarray) -> str:
    # OpenCV يحتاج BGR عند الحفظ
    if len(image.shape) == 3 and image.shape[2] == 3:
        image_to_save = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        image_to_save = image

    success, buffer = cv2.imencode(".png", image_to_save)
    if not success:
        raise ValueError("Failed to encode image")
    return base64.b64encode(buffer).decode("utf-8")


def create_mask_preview(mask_resized: np.ndarray) -> np.ndarray:
    """
    خلفية سوداء + الورم أبيض
    حتى يظهر بوضوح في الواجهة
    """
    mask_image = (mask_resized * 255).astype("uint8")
    mask_rgb = cv2.cvtColor(mask_image, cv2.COLOR_GRAY2RGB)
    return mask_rgb


def create_overlay(original_image: np.ndarray, mask_resized: np.ndarray) -> np.ndarray:
    """
    يرسم فقط على منطقة الورم وليس الصورة كاملة
    """
    overlay = original_image.copy()

    # قناع منطقي
    mask_bool = mask_resized.astype(bool)

    # طبقة حمراء
    red_layer = np.zeros_like(original_image)
    red_layer[:, :, 0] = 255  # RGB red

    # تلوين منطقة الورم فقط
    overlay[mask_bool] = (
        0.65 * original_image[mask_bool] + 0.35 * red_layer[mask_bool]
    ).astype("uint8")

    # رسم حدود الورم فوق الصورة
    contours, _ = cv2.findContours(
        mask_resized.astype("uint8"),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    cv2.drawContours(overlay_bgr, contours, -1, (0, 0, 255), 2)  # red contour in BGR
    overlay = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

    return overlay


def run_inference(model, image):
    """
    image = RGB numpy array
    """
    original_h, original_w = image.shape[:2]

    tensor = preprocess_image(image)

    mask_logits, class_logits = model(tensor)

    mask_binary, predicted_class, label, confidence = process_output(mask_logits, class_logits)

    # resize mask to original size
    mask_resized = cv2.resize(
        mask_binary.astype("uint8"),
        (original_w, original_h),
        interpolation=cv2.INTER_NEAREST,
    )

    lesion_ratio = float(mask_resized.sum()) / float(mask_resized.size)

    mask_preview = create_mask_preview(mask_resized)
    overlay_image = create_overlay(image, mask_resized)

    mask_base64 = encode_image_to_base64(mask_preview)
    overlay_base64 = encode_image_to_base64(overlay_image)

    return {
        "predicted_class": int(predicted_class),
        "label": label,
        "confidence": float(confidence),
        "lesion_area_ratio": float(lesion_ratio),
        "mask_base64": mask_base64,
        "overlay_base64": overlay_base64,
    }