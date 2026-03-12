import torch
import numpy as np
import torch.nn.functional as F
import cv2


LABEL_MAP = {
    0: "Benign",
    1: "Malignant",
}


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype("uint8"))

    if num_labels <= 1:
        return mask.astype("uint8")

    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    largest_mask = (labels == largest_label).astype("uint8")
    return largest_mask


def process_output(mask_logits, class_logits):
    # segmentation probabilities
    mask_probs = torch.sigmoid(mask_logits)
    mask_probs = mask_probs.squeeze().detach().cpu().numpy()

    # normalize to 0..1
    mask_probs = mask_probs - mask_probs.min()
    if mask_probs.max() > 0:
        mask_probs = mask_probs / mask_probs.max()

    # adaptive threshold بدل 0.5 الثابت
    adaptive_threshold = max(0.60, float(np.percentile(mask_probs, 85)))
    mask_binary = (mask_probs >= adaptive_threshold).astype(np.uint8)

    # تنظيف خفيف
    kernel = np.ones((5, 5), np.uint8)
    mask_binary = cv2.morphologyEx(mask_binary, cv2.MORPH_OPEN, kernel)
    mask_binary = cv2.morphologyEx(mask_binary, cv2.MORPH_CLOSE, kernel)

    # احتفظ بأكبر منطقة فقط
    mask_binary = keep_largest_component(mask_binary)

    # classification
    class_probs = F.softmax(class_logits, dim=1)
    predicted_class = class_probs.argmax(dim=1).item()
    confidence = class_probs.max().item()
    label = LABEL_MAP.get(predicted_class, "Unknown")

    return mask_binary, predicted_class, label, confidence