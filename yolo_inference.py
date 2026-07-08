import cv2
import numpy as np
from ultralytics import YOLO


def analyze_flies(image_path, model_path):
    # 1. Load your specific trained weights
    model = YOLO(model_path)

    # 2. Run inference at your chosen operational threshold
    # We use stream=True for memory efficiency in web apps later
    results = model.predict(source=image_path, conf=0.5, save=False)

    # 3. Pull out the first (and only) image result
    res = results[0]

    # 4. Extract Class IDs (0=immature, 1=mature based on your previous training)
    # .cpu().numpy() ensures the data is in a format Python can easily count
    classes = res.boxes.cls.cpu().numpy()

    # 5. The Math
    counts = {
        "total": len(classes),
        "immature": int(np.count_nonzero(classes == 0)),
        "mature": int(np.count_nonzero(classes == 1))
    }

    return counts, res


# --- TEST IT LOCALLY ---
if __name__ == "__main__":
    my_model = r"C:\Users\fvazq\PycharmProjects\USDA_eyecolor\runs\detect\USDA_pupae_MvsIM\yolov8n_pupae_head_2_100epochs\weights\best.pt"
    test_image = r"C:\Users\fvazq\OneDrive\Documents\Jobs\USDA\pupae head_dataset2\test\images\000c6212-9188389a-Bre_BPSED0012448_41.jpg"

    data, raw_results = analyze_flies(test_image, my_model)

    print(f"Results for {test_image}:")
    print(f" - Total detected: {data['total']}")
    print(f" - Immature: {data['immature']}")
    print(f" - Mature: {data['mature']}")

    # Optional: Pop open a window to see if it worked
    # plotted_img = raw_results.plot()
    # cv2.imshow("Detection Check", plotted_img)
    # cv2.waitKey(0)
