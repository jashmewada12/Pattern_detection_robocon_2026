import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from ultralytics import YOLO


def main():
    # --- CONFIGURATION ---
    FINDER_MODEL = "finder.pt"
    READER_MODEL = "reader_best.pth"
    CLASS_FILE = "classes.txt"
    CONF_THRESHOLD = 0.5
    # ---------------------

    # 1. Load Class Names
    try:
        with open(CLASS_FILE, "r") as f:
            class_names = [line.strip() for line in f.readlines() if line.strip()]
        print(f"✅ Loaded {len(class_names)} classes: {class_names}")
    except FileNotFoundError:
        print(f"❌ Error: Could not find {CLASS_FILE}.")
        return

    # 2. Load Models
    print("🚀 Loading Models...")
    try:
        # Load Finder
        finder = YOLO(FINDER_MODEL, task="detect")

        # Load Reader (PyTorch)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device} for Reader")

        # Load the PyTorch model (weights_only=True to avoid FutureWarning and for security)
        reader_state = torch.load(READER_MODEL, map_location=device, weights_only=True)

        # If reader.pth is a state_dict instead of a full model
        if isinstance(reader_state, dict):
            print(
                "⚠️ reader.pth is a state_dict. Attempting to auto-detect architecture..."
            )
            import torchvision.models as models

            loaded = False

            # Architectures to try (MobileNetV3 small/large, ResNet18, ResNet50)
            archs = [
                (models.mobilenet_v3_small, "classifier", 3),
                (models.mobilenet_v3_large, "classifier", 3),
                (models.resnet18, "fc", None),
                (models.resnet50, "fc", None),
            ]

            for arch_fn, clf_name, idx in archs:
                try:
                    model = arch_fn(weights=None)

                    # Determine num_classes from the state_dict directly
                    weight_key = (
                        f"{clf_name}.{idx}.weight"
                        if idx is not None
                        else f"{clf_name}.weight"
                    )
                    if weight_key in reader_state:
                        num_cls = reader_state[weight_key].shape[0]
                    else:
                        num_cls = len(class_names)  # fallback

                    if clf_name == "classifier":
                        in_feat = model.classifier[idx].in_features
                        model.classifier[idx] = torch.nn.Linear(in_feat, num_cls)
                    elif clf_name == "fc":
                        in_feat = model.fc.in_features
                        model.fc = torch.nn.Linear(in_feat, num_cls)

                    # Try loading
                    model.load_state_dict(reader_state, strict=True)
                    reader = model.to(device)
                    print(
                        f"✅ Successfully loaded state_dict as {arch_fn.__name__} (Classes: {num_cls})!"
                    )
                    if num_cls != len(class_names):
                        print(
                            f"⚠️ Warning: Model was trained on {num_cls} classes, but classes.txt has {len(class_names)} classes."
                        )
                    loaded = True
                    break
                except Exception as e:
                    print(f"   [Debug] Failed to load as {arch_fn.__name__}: {e}")
                    pass

            if not loaded:
                raise TypeError(
                    "Could not automatically determine the model architecture from the state_dict. Please specify it manually."
                )
        else:
            reader = reader_state

        reader.eval()

    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return

    # Transformations for the reader model (ImageNet stats are typical)
    reader_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # 3. Start Camera (change 1 to 0 if your webcam is on index 0)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("✅ System Ready. Press 'q' to exit.")

    while True:
        success, frame = cap.read()
        if not success:
            break

        # A. Run YOLO Finder
        results = finder(frame, conf=CONF_THRESHOLD, verbose=False)

        for r in results:
            for box in r.boxes:
                # Get coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                # Safety Clip
                h, w, _ = frame.shape
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                # B. Preprocess for Reader
                try:
                    # Convert OpenCV BGR to PIL RGB
                    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(crop_rgb)

                    # Apply transforms and add batch dimension
                    input_tensor = reader_transforms(pil_img).unsqueeze(0).to(device)

                    # C. Run Classification
                    with torch.no_grad():
                        outputs = reader(input_tensor)
                        probs = torch.nn.functional.softmax(outputs, dim=1)
                        pred_id = torch.argmax(probs, dim=1).item()
                        confidence = probs[0][pred_id].item()

                    # Get Label Name
                    if pred_id < len(class_names):
                        pred_label = class_names[pred_id]
                    else:
                        pred_label = "Unknown"

                    # --- D. COLOR LOGIC ---
                    if "fake" in pred_label.lower():
                        color = (0, 0, 255)  # RED for fake
                        status = "FAKE"
                    elif "real" in pred_label.lower():
                        color = (0, 255, 0)  # GREEN for real
                        status = "REAL"
                    else:
                        color = (255, 0, 0)  # BLUE for unknown
                        status = pred_label

                    # Draw Box and Text
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                    cv2.putText(
                        frame,
                        f"{status}: {pred_label} ({confidence:.2f})",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        color,
                        2,
                    )

                except Exception as e:
                    print(f"Error processing crop: {e}")
                    pass  # Skip bad crops

        cv2.imshow("Inference (Press 'q' to exit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
