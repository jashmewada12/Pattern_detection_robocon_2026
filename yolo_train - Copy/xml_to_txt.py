import xml.etree.ElementTree as ET
import os


def convert_to_dummy_pose(xml_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    tree = ET.parse(xml_path)
    root = tree.getroot()

    for img in root.findall("image"):
        img_w = float(img.get("width"))
        img_h = float(img.get("height"))
        txt_name = os.path.splitext(img.get("name"))[0] + ".txt"

        with open(os.path.join(output_folder, txt_name), "w") as f:
            for box in img.findall("box"):
                # Normalized Box Coordinates
                x_center = ((float(box.get("xtl")) + float(box.get("xbr"))) / 2) / img_w
                y_center = ((float(box.get("ytl")) + float(box.get("ybr"))) / 2) / img_h
                w = (float(box.get("xbr")) - float(box.get("xtl"))) / img_w
                h = (float(box.get("ybr")) - float(box.get("ytl"))) / img_h

                # Create a Dummy Keypoint at the center (x, y, visibility=2)
                # Format: class_id x_c y_c w h px1 py1 pv1
                f.write(
                    f"0 {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f} {x_center:.6f} {y_center:.6f} 2.0\n"
                )


# Run this to populate your label folders
convert_to_dummy_pose(
    "C:/Users/ASUS/Desktop/yolo_train/pose_dataset/labels/annotations.xml",
    "yolo_pose_dataset/train/labels",
)
convert_to_dummy_pose(
    "C:/Users/ASUS/Desktop/yolo_train/pose_dataset/labels/annotations.xml",
    "yolo_pose_dataset/val/labels",
)
