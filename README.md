# Player Tracking with YOLO and Deep SORT

This repository contains a complete pipeline for soccer player detection and tracking using YOLOv11 and Deep SORT. Players are tracked with consistent IDs across frames. This version uses CPU-only inference and does not include team classification.

---

## Folder Structure

soccer-tracking/  
├── Script.py (python script)  
├── best.pt   (Ultralytics YOLOv11 model)  
├── soccer2.mp4      (Input video)  
├── clean_output.mp4    (Output video)  
├── README.md     Setup and usage instructions)  

# Model & Tracker Used :
  The YOLOv11 model file (best.pt) is not included in this repo due to GitHub’s 100 MB limit.
  LINK :-

##  Setup Instructions

### Requirements

* Python 3.8+
* pip

### Install Dependencies

```bash
pip install ultralytics opencv-python deep_sort_realtime numpy
```

### Run the Script

```bash
python Script.py
```

The output video will be saved as clean_output.mp4.

---

##  Model & Tracker Used

* **Detection:** YOLOv11 (best.pt trained on player + goalkeeper)
* **Tracking:** Deep SORT (CPU-compatible)

---
# Input / Output Videos
1. Input Link :- https://drive.google.com/file/d/1fAE7oTx5kVz9gSEz4OW5Uoia-3qea9Pl/view?usp=drive_link
2. Output Link :- https://drive.google.com/file/d/1mwxmjrPR_d2rHUw6S1KXtjcgtY2DdCRT/view?usp=drive_link



## Notes

* The `best.pt` file should detect "player" and "goalkeeper" classes.
* This version only adds bounding boxes and IDs for tracking.
* Jersey color classification and team separation are not included in this simplified version.

---

## Report Summary

### Approach & Methodology

* **YOLOv11** detects players and goalkeepers frame-by-frame.
* **Deep SORT** assigns consistent IDs using motion and spatial correlation.

### Techniques Used

* Bounding box formatting to (x, y, w, h) for tracker compatibility.
* Consistent ID drawing using rectangle overlays in OpenCV.

### Challenges Faced

* Initially, bounding boxes were too large and caused visual clutter. This was adjusted for better clarity.
* Deep SORT fails to maintain consistent IDs when a player leaves and re-enters the frame.
* I used CPU-only inference, which limits performance and tracking robustness.
* If I had access to GPU, I could have used advanced trackers like **BoT-SORT** or **StrongSORT** which are more reliable in maintaining identity.

### Potential Improvements

* Integrate **BoT-SORT** for stronger ID persistence.
* We can add jersey classification using clustering or HSV filters.
* Use lighter model like YOLOv8n for faster CPU inference.

---
