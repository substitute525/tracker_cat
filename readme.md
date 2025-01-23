# Pet Tracking Project

This project implements **pet tracking** using an OpenCV2-based algorithm combined with a YOLO model. The application is designed to analyze video frames and track pets dynamically.

---

## **Environment Setup**

### **Requirements**
- **Conda version**: 3
- **Python version**: 3.10
- **OpenCV version**: 4.5.5

### **Setup Instructions**
1. Create a Conda environment:
   ```bash
   conda env create -f enviroment.yml -n net_tracking
   ```
2. Export your own environment:
   ```bash
   conda env export > environment.yml 
   ```
3. Clone the YOLO model and set up the classification ID file:
   - [YOLO Model Classification IDs](https://github.com/substitute525/tracker_cat/blob/main/app/model/yolo/coco.names)

---

## **How to Run**

### **Executable Script**
The main script for running the pet tracking project is `tracker_test.exe`. This is a compiled executable built with PyInstaller for ease of use.

### **Command Structure**
```bash
.\dist\tracker_test.exe <video_path> <model_class_id> <model_confidence> -d <debug_mode> -s <strategy> --minInterval <min_time> --maxInterval <max_time> --strategyInterval <strategy_time>
```

### **Example Command**
```bash
.\dist\tracker_test.exe D:\video.mp4 15 0.1 -d True -s WHEN_LOST --minInterval 500 --maxInterval 0 --strategyInterval 0
```

### **Parameter Descriptions**
| Parameter             | Description                                               | Example       |
|-----------------------|-----------------------------------------------------------|---------------|
| `<video_path>`        | Path to the input video file.                             | `D:\video.mp4` |
| `<model_class_id>`    | The class id of the model match                           | `15`           |
| `<model_confidence>`  | Minimum confidence of the model                           | `0.1`          |
| `-d <debug_mode>`     | Debug mode (True/False).                                  | `True`         |
| `-s <strategy>`       | Tracking strategy (e.g., `WHEN_LOST`, `WHEN_FREE`, etc.). | `WHEN_LOST`    |
| `--minInterval`       | Minimum time interval (in milliseconds).                  | `500`          |
| `--maxInterval`       | Maximum time interval (in milliseconds).                  | `0`            |
| `--strategyInterval`  | Interval for applying strategy (in milliseconds).         | `0`            |
Use the ```.\dist\tracker_test.exe --help``` command to know more

---

## **YOLO Model Information**

The YOLO model is used for pet detection and classification. The classification IDs are defined in the file:
- [coco.names](https://github.com/substitute525/tracker_cat/blob/main/app/model/yolo/coco.names)

You can find YOLO models at the following web sites:
- [YOLO4](https://huggingface.co/homohapiens/darknet-yolov4/tree/main)

Ensure that this file is downloaded and correctly referenced in your project.

---

## Compile and pack
This project uses pyInstaller for packaging, the following is the packaging command

### For Windows
``` bash
pyinstaller --onefile --add-binary "~\env\conda\envs\my_project_env\Library\bin\*.dll;." .\tests\csrt\tracker_test.py
```
### For Linux
``` bash
pyinstaller --onefile --add-binary "~/env/conda/envs/my_project_env/lib/*.so;." .\tests\csrt\tracker_test.py
```


---

## **Notes and Troubleshooting**

1. **Environment Compatibility:** This project is designed for Python 3.10 and OpenCV 4.5.5. Ensure your environment matches these versions.
2. **Debug Mode:** If you encounter unexpected behavior, enable debug mode (`-d True`) to log additional details, This outputs the processed video stream.
3. **Model Dependencies:** Make sure all YOLO dependencies and weights are properly downloaded and configured.
4. **Performance:** Adjust parameters like `max_distance` and `iou_threshold` to optimize tracking performance for your specific video input.

---

## **License**
This project is released under the MIT License. Feel free to use and modify it as needed.

---

For additional information or issues, feel free to contact us or raise an issue in the project repository.

