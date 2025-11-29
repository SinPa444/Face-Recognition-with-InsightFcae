# Face Recognition with InsightFace (Access Control Demo)

This repository provides a Jupyter notebook for **real-time face recognition** using [InsightFace](https://insightface.ai/) ('buffalo_s' model). Detects faces via webcam, matches against a dataset of known persons, grants/denies "access" based on cosine similarity threshold (0.4).

**Use Case**: Simple door access system – green "Access Granted" for matches, red "Access Denied" for unknowns.

## Features
- **Model**: InsightFace buffalo_s (detection + recognition embeddings)
- **Dataset**: Folder structure `./dataset/{person_name}/{images}.jpg` (e.g., ./dataset/John/img1.jpg)
- **Matching**: Cosine similarity > 0.4 → match
- **Live Webcam**: Overlay bounding box + label on video feed
- **Threshold**: Adjustable (default 0.4)

## Setup
1. **Install Dependencies** (auto-downloads models):
   ```
   pip install insightface opencv-python pillow numpy torch onnxruntime
   ```

2. **Prepare Dataset**:
   ```
   mkdir dataset
   mkdir dataset/John dataset/Jane  # Person folders
   # Add 5-10 images per person (frontal faces best)
   ```

3. **Run Notebook**: `Face Recognition.ipynb`
   - Set `dataset_path = "./dataset"`
   - Extracts embeddings (run once)
   - Webcam demo (press 'q' to quit)

## Code Overview
- **Imports & Init**: InsightFace app (buffalo_s), OpenCV webcam
- **load_dataset()**: Loads images → embeddings (512-dim normed vectors) + labels
- **recognize_face()**: Cosine sim → name/match (thresh=0.4)
- **Live Loop**: Detect faces, match, draw bbox/text (green/red)

**Example Output**:
```
Access Granted - John (sim=0.85)
Access Denied - Unknown (sim=0.32)
```

## Improvements Made
- Requirements cell added
- Typos fixed (e.g., "Intilize" → "Initialize")
- User-friendly paths (e.g., `./dataset`)
- Error handling for images/no-face

**Tips**:
- More images/person → better accuracy
- Adjust thresh (0.3-0.6)
- GPU: `ctx_id=0` (CUDA), else CPU
- Save embeddings: Add `np.savez("embeddings.npz", embeddings=dataset_embeddings, labels=dataset_labels)`

## Repo Structure
```
.
├── README.md
└── Face Recognition.ipynb  # Full demo notebook
```
*(Create `./dataset/` for persons)*

## Credits
- [InsightFace](https://insightface.ai/) (buffalo_s model)
- OpenCV, Pillow, NumPy

Fork/star/run! 🚀
