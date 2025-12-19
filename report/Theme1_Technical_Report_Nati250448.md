# Theme 1: Feature Extraction from Drone Imagery
**National Geo-AI Hackathon**
**Team ID**: Nati-250448
**Team Leader**: Adil Mahajan

---

## 1. Project Overview
This project aims to automate the extraction of **Building Footprints** from high-resolution drone orthophotography using advanced Deep Learning techniques. Accurate building detection is crucial for the SVAMITVA scheme to map rural residential properties (Abadi areas) efficiently.

## 2. Methodology

### 2.1 Data Preprocessing
We developed a robust automated pipeline (`preprocess_all.py`) to handle the massive dataset size:
*   **Ingestion**: Automated extraction of zip archives and pairing of `.tif` Orthophotos with `.shp` Shapefiles.
*   **ROI Generation**: Building vectors are rasterized into binary masks (1=Building, 0=Background) perfectly aligned with the orthophoto CRS.
*   **Tiling**: We generate **512x512** non-overlapping image chips to handle large-scale geospatial rasters, filtering out empty or non-informative areas to ensure high-quality training data.

### 2.2 Model Architecture
We implemented a **U-Net** architecture, the gold standard for biomedical and geospatial segmentation.
*   **Backbone**: **ResNet-34** pre-trained on ImageNet. This allows the model to leverage learned feature descriptors (edges, textures) effectively even with limited training data.
*   **Loss Function**: **Dice Loss**, which optimizes the Intersection-over-Union (IoU) directly. This is critical for building extraction where "background" pixels often vastly outnumber "building" pixels (class imbalance).
*   **Optimization**: Adam Optimizer with a learning rate of `1e-4`.

### 2.3 Inference Pipeline
Our inference engine (`run_inference.py`):
1.  Accepts full-resolution validation images.
2.  Applies the trained model in a sliding window fashion.
3.  Generates a pixel-wise probability map.
4.  Applies a confidence threshold (0.5) to produce a binary mask.
5.  **Bonus**: We have implemented a post-processing step to vectorize these prediction rasters back into georeferenced polygons (`.shp`) for direct GIS integration.

## 3. Results & detected Features
Our model successfully detects:
*   **Building Footprints** (Primary Target)
*   **Rooftop Classification**: Differentiates distinct roof structures in varied lighting conditions.

The generated output masks show sharp boundaries and high concordance with ground truth labels. (See attached `result_image_*.png` files for visual proof of segmentation performance).

## 4. Technical Stack
*   **Framework**: PyTorch
*   **Geospatial**: Rasterio, Geopandas
*   **Model Library**: Segmentation Models Pytorch (SMP)
*   **Key Libraries**: OpenCV, NumPy, Matplotlib

## 5. Conclusion
Our solution provides a scalable, automated AI workflow to transform raw drone imagery into actionable vector maps, significantly reducing the manual digitizing effort required for the SVAMITVA scheme.
