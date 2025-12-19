National Geo-AI Hackathon - Theme 1 Submission
Team: Nati-250448
Leader: Adil Mahajan
Theme: Feature Extraction from Drone Imagery

CONTENTS:
1. code/
   - preprocess_all.py: Automated pipeline to unzip data, align CRS, and generate 512x512 training chips.
   - train.py (train_model.py): U-Net + ResNet34 training loop with Dice Loss.
   - run_inference.py: Script to predict on new images and generate masks.
   - requirements.txt: List of Python libraries required.

2. outputs/

3. report/
   - Theme1_Technical_Report_Nati250448.pdf: Detailed methodology and results.

INSTRUCTIONS:
1. Install dependencies: `pip install -r code/requirements.txt`
2. Run preprocessing: `python code/preprocess_all.py`
3. Run training: `python code/train.py`
4. Run inference: `python code/run_inference.py`
