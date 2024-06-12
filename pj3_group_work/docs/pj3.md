## Project3: Drone Perspective Crowd Counting using Dual Optical Sensors

This repository contains the baseline code for the challenge titled "Drone Perspective Crowd Counting" from the GAIIChallenge hosted at [HeyWhale](https://www.heywhale.com/org/2024gaiic/competition/area/65f7b42e019d8282037f8924). The objective is to develop an algorithm that accurately counts the number of people from a drone's perspective using dual optical sensors (presumably RGB and thermal or depth cameras), addressing challenges such as varying densities, occlusions, and altitude-induced perspective changes.

### Repository Structure

- **`model/`**: Directory to store trained models. Created during setup to separate trained weights and configurations from source code.
- **`train.py`**: Script responsible for training the crowd counting model using provided or custom datasets. It implements necessary data preprocessing, model architecture, and training loop.
- **`test.py`**: Script designed to evaluate the trained model on a test dataset and generate output predictions, which are then redirected to `ans.txt` for submission or evaluation purposes.

### Quick Start Guide

#### Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/AI-FDU/ai-fdu.github.io.git
   ```

2. **Enter Project Directory**:
   ```bash
   cd ai-fdu.github.io/pj3
   ```

3. **Create Model Directory** (if not done during cloning):
   ```bash
   mkdir model
   ```

#### Training

To start the training process, execute:
```bash
python train.py
```
Ensure you have configured the dataset paths and any other necessary parameters within `train.py` according to your setup.

#### Testing

Once training is complete, you can test the model on a designated test set by running:
```bash
python test.py > ans.txt
```
This command runs the testing script and redirects its output to `ans.txt`, which typically contains the predicted counts per image or video frame.

### Experimental Requirements
- Develop a solution surpassing or equaling the baseline - 4 points
- Competition results: Determined by ranking and score - 10 points
- Effort and innovation in method: Reflected in the report - 6 points
- Extra points: Pre-presentation plan scheduled for 07/06 - 2 points (as it's hard to max out)

Final Score: max(20, s1+s2+s3+s4), total of 20 points.

Competition Deadline: June 2, 2024 ‼️
Submit A-board's best ranking and score screenshots, txt files to eLearning.

Report Deadline: June 14, 2024 ‼️
Submit B-board result screenshots, final code, project report (4 pages), PPT (if applicable) to eLearning.

### Submission
Please ensure submission in accordance with the following:
- Develop a solution surpassing or equaling the baseline.
- Provide screenshots of A-board's best ranking and score, along with txt files.
- Submit B-board result screenshots, final code, project report (4 pages), and PPT (if applicable).
- Submit all materials to eLearning.
