## Project 2: Image Classification Using CNN

### Environment Setup:
1. **Python Environment:** Ensure you have Python installed (preferably Python 3.x).
2. **Dependencies:** Install required dependencies using pip:

```bash
pip install torch torchvision numpy matplotlib tqdm
```
3. **Colab:** no need to pip


### Code Flow:

1. **Model Definition:**
- The code defines a Convolutional Neural Network (CNN) model for image classification.
- ResNet-18 architecture is used as an example, but you can analyze and discuss other CNN architectures as well.

2. **Training and Testing:**
- The code includes functions for training and testing the CNN model.
- It utilizes datasets provided by PyTorch and DataLoader for efficient data handling.

3. **Data Augmentation:**
- You need to design data augmentation techniques to improve accuracy. This may include rotations, flips, or other transformations.

4. **Optimizer and Learning Rate Scheduler:**
- Stochastic Gradient Descent (SGD) optimizer with momentum is used.
- Learning rate scheduling is implemented using ReduceLROnPlateau scheduler.

5. **Experimental Requirements:**
    - Complete the model training and validation code.
    - Design data augmentation techniques to enhance accuracy.
    - Analyze and discuss the design of other optimizers and learning rate schedules.
    - Analyze and discuss different CNN architectures.

### Filling in the Blanks:
1. **Model Training Function:**
- Complete the `model_training()` function to train the CNN model.
- Fill in the code for optimizer operations, obtaining model outputs, designing and calculating loss, and backpropagation.

2. **Model Testing Function:**
- Complete the `model_testing()` function to test the CNN model.
- Fill in the code for obtaining model outputs and calculating loss.

3. **Data Augmentation:**
- Design image augmentation techniques inside the `train_transforms` to improve model performance.

### Running the Script:
- Execute the script `main.py` from the command line: `python main.py`.
- Adjust parameters and configurations as needed for experimentation.
- For colab, just click the "run" button.

### Experimental Requirements:

1. **Complete Implementation of Image Classification (Model Training and Testing) - 4 points**
    - Ensure the code effectively trains and tests the CNN model.

2. **Design of Data Augmentation Techniques to Enhance Accuracy - 2 points**
    - Implement image augmentation techniques such as rotations, flips, etc., to improve model accuracy.

3. **Analysis and Discussion on Optimizer and Learning Rate Schedule Design - 2 points**
    - Explore and discuss alternative optimizers and learning rate schedules for training CNN models.
    - Analyze their impact on training convergence and model performance.

4. **Analysis and Discussion on Different CNN Architectures - 2 points**
    - Analyze and discuss various CNN architectures beyond ResNet-18.
    - Compare their performance, complexity, and suitability for image classification tasks.

**Total Score: 10 points**

### Submission:
Submit the code along with an experimental report (up to 4 pages, No involution) to the e-learning platform.

### Note:
- Modify and adjust the arguments, configurations, and experimental setup as necessary.
- Provide detailed explanations in the experimental report, covering code implementation, experimental setup, results, analysis, and discussions based on the specified requirements.