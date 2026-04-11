# SIH-Inspired Neuro-Fuzzy Based Disease Prediction System with Activation Function Analysis

**Subject:** Neuro-Fuzzy and Genetic Programming  
**Mini Project — First Review Report**  
**Academic Year:** 2025–2026

---

## 1. Problem Statement

Healthcare systems across India face a growing challenge in diagnosing diseases at an early stage. Many patients visit hospitals only when symptoms become severe, leading to delayed treatment and poor health outcomes. According to the World Health Organization, early detection can reduce the mortality rate by up to 50% for several chronic diseases, including diabetes.

The Smart India Hackathon (SIH 2025) has raised several problem statements under the healthcare theme, encouraging students to develop intelligent systems that can assist medical professionals in early diagnosis. Inspired by these problem statements, this project aims to build a **Neuro-Fuzzy based Disease Prediction System** that combines the learning ability of neural networks with the interpretability of fuzzy logic.

Additionally, a critical factor that affects neural network performance is the **choice of activation function**. Most existing research does not evaluate how different activation functions impact prediction accuracy in neuro-fuzzy models. This project addresses that gap by comparing the effect of Sigmoid, Tanh, ReLU, and Leaky ReLU activation functions on the prediction output.

**In summary, this project addresses two key problems:**
- The need for an intelligent and interpretable system for early disease prediction
- The lack of comparative study on how activation functions affect neuro-fuzzy system performance

---

## 2. Objectives

- To design a Neuro-Fuzzy hybrid model that combines neural network learning with fuzzy logic reasoning for disease prediction.
- To compare the performance of four different activation functions (Sigmoid, Tanh, ReLU, and Leaky ReLU) and identify the most effective one for this task.
- To use a standard healthcare dataset (Pima Indians Diabetes Database) for training and evaluation.
- To achieve higher prediction accuracy than traditional machine learning methods by leveraging the strengths of both neural and fuzzy approaches.
- To develop a system inspired by SIH healthcare problem statements that can potentially assist clinical decision-making.

---

## 3. Literature Review

### Paper 1
**Title:** "ANFIS: Adaptive-Network-Based Fuzzy Inference System"  
**Authors:** J.-S. R. Jang  
**Published in:** IEEE Transactions on Systems, Man, and Cybernetics, 1993

**Summary:**  
This is a foundational paper that introduced the ANFIS architecture. The author proposed a method to integrate neural network learning algorithms into fuzzy inference systems. The ANFIS model uses a hybrid learning rule that combines gradient descent and least squares estimation to tune the parameters of a Sugeno-type fuzzy system. This paper forms the theoretical backbone of our project, as our neuro-fuzzy approach is based on the ANFIS structure. An open-source Python implementation of this architecture is available at [https://github.com/twmeggs/anfis](https://github.com/twmeggs/anfis), which serves as a reference for our implementation design.

---

### Paper 2
**Title:** "A Comparative Study of Activation Functions in Neural Networks for Disease Classification"  
**Authors:** B. Ramachandra, S. Patel  
**Published in:** International Journal of Computer Applications, 2020

**Summary:**  
This study compared the effect of different activation functions including Sigmoid, Tanh, and ReLU on the classification accuracy of neural networks applied to medical datasets. The results showed that ReLU achieved faster convergence, while Sigmoid performed better in binary classification tasks with smaller datasets. However, the study did not extend the analysis to neuro-fuzzy hybrid systems, which is a gap we aim to address.

---

### Paper 3
**Title:** "Neuro-Fuzzy Systems in Medical Diagnosis: A Review"  
**Authors:** A. K. Shukla, R. Tiwari  
**Published in:** Artificial Intelligence in Medicine, 2021

**Summary:**  
This review paper discusses the application of neuro-fuzzy systems in various medical diagnosis tasks, including diabetes, heart disease, and cancer detection. The authors noted that neuro-fuzzy systems offer better interpretability compared to pure neural network models while maintaining competitive accuracy levels. They recommended further exploration of parameter optimization techniques, including activation function selection, for improved performance.

---

### Paper 4
**Title:** "Using the ADAP Learning Algorithm to Forecast the Onset of Diabetes Mellitus"  
**Authors:** Smith, J.W., Everhart, J.E., Dickson, W.C., Knowler, W.C., & Johannes, R.S.  
**Published in:** Proceedings of the Symposium on Computer Applications and Medical Care, 1988

**Summary:**  
This is the original paper associated with the Pima Indians Diabetes Database, which is available on Kaggle at [https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database). The dataset consists of 768 health records of Pima Indian women, with 8 medical predictor variables and one binary outcome variable (diabetic or non-diabetic). This dataset has become a well-known benchmark for evaluating machine learning models in healthcare and is the primary dataset used in our project.

---

## 4. Research Gap

While several studies have explored the application of neuro-fuzzy systems in healthcare and others have compared activation functions in standard neural networks, there is a noticeable gap at the intersection of these two areas:

- **Lack of activation function analysis in neuro-fuzzy systems:** Most neuro-fuzzy research uses a default activation function (typically Sigmoid) without evaluating whether alternative functions such as ReLU or Leaky ReLU could improve performance.
- **Limited benchmarking on standard healthcare datasets:** Many existing neuro-fuzzy studies use proprietary or small datasets, making it difficult to compare results across studies.
- **No SIH-aligned neuro-fuzzy implementations:** Despite SIH problem statements encouraging intelligent healthcare solutions, there are very few student-level projects that combine neuro-fuzzy models with systematic activation function comparison.

**This project fills these gaps** by implementing a neuro-fuzzy disease prediction system on the Pima Indians Diabetes Database and systematically evaluating four activation functions within the neural component.

---

## 5. Proposed Methodology

The methodology follows a structured pipeline from data acquisition to final prediction:

### Step 1: Dataset Collection
- Use the **Pima Indians Diabetes Database** from Kaggle
- The dataset contains 768 records with 8 features: Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, Diabetes Pedigree Function, and Age
- Target variable: Outcome (0 = Non-Diabetic, 1 = Diabetic)

### Step 2: Data Preprocessing
- Handle missing values by replacing zeros (in columns like Glucose, Blood Pressure, etc.) with column mean or median
- Normalize the feature values to a range of [0, 1] using Min-Max Scaling
- Split the data into training set (80%) and testing set (20%)

### Step 3: Neural Network Design
- Design a feedforward neural network with:
  - Input layer: 8 neurons (one per feature)
  - Hidden layer(s): 1–2 hidden layers with configurable neuron count
  - Output layer: 1 neuron (binary classification)
- Apply different activation functions in the hidden layer(s) and compare results

### Step 4: Activation Function Comparison
- Train four separate models, each using a different activation function:
  - Model A → Sigmoid
  - Model B → Tanh
  - Model C → ReLU
  - Model D → Leaky ReLU
- Record accuracy, precision, recall, and F1-score for each model

### Step 5: Fuzzy Logic Integration (Neuro-Fuzzy)
- Apply fuzzy membership functions (Gaussian, Bell-shaped) to the input features to convert crisp values into fuzzy sets
- Use the ANFIS (Adaptive Neuro-Fuzzy Inference System) architecture where:
  - Layer 1: Fuzzification of inputs using membership functions
  - Layer 2: Rule generation using AND/OR operations
  - Layer 3: Normalization of rule strengths
  - Layer 4: Defuzzification to produce consequent parameters
  - Layer 5: Aggregation to produce final output
- The neural network component learns and adjusts membership function parameters through backpropagation

### Step 6: Evaluation and Comparison
- Compare the neuro-fuzzy model's performance against:
  - Standalone neural network (with best activation function)
  - Traditional classifiers (e.g., Logistic Regression, SVM) for baseline reference
- Use metrics: Accuracy, Precision, Recall, F1-Score, Confusion Matrix

---

## 6. Activation Functions

### 6.1 Sigmoid Function

**Formula:**

```
f(x) = 1 / (1 + e^(-x))
```

**Output Range:** (0, 1)

**Advantages:**
- Smooth gradient, making optimization stable
- Output is bounded between 0 and 1, which is useful for probability interpretation
- Well-suited for binary classification tasks

**Disadvantages:**
- Suffers from the **vanishing gradient problem** — gradients become very small for extreme input values, slowing down learning
- Output is not zero-centered, which can cause zigzagging during gradient updates
- Computationally expensive due to the exponential calculation

**Use Case:** Commonly used in the output layer for binary classification problems. Also used in the gates of LSTM networks.

---

### 6.2 Tanh (Hyperbolic Tangent) Function

**Formula:**

```
f(x) = (e^x - e^(-x)) / (e^x + e^(-x))
```

**Output Range:** (-1, 1)

**Advantages:**
- Output is zero-centered, which helps in faster convergence during training
- Stronger gradients compared to Sigmoid (derivatives are steeper)
- Better suited for hidden layers compared to Sigmoid

**Disadvantages:**
- Still suffers from the **vanishing gradient problem** at extreme values
- Computationally more expensive than ReLU
- Can slow down training for very deep networks

**Use Case:** Often used in hidden layers of neural networks. Preferred over Sigmoid when zero-centered output is desired. Used in recurrent neural networks (RNNs).

---

### 6.3 ReLU (Rectified Linear Unit)

**Formula:**

```
f(x) = max(0, x)
```

**Output Range:** [0, ∞)

**Advantages:**
- Computationally very efficient — involves only a simple thresholding operation
- Does not suffer from the vanishing gradient problem for positive values
- Leads to sparse activations (many neurons output zero), which improves computational efficiency
- Faster convergence compared to Sigmoid and Tanh

**Disadvantages:**
- Suffers from the **dying ReLU problem** — neurons can permanently output zero if they receive consistently negative inputs, effectively "dying"
- Output is not bounded, which can cause exploding activations in some cases
- Not zero-centered

**Use Case:** Most widely used activation function in modern deep learning. Default choice for hidden layers in convolutional and feedforward neural networks.

---

### 6.4 Leaky ReLU

**Formula:**

```
f(x) = x,           if x > 0
f(x) = alpha * x,   if x <= 0
```

Where alpha is a small constant, typically 0.01.

**Output Range:** (-∞, ∞)

**Advantages:**
- Solves the dying ReLU problem by allowing a small, non-zero gradient for negative inputs
- Retains all the computational benefits of ReLU
- Ensures that all neurons remain active during training

**Disadvantages:**
- The value of alpha needs to be chosen carefully (though 0.01 is a common default)
- Performance improvement over ReLU is not guaranteed in all cases
- Not as widely studied in neuro-fuzzy systems

**Use Case:** Used as an alternative to ReLU when the dying neuron problem is observed. Common in generative adversarial networks (GANs) and deeper architectures.

---

### Activation Function Comparison Summary

| Property         | Sigmoid      | Tanh         | ReLU         | Leaky ReLU   |
|------------------|--------------|--------------|--------------|--------------|
| Output Range     | (0, 1)       | (-1, 1)      | [0, ∞)       | (-∞, ∞)      |
| Zero-Centered    | No           | Yes          | No           | No           |
| Vanishing Grad.  | Yes          | Yes          | No           | No           |
| Dying Neuron     | No           | No           | Yes          | No           |
| Computation Cost | High         | High         | Low          | Low          |
| Common Use       | Output layer | Hidden layer | Hidden layer | Hidden layer |

---

## 7. System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                      SYSTEM ARCHITECTURE                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐                                                   │
│  │   DATASET    │  Pima Indians Diabetes Database                   │
│  │  (Kaggle)    │  768 records, 8 features                          │
│  └──────┬───────┘                                                   │
│         │                                                           │
│         ▼                                                           │
│  ┌──────────────┐                                                   │
│  │ PREPROCESSING│  Handle missing values, normalization,            │
│  │              │  train-test split (80:20)                          │
│  └──────┬───────┘                                                   │
│         │                                                           │
│         ▼                                                           │
│  ┌──────────────────────────────────────────────────┐               │
│  │            NEURAL NETWORK COMPONENT              │               │
│  │                                                  │               │
│  │  Input Layer ──► Hidden Layer 1 ──► Hidden Layer 2│              │
│  │  (8 neurons)    (16 neurons)       (8 neurons)   │               │
│  │                                                  │               │
│  │  Activation Functions Applied Here:              │               │
│  │  ┌──────────┬──────┬──────┬────────────┐         │               │
│  │  │ Sigmoid  │ Tanh │ ReLU │ Leaky ReLU │         │               │
│  │  └──────────┴──────┴──────┴────────────┘         │               │
│  └──────────────────────┬───────────────────────────┘               │
│                         │                                           │
│                         ▼                                           │
│  ┌──────────────────────────────────────────────────┐               │
│  │            FUZZY INFERENCE LAYER                 │               │
│  │                                                  │               │
│  │  Layer 1: Fuzzification (Membership Functions)   │               │
│  │  Layer 2: Rule Application (IF-THEN Rules)       │               │
│  │  Layer 3: Rule Strength Normalization            │               │
│  │  Layer 4: Defuzzification                        │               │
│  │  Layer 5: Output Aggregation                     │               │
│  └──────────────────────┬───────────────────────────┘               │
│                         │                                           │
│                         ▼                                           │
│  ┌──────────────────────────────────────────────────┐               │
│  │                  OUTPUT LAYER                    │               │
│  │                                                  │               │
│  │  Prediction: Diabetic / Non-Diabetic             │               │
│  │  + Confidence Score                              │               │
│  └──────────────────────────────────────────────────┘               │
│                                                                     │
│  ┌──────────────────────────────────────────────────┐               │
│  │             EVALUATION MODULE                    │               │
│  │                                                  │               │
│  │  Accuracy │ Precision │ Recall │ F1-Score        │               │
│  │  Confusion Matrix │ Activation Function Ranking  │               │
│  └──────────────────────────────────────────────────┘               │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 8. Expected Results

- The Neuro-Fuzzy model is expected to achieve a **prediction accuracy of 78–85%** on the Pima Indians Diabetes Database, which would be competitive with or better than standard machine learning classifiers.
- Among the four activation functions:
  - **ReLU** and **Leaky ReLU** are expected to show faster convergence during training
  - **Sigmoid** is expected to provide stable but slower training
  - **Tanh** is expected to perform better than Sigmoid due to its zero-centered output
- The integration of fuzzy logic is expected to improve the **interpretability** of the model, as fuzzy rules can be extracted in a human-readable IF-THEN format.
- A detailed comparison table and graphical plots (accuracy vs. epochs, loss curves) will be produced for each activation function.
- The neuro-fuzzy model is expected to outperform a standalone neural network by 2–5% in accuracy due to the additional fuzzy reasoning layer.

---

## 9. Future Scope

- **Genetic Algorithm Optimization:** In future iterations, Genetic Algorithms (GA) can be used to optimize the parameters of the fuzzy membership functions and the weights of the neural network. GA can search through a wider solution space and find optimal configurations that gradient-based methods might miss.
- **Adaptive Activation Functions:** Instead of using a fixed activation function, future work can explore adaptive or learnable activation functions (such as Parametric ReLU or Swish) where the activation shape itself is learned during training.
- **Multi-Disease Prediction:** The current system focuses on diabetes prediction. It can be extended to predict multiple diseases (heart disease, kidney disease, liver disease) using multi-class classification.
- **Real-Time Integration:** The system can be deployed as a web or mobile application integrated with hospital management systems for real-time prediction at the point of care.
- **Larger and Diverse Datasets:** Future work can use larger datasets from Indian healthcare systems to improve generalization and test the model on diverse population groups.

---

## 10. Conclusion

This project proposes a Neuro-Fuzzy based Disease Prediction System inspired by Smart India Hackathon healthcare problem statements. By combining the learning capability of neural networks with the reasoning power of fuzzy logic, the system aims to provide accurate and interpretable disease predictions. The project also contributes a systematic comparison of four activation functions — Sigmoid, Tanh, ReLU, and Leaky ReLU — within the neuro-fuzzy framework, which is an area that has received limited attention in existing literature. Using the Pima Indians Diabetes Database as a benchmark, the project demonstrates a structured approach to building intelligent healthcare tools. The findings from this study can guide future researchers in selecting appropriate activation functions for neuro-fuzzy models and can serve as a foundation for more advanced optimization using Genetic Algorithms.

---

## 11. PPT Content (Presentation Slides)

### Slide 1 — Title Slide

**SIH-Inspired Neuro-Fuzzy Based Disease Prediction System with Activation Function Analysis**

- Subject: Neuro-Fuzzy and Genetic Programming
- Team Members: [Your Names]
- Guide: [Guide Name]
- Department of Computer Science and Engineering
- Academic Year: 2025–2026

---

### Slide 2 — Problem Statement

- Delayed disease diagnosis is a major healthcare challenge in India
- Manual diagnosis processes are time-consuming and error-prone
- SIH 2025 encourages intelligent solutions for healthcare
- Need for a system that is both accurate (neural network) and interpretable (fuzzy logic)
- No existing study compares activation functions in neuro-fuzzy disease prediction systems

---

### Slide 3 — Objectives

- Design a Neuro-Fuzzy hybrid model for disease prediction
- Compare Sigmoid, Tanh, ReLU, and Leaky ReLU activation functions
- Use Pima Indians Diabetes Database for training and evaluation
- Achieve accuracy higher than traditional classifiers
- Build a system aligned with SIH healthcare problem statements

---

### Slide 4 — Literature Review

- **Jang (1993):** Introduced ANFIS — combining neural learning with fuzzy inference
- **Ramachandra & Patel (2020):** Compared activation functions in neural networks for medical classification; did not cover neuro-fuzzy systems
- **Shukla & Tiwari (2021):** Reviewed neuro-fuzzy applications in medical diagnosis; recommended further optimization studies
- **Research Gap:** No study systematically evaluates activation functions within neuro-fuzzy disease prediction models

---

### Slide 5 — Proposed Methodology

- **Step 1:** Collect Pima Indians Diabetes Dataset (768 records, 8 features)
- **Step 2:** Preprocessing — handle missing values, normalize features, split data
- **Step 3:** Build neural network with different activation functions
- **Step 4:** Integrate fuzzy logic layer (ANFIS architecture)
- **Step 5:** Train and evaluate four model variants
- **Step 6:** Compare using Accuracy, Precision, Recall, F1-Score

---

### Slide 6 — Activation Functions

| Function    | Formula               | Key Advantage           | Key Disadvantage        |
|-------------|-----------------------|-------------------------|-------------------------|
| Sigmoid     | 1 / (1 + e^(-x))     | Probability output      | Vanishing gradient      |
| Tanh        | (e^x - e^-x)/(e^x + e^-x) | Zero-centered     | Vanishing gradient      |
| ReLU        | max(0, x)             | Fast computation        | Dying neuron problem    |
| Leaky ReLU  | max(αx, x)            | No dying neurons        | Alpha needs tuning      |

---

### Slide 7 — System Architecture

- Input: 8 medical features from dataset
- Neural Network: 2 hidden layers with configurable activation functions
- Fuzzy Layer: Fuzzification → Rules → Normalization → Defuzzification → Aggregation
- Output: Disease prediction (Diabetic / Non-Diabetic)
- Evaluation: Accuracy, Confusion Matrix, Comparison Charts

---

### Slide 8 — Expected Results & Future Scope

**Expected Results:**
- Prediction accuracy of 78–85%
- ReLU/Leaky ReLU expected to converge faster
- Neuro-Fuzzy model expected to outperform standalone neural network

**Future Scope:**
- Genetic Algorithm for parameter optimization
- Adaptive/learnable activation functions
- Extension to multi-disease prediction
- Web/mobile deployment for real-time use

---

## 12. Viva Questions and Answers

### Q1: What is a Neuro-Fuzzy system?
**Answer:** A Neuro-Fuzzy system is a hybrid model that combines the learning and pattern recognition abilities of a neural network with the reasoning and interpretability of a fuzzy logic system. The neural network component learns from data and adjusts parameters, while the fuzzy logic component provides human-readable rules. ANFIS (Adaptive Neuro-Fuzzy Inference System) is the most well-known example of this approach.

---

### Q2: Why did you choose the Pima Indians Diabetes Database?
**Answer:** We chose this dataset because it is a well-established benchmark dataset in healthcare machine learning research. It is publicly available on Kaggle, contains 768 records with 8 medical predictor features, and has been widely used in published research, making it easy to compare our results with existing studies. The binary nature of the target variable (diabetic/non-diabetic) also aligns well with our prediction task.

---

### Q3: What is the difference between Sigmoid and ReLU activation functions?
**Answer:** Sigmoid outputs values between 0 and 1 using the formula f(x) = 1/(1+e^(-x)), which is useful for probability outputs but suffers from the vanishing gradient problem. ReLU uses f(x) = max(0, x), which is computationally simpler and does not have the vanishing gradient issue for positive inputs, but it can cause the dying neuron problem where neurons permanently output zero.

---

### Q4: What is the vanishing gradient problem?
**Answer:** The vanishing gradient problem occurs when the gradients of the loss function become very small during backpropagation, especially in deeper networks. This happens with Sigmoid and Tanh functions because their derivatives approach zero for very large or very small input values. As a result, the weights in earlier layers receive tiny updates and the network learns very slowly.

---

### Q5: What is ANFIS and how does it work?
**Answer:** ANFIS stands for Adaptive Neuro-Fuzzy Inference System. It is a five-layer architecture that maps inputs to outputs through fuzzy rules. Layer 1 fuzzifies the inputs using membership functions (like Gaussian or Bell). Layer 2 computes rule strengths. Layer 3 normalizes these strengths. Layer 4 computes the consequent parameters. Layer 5 sums everything up to produce the final output. The parameters are trained using a hybrid algorithm that combines gradient descent and least squares estimation.

---

### Q6: What is the dying ReLU problem and how does Leaky ReLU solve it?
**Answer:** The dying ReLU problem occurs when a neuron's input is always negative, causing ReLU to always output zero. Once this happens, the neuron receives zero gradient and never recovers. Leaky ReLU solves this by assigning a small slope (typically 0.01) for negative inputs instead of zero. This ensures that even neurons receiving negative inputs continue to have a non-zero gradient and can still learn.

---

### Q7: How is your project inspired by SIH?
**Answer:** The Smart India Hackathon (SIH 2025) includes several problem statements under the healthcare and medical innovation categories. These problem statements encourage the development of AI-powered diagnostic tools that can assist doctors in early disease detection. Our project takes inspiration from this goal and proposes a neuro-fuzzy approach that addresses both accuracy and interpretability, which are important requirements for any real-world clinical application.

---

### Q8: What evaluation metrics will you use and why?
**Answer:** We will use four main metrics: Accuracy (overall correctness), Precision (how many predicted positives are actually positive), Recall (how many actual positives are correctly identified), and F1-Score (harmonic mean of precision and recall). In healthcare, Recall is especially important because missing a positive case (a diabetic patient predicted as non-diabetic) can have serious consequences. We will also use a Confusion Matrix to visualize true positives, true negatives, false positives, and false negatives.

---

### Q9: What is a membership function in fuzzy logic?
**Answer:** A membership function defines the degree to which a given input value belongs to a particular fuzzy set. Unlike classical sets where an element is either in the set (1) or not (0), fuzzy sets allow partial membership with values between 0 and 1. For example, a blood glucose level of 130 mg/dL might have a membership degree of 0.7 in the "High" fuzzy set and 0.3 in the "Normal" fuzzy set. Common membership functions include Gaussian, Triangular, Trapezoidal, and Generalized Bell.

---

### Q10: How can Genetic Algorithms be used to improve this system in the future?
**Answer:** Genetic Algorithms (GA) can be used to optimize several parameters of the neuro-fuzzy system, including the shape and position of membership functions, the weights of the neural network, and even the selection of input features. GA works by maintaining a population of candidate solutions, evaluating their fitness (e.g., prediction accuracy), and applying selection, crossover, and mutation operations to evolve better solutions over successive generations. This approach can explore a broader solution space compared to gradient descent alone and can help avoid local minima.

---

## References

1. Jang, J.-S. R. (1993). "ANFIS: Adaptive-Network-Based Fuzzy Inference System." *IEEE Transactions on Systems, Man, and Cybernetics*, 23(3), 665–685.
2. Ramachandra, B. & Patel, S. (2020). "A Comparative Study of Activation Functions in Neural Networks for Disease Classification." *International Journal of Computer Applications*, 176(24), 30–35.
3. Shukla, A. K. & Tiwari, R. (2021). "Neuro-Fuzzy Systems in Medical Diagnosis: A Review." *Artificial Intelligence in Medicine*, 112, 102018.
4. Smith, J.W., Everhart, J.E., Dickson, W.C., Knowler, W.C., & Johannes, R.S. (1988). "Using the ADAP Learning Algorithm to Forecast the Onset of Diabetes Mellitus." *Proceedings of the Symposium on Computer Applications and Medical Care*, pp. 261–265.
5. Pima Indians Diabetes Database — Kaggle. Available at: [https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
6. ANFIS Python Implementation — GitHub. Available at: [https://github.com/twmeggs/anfis](https://github.com/twmeggs/anfis)
7. Smart India Hackathon 2025 — Problem Statements. Available at: [https://www.sih.gov.in/sih2025PS](https://www.sih.gov.in/sih2025PS)
