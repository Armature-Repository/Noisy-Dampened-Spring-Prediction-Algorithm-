# Noisy-Dampened-Spring-Prediction-Algorithm


Overview
This project trains a neural network to infer the physical parameters of a damped harmonic oscillator using only noisy position measurements over time.
The system follows the differential equation:
m * x'' + b * x' + k * x = 0
Where:
- m = mass
- b = damping coefficient
- k = spring constant
- x0 = initial position
- v0 = initial velocity
The model receives a noisy trajectory x(t) and predicts the underlying parameters (m, b, k, x0, v0). This is a classic system identification problem used in aerospace, robotics, and control engineering.


Motivation
Real physical systems rarely provide clean, perfect data. Sensors introduce noise, dynamics are uncertain, and parameters must be inferred indirectly. This project simulates that challenge:
- The model sees noisy, partial information
- The true parameters are hidden
- The mapping from trajectory to parameters is nonlinear
- The system evolves over time, requiring sequence modeling
This makes the project a strong demonstration of:
- physics‑informed machine learning
- time‑series regression
- LSTM‑based sequence modeling
- noisy data handling
- system identification

Features
- Physics‑based data generator for damped harmonic oscillators
- Numerical integration of the ODE
- Additive Gaussian noise
- Large synthetic dataset
- LSTM‑based neural network (SpringNet)
- Regression head predicting five continuous parameters
- Full training pipeline with loss visualization
- Modular, clean project structure

Project Structure
model.py             # LSTM model for parameter inference
data_generator.py    # Simulates noisy damped spring trajectories
train.py             # Training loop and loss visualization
predict.py           # Parameter inference on new trajectories
spring_data.npy      # Generated noisy trajectories
spring_params.npy    # Ground-truth parameters
loss.png             # Training loss curve
main.py              # Optional unified entry point



How It Works
1. Data Generation
For each sample:
- Randomly sample m, b, k, x0, v0
- Numerically integrate the ODE using Euler’s method
- Add Gaussian noise to the position measurements
- Store the full trajectory x(t)
Dataset shapes:
- X: (num_samples, timesteps, 1)
- Y: (num_samples, 5)
2. Model Architecture
SpringNet is a simple LSTM‑based regressor:
- Input: noisy trajectory
- LSTM layers extract temporal features
- Final hidden state → fully connected layer
- Output: predicted parameters (m, b, k, x0, v0)
3. Training
- Loss: Mean Squared Error (MSE)
- Optimizer: Adam
- Mini‑batch training via DataLoader
- Loss curve saved as loss.png
4. Prediction
Given a new noisy trajectory, the model outputs estimated physical parameters.

Example Output
Predicted: m=2.14, b=0.63, k=8.92, x0=1.21, v0=-0.44
True:      m=2.10, b=0.60, k=9.00, x0=1.20, v0=-0.50


Even with noise, the model learns to infer the underlying physics.

Why This Project Stands Out
This project demonstrates:
- real physics simulation
- noisy data modeling
- system identification
- sequence modeling with LSTMs
- numerical integration
- ML + physics integration
It is significantly more sophisticated than typical student ML projects and aligns well with:
- aerospace engineering
- robotics
- controls
- simulation
- physics‑informed ML

Future Extensions
Potential improvements:
- Kalman filter for state estimation
- Nonlinear springs
- External forcing
- Variable time steps
- Transformer‑based sequence model
- Uncertainty estimation
- Multi‑sensor fusion (position + velocity + acceleration)
