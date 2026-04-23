import torch
import numpy as np

# Simulate what train_gnn_ppo does
y_true = np.random.randn(4096)
y_pred = np.zeros(4096)
y_var = np.var(y_true)
explained_var = np.nan if y_var == 0 else 1.0 - np.var(y_true - y_pred) / y_var
print(f"ev with zeros: {explained_var:.4f}")

y_pred = np.ones(4096) * 5.0
explained_var = np.nan if y_var == 0 else 1.0 - np.var(y_true - y_pred) / y_var
print(f"ev with constant 5: {explained_var:.4f}")

y_pred = y_true + np.random.randn(4096)*0.1
explained_var = np.nan if y_var == 0 else 1.0 - np.var(y_true - y_pred) / y_var
print(f"ev with good pred: {explained_var:.4f}")

