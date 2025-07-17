from scipy.stats import pearsonr
import numpy as np
from sklearn.metrics import confusion_matrix

def RMSE(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    return np.sqrt(mse)


def R2_Score(y_true, y_pred):
    corr_matrix = np.corrcoef(y_true, y_pred)
    corr = corr_matrix[0, 1]
    R2 = corr ** 2

    return R2


def PCC(y_true, y_pred):
    corr, _ = pearsonr(y_true, y_pred)
    return corr


def evaluate(y_true, y_pred):
    rmse = RMSE(y_true, y_pred)
    # mae = np.mean(np.abs(y_true - y_pred))
    r2 = R2_Score(y_true, y_pred)
    pcc = PCC(y_true, y_pred)

    # return rmse, mae, r2, pcc
    return rmse, r2, pcc

def calculate_csi(observed, predicted, threshold=1.0):
    """
    Calculate Critical Success Index (CSI)
    
    Parameters:
    observed (array-like): Actual rainfall values
    predicted (array-like): Predicted rainfall values
    threshold (float): Threshold for rain/no-rain classification
    
    Returns:
    float: CSI value (0 to 1, where 1 is perfect)
    """
    obs_binary = (np.array(observed) >= threshold).astype(int)
    pred_binary = (np.array(predicted) >= threshold).astype(int)
    
    cm = confusion_matrix(obs_binary, pred_binary)
    
    # Handle edge cases
    if cm.shape == (1, 1):
        return 1.0 if obs_binary[0] == pred_binary[0] else 0.0
    
    tn, fp, fn, tp = cm.ravel()
    
    if tp + fp + fn == 0:
        return 1.0
    
    return tp / (tp + fp + fn)


def evaluate_test(y_true, y_pred):
    rmse = RMSE(y_true, y_pred)
    mae = np.mean(np.abs(y_true - y_pred))
    r2 = R2_Score(y_true, y_pred)
    pcc = PCC(y_true, y_pred)
    
    threshold = 0.1
    observed = y_true
    predicted = y_pred
    print(f"CSI: {calculate_csi(observed, predicted, threshold):.3f}")
       
    threshold = 10
    print(f"CSI: {calculate_csi(observed, predicted, threshold):.3f}")

    return rmse, mae, r2, pcc
    # return rmse, r2, pcc


# Example usage
if __name__ == "__main__":
    # Sample data
    observed = [0, 2.5, 0, 1.2, 3.0, 0, 0.8, 4.1, 0, 1.5]
    predicted = [0.1, 2.0, 0.3, 1.0, 2.8, 0, 1.2, 3.9, 0.2, 1.8]
    
    threshold = 0.1
    
    print(f"CSI: {calculate_csi(observed, predicted, threshold):.3f}")