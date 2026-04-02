import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
pred = pd.read_csv('outputs/2026_predictions_context.csv')
print('Test rows:', len(pred))
for t in ['over_195','over_245','over_295']:
    y_true = pred[f'actual_{t}']
    y_prob = pred[f'prob_{t}']
    roc = roc_auc_score(y_true, y_prob)
    pr = average_precision_score(y_true, y_prob)
    print(f'{t}: ROC {roc:.4f}, PR {pr:.4f}')
