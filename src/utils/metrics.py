import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold

def evaluate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calculate RMSE, MAE, R2, Pearson r, and Spearman rho metrics."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    pear_r, _ = pearsonr(y_true, y_pred)
    spear_r, _ = spearmanr(y_true, y_pred)
    
    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "pearson_r": float(pear_r),
        "spearman_rho": float(spear_r)
    }

def project_disjoint_split(pairs_df: pd.DataFrame, test_ratio: float = 0.2, random_state: int = 42):
    """
    Partition dataset by unique projects to guarantee that projects in test pairs 
    never appear in training pairs (zero data leakage).
    """
    all_projects = np.array(sorted(list(set(pairs_df["project_A_id"]).union(set(pairs_df["project_B_id"])))))
    
    rng = np.random.RandomState(random_state)
    shuffled_projects = all_projects.copy()
    rng.shuffle(shuffled_projects)
    
    n_train = int(len(shuffled_projects) * (1.0 - test_ratio))
    train_proj_set = set(shuffled_projects[:n_train])
    test_proj_set = set(shuffled_projects[n_train:])
    
    is_train = pairs_df["project_A_id"].isin(train_proj_set) & pairs_df["project_B_id"].isin(train_proj_set)
    is_test = pairs_df["project_A_id"].isin(test_proj_set) & pairs_df["project_B_id"].isin(test_proj_set)
    
    train_df = pairs_df[is_train].copy()
    test_df = pairs_df[is_test].copy()
    
    return train_df, test_df

def project_disjoint_cv(pairs_df: pd.DataFrame, feature_cols: list[str], target_col: str = "expert_similarity_label", n_splits: int = 5, random_state: int = 42, model_cls=None):
    """Run N-Fold Project-Disjoint Cross Validation."""
    if model_cls is None:
        from sklearn.linear_model import Ridge
        model_cls = lambda: Ridge(alpha=1.0)
        
    all_projects = np.array(sorted(list(set(pairs_df["project_A_id"]).union(set(pairs_df["project_B_id"])))))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    fold_metrics = []
    for fold, (train_proj_idx, test_proj_idx) in enumerate(kf.split(all_projects)):
        tr_set = set(all_projects[train_proj_idx])
        te_set = set(all_projects[test_proj_idx])
        
        tr_mask = pairs_df["project_A_id"].isin(tr_set) & pairs_df["project_B_id"].isin(tr_set)
        te_mask = pairs_df["project_A_id"].isin(te_set) & pairs_df["project_B_id"].isin(te_set)
        
        X_tr = pairs_df.loc[tr_mask, feature_cols].values
        y_tr = pairs_df.loc[tr_mask, target_col].values
        X_te = pairs_df.loc[te_mask, feature_cols].values
        y_te = pairs_df.loc[te_mask, target_col].values
        
        model = model_cls()
        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)
        
        m = evaluate_regression_metrics(y_te, preds)
        m["fold"] = fold + 1
        m["n_test_pairs"] = len(y_te)
        fold_metrics.append(m)
        
    res_df = pd.DataFrame(fold_metrics)
    summary = {
        "mean_rmse": float(res_df["rmse"].mean()),
        "std_rmse": float(res_df["rmse"].std()),
        "mean_r2": float(res_df["r2"].mean()),
        "std_r2": float(res_df["r2"].std()),
        "mean_mae": float(res_df["mae"].mean()),
        "folds_detail": fold_metrics
    }
    return summary

def calculate_novelty_score(predicted_similarity_label: float | np.ndarray) -> float | np.ndarray:
    """
    Calculate Novelty Score (%):
    Novelty (%) = 100 * (1 - predicted_similarity / 4.0)
    """
    sim_norm = np.clip(np.array(predicted_similarity_label) / 4.0, 0.0, 1.0)
    novelty = (1.0 - sim_norm) * 100.0
    return novelty

def get_novelty_level(novelty_score_pct: float) -> str:
    """Classify novelty percentage into human-readable risk category."""
    if novelty_score_pct >= 75.0:
        return "High Novelty (Original Work)"
    elif novelty_score_pct >= 50.0:
        return "Moderate Novelty (Partial Overlap)"
    elif novelty_score_pct >= 25.0:
        return "Low Novelty (Significant Prior Work)"
    else:
        return "Very Low Novelty / Potential Duplicate"
