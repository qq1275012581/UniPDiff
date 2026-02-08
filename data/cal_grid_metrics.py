import numpy as np

def era5_ts_scores(predict: np.ndarray,
                   truth: np.ndarray,
                   variable_names: list,
                   threshold: float = [0.1, 1.0, 4.0, 13.0, 25.0],
                   predict_avg: np.ndarray = None,
                   truth_avg: np.ndarray = None,
                   lat: np.ndarray = None) -> dict:
    """
    Compute Threat Score (TS) for ERA5 gridded data, supporting both 4D and 5D inputs.

    Parameters
    ----------
    predict : np.ndarray
        Forecasted precipitation or other variables. Shape can be:
        - (B, V, H, W): single timestep
        - (B, T, V, H, W): multiple timesteps

    truth : np.ndarray
        Ground truth data, same shape as predict.

    variable_names : list of str
        Names of the variables, should match the V dimension.

    thresholds : list of float, optional
        Predefined precipitation thresholds for TS calculation.
        Default: [0.1, 1.0, 4.0, 13.0, 25.0]

    Returns
    -------
    ts_dict : dict
        Dictionary containing TS scores. Each key is formatted as:
        - 'TS_<threshold>_<variable_name>'
        The value is either a float (if no time dimension) or a numpy array of shape (T,).
    """
    assert predict.shape == truth.shape, "Shapes of predict and truth must match."
    assert predict.ndim in (4, 5), "Input must be either 4D (B, V, H, W) or 5D (B, T, V, H, W)."

    # Standardize to 5D shape: (B, T, V, H, W)
    if predict.ndim == 4:
        predict = predict[:, None, ...]
        truth = truth[:, None, ...]
        is_temporal = False
    else:
        is_temporal = True

    B, T, V, H, W = predict.shape
    assert len(variable_names) == V, "Length of variable_names must match V dimension."

    ts_dict = {}

    for v in range(V):
        var_name = variable_names[v]
        for th in threshold:
            ts_scores = []
            for t in range(T):
                pred_mask = predict[:, t, v] >= th
                truth_mask = truth[:, t, v] >= th

                A = np.sum(pred_mask & truth_mask)        # Hits: forecast and truth both exceed threshold
                B_ = np.sum(pred_mask & ~truth_mask)       # False alarms: forecast exceeds, truth does not
                C = np.sum(~pred_mask & truth_mask)        # Misses: truth exceeds, forecast does not

                denom = A + B_ + C
                ts = 0.0 if denom == 0 else A / denom       # Avoid division by zero
                ts_scores.append(ts)

            ts_key = f"TS_{th}_{var_name}"
            ts_dict[ts_key] = np.array(ts_scores) if is_temporal else ts_scores[0]

    return ts_dict


def era5_cri(predict: np.ndarray,
             truth: np.ndarray,
             variable_names: list,
             threshold: float = 0.1,
             predict_avg: np.ndarray = None,
             truth_avg: np.ndarray = None,
             lat: np.ndarray = None) -> dict:
    """
    Compute the Clear-Rain Index (CRI) for ERA5 gridded data, indicating binary classification accuracy
    between forecast and observation (rain vs. no-rain).

    Supports both single timestep and temporal inputs. Additional arguments `predict_avg` and `truth_avg`
    are reserved for interface consistency but not used in this function.

    Parameters
    ----------
    predict : np.ndarray
        Forecasted data. Shape must be:
        - (B, V, H, W) for single timestep
        - (B, T, V, H, W) for multiple timesteps

    truth : np.ndarray
        Ground truth data, same shape as predict.

    variable_names : list of str
        Names of the variables (length V).

    threshold : float, default=0.1
        Threshold for binarization. Values >= threshold are considered "rain".

    predict_avg : np.ndarray, optional
        Average forecast values over historical data. Not used.

    truth_avg : np.ndarray, optional
        Average truth values over historical data. Not used.

    Returns
    -------
    cri_dict : dict
        Dictionary with CRI scores per variable. Each value is:
        - A float if input has no time dimension.
        - A numpy array of shape (T,) if input has time dimension.
        Example:
            {
                'CRI_Precip': 0.87,
                'CRI_Temp': np.array([...])
            }
    """
    assert predict.shape == truth.shape, "predict and truth must have the same shape."
    assert predict.ndim in (4, 5), "Input must be 4D or 5D."

    # Convert to shape (B, T, V, H, W)
    if predict.ndim == 4:
        predict = predict[:, None, ...]
        truth = truth[:, None, ...]
        is_temporal = False
    else:
        is_temporal = True

    B, T, V, H, W = predict.shape
    assert len(variable_names) == V, "Length of variable_names must match V dimension."

    cri_dict = {}

    for v in range(V):
        var_name = variable_names[v]
        cri_list = []

        for t in range(T):
            pred_t = predict[:, t, v]
            truth_t = truth[:, t, v]

            mask = (~np.isnan(truth_t)) & (truth_t <= 10000)
            if mask.sum() == 0:
                cri = 0.0
            else:
                pred_bin = (pred_t[mask] >= threshold)
                truth_bin = (truth_t[mask] >= threshold)
                cri = np.mean(pred_bin == truth_bin)

            cri_list.append(cri)

        cri_dict[f'CRI_{var_name}'] = np.array(cri_list) if is_temporal else cri_list[0]

    return cri_dict

def era5_lat_weighted_rmse(predict: np.ndarray,
              truth: np.ndarray,
              variable_names: list,
              threshold: float = 0.1,
              predict_avg: np.ndarray = None,
              truth_avg: np.ndarray = None,
              lat: np.ndarray = None) -> dict:
    """
    Compute Root Mean Square Error (RMSE) for ERA5 gridded data.
    Supports both single timestep and temporal inputs.

    Parameters
    ----------
    predict : np.ndarray
        Forecasted data. Shape:
        - (B, V, H, W) for single timestep
        - (B, T, V, H, W) for multiple timesteps

    truth : np.ndarray
        Ground truth data, same shape as predict.

    variable_names : list of str
        Names of variables (length must match V).

    threshold : float, optional
        Not used. Included for interface consistency.

    predict_avg : np.ndarray, optional
        Historical average forecast (not used here).

    truth_avg : np.ndarray, optional
        Historical average truth (not used here).

    Returns
    -------
    rmse_dict : dict
        Dictionary of RMSE scores for each variable.
        - If input is 4D: float
        - If input is 5D: np.ndarray of shape (T,)
        Example:
            {
                'RMSE_Precip': 0.35,
                'RMSE_Temp': np.array([0.31, 0.32, ...])
            }
    """
    assert predict.shape == truth.shape, "predict and truth must have the same shape."
    assert predict.ndim in (4, 5), "Input must be 4D or 5D."

    if predict.ndim == 4:
        predict = predict[:, None, ...]  # (B, 1, V, H, W)
        truth = truth[:, None, ...]
        is_temporal = False
    else:
        is_temporal = True

    B, T, V, H, W = predict.shape
    assert len(variable_names) == V, "Length of variable_names must match V dimension."

    w_lat = np.cos(np.deg2rad(lat))
    w_lat = w_lat / w_lat.mean()  # (H, )
    w_lat = w_lat[np.newaxis, :, np.newaxis]  # [1, H, 1]

    rmse_dict = {}

    for v in range(V):
        var_name = variable_names[v]
        rmse_list = []

        for t in range(T):
            pred_t = predict[:, t, v]
            truth_t = truth[:, t, v]
            rmse = np.sqrt(np.mean(w_lat * (pred_t - truth_t) ** 2))
            rmse_list.append(rmse)

        rmse_dict[f'Lat_RMSE_{var_name}'] = np.array(rmse_list) if is_temporal else rmse_list[0]

    return rmse_dict

def era5_lat_weighted_acc(predict: np.ndarray,
             truth: np.ndarray,
             variable_names: list,
             threshold: float = 0.1,
             predict_avg: np.ndarray = None,
             truth_avg: np.ndarray = None,
             lat: np.ndarray = None) -> dict:
    """
    Compute Anomaly Correlation Coefficient (ACC) for ERA5 gridded data.
    Measures spatial pattern correlation between forecast and truth anomalies.

    Supports both single timestep and temporal inputs.

    Parameters
    ----------
    predict : np.ndarray
        Forecasted data. Shape:
        - (B, V, H, W) for single timestep
        - (B, T, V, H, W) for multiple timesteps

    truth : np.ndarray
        Ground truth data, same shape as predict.

    variable_names : list of str
        Names of variables (length must match V dimension).

    threshold : float, optional
        Not used here. Included for interface consistency.

    predict_avg : np.ndarray
        Climatological mean forecast. Shape must match (T, V, H, W) or (V, H, W).

    truth_avg : np.ndarray
        Climatological mean truth. Shape must match (T, V, H, W) or (V, H, W).

    Returns
    -------
    acc_dict : dict
        Dictionary with ACC scores per variable.
        - If input is 4D: float
        - If input is 5D: np.ndarray of shape (T,)
        Example:
            {
                'ACC_Precip': 0.61,
                'ACC_Temp': np.array([0.59, 0.63, ...])
            }
    """
    assert predict.shape == truth.shape, "predict and truth must have the same shape."
    assert predict.ndim in (4, 5), "Input must be 4D or 5D."
    assert predict_avg is not None and truth_avg is not None, "predict_avg and truth_avg are required for ACC."

    # Standardize to shape (B, T, V, H, W)
    if predict.ndim == 4:
        predict = predict[:, None, ...]
        truth = truth[:, None, ...]
        predict_avg = predict_avg[None]
        truth_avg = truth_avg[None]
        is_temporal = False
    else:
        is_temporal = True

    B, T, V, H, W = predict.shape
    assert len(variable_names) == V, "Length of variable_names must match V dimension."
    assert predict_avg.shape == (T, V, H, W), "predict_avg shape must match (T, V, H, W)."
    assert truth_avg.shape == (T, V, H, W), "truth_avg shape must match (T, V, H, W)."

    w_lat = np.cos(np.deg2rad(lat))
    w_lat = w_lat / w_lat.mean()  # (H, )
    w_lat = w_lat[np.newaxis, :, np.newaxis]  # [1, H, 1]

    acc_dict = {}

    for v in range(V):
        var_name = variable_names[v]
        acc_list = []

        for t in range(T):
            pred_anom = predict[:, t, v] - predict_avg[t, v]
            truth_anom = truth[:, t, v] - truth_avg[t, v]

            numerator = np.sum(w_lat * pred_anom * truth_anom)
            denominator = np.sqrt(np.sum(w_lat * pred_anom ** 2) * np.sum(w_lat * truth_anom ** 2))

            acc = 0.0 if denominator == 0 else numerator / denominator
            acc_list.append(acc)

        acc_dict[f'Lat_ACC_{var_name}'] = np.array(acc_list) if is_temporal else acc_list[0]

    return acc_dict

def era5_rmse(predict: np.ndarray,
              truth: np.ndarray,
              variable_names: list,
              threshold: float = 0.1,
              predict_avg: np.ndarray = None,
              truth_avg: np.ndarray = None,
              lat: np.ndarray = None) -> dict:
    """
    Compute Root Mean Square Error (RMSE) for ERA5 gridded data.
    Supports both single timestep and temporal inputs.

    Parameters
    ----------
    predict : np.ndarray
        Forecasted data. Shape:
        - (B, V, H, W) for single timestep
        - (B, T, V, H, W) for multiple timesteps

    truth : np.ndarray
        Ground truth data, same shape as predict.

    variable_names : list of str
        Names of variables (length must match V).

    threshold : float, optional
        Not used. Included for interface consistency.

    predict_avg : np.ndarray, optional
        Historical average forecast (not used here).

    truth_avg : np.ndarray, optional
        Historical average truth (not used here).

    Returns
    -------
    rmse_dict : dict
        Dictionary of RMSE scores for each variable.
        - If input is 4D: float
        - If input is 5D: np.ndarray of shape (T,)
        Example:
            {
                'RMSE_Precip': 0.35,
                'RMSE_Temp': np.array([0.31, 0.32, ...])
            }
    """
    assert predict.shape == truth.shape, "predict and truth must have the same shape."
    assert predict.ndim in (4, 5), "Input must be 4D or 5D."

    if predict.ndim == 4:
        predict = predict[:, None, ...]  # (B, 1, V, H, W)
        truth = truth[:, None, ...]
        is_temporal = False
    else:
        is_temporal = True

    B, T, V, H, W = predict.shape
    assert len(variable_names) == V, "Length of variable_names must match V dimension."

    rmse_dict = {}

    for v in range(V):
        var_name = variable_names[v]
        rmse_list = []

        for t in range(T):
            pred_t = predict[:, t, v]
            truth_t = truth[:, t, v]
            rmse = np.sqrt(np.mean((pred_t - truth_t) ** 2))
            rmse_list.append(rmse)

        rmse_dict[f'RMSE_{var_name}'] = np.array(rmse_list) if is_temporal else rmse_list[0]

    return rmse_dict

def era5_hss(predict: np.ndarray,
             truth: np.ndarray,
             variable_names: list,
             threshold: float = 1,
             predict_avg: np.ndarray = None,
             truth_avg: np.ndarray = None,
             lat: np.ndarray = None) -> dict:
    """
    Compute Heidke Skill Score (HSS) for ERA5 gridded data.
    
    Parameters
    ----------
    predict/truth : np.ndarray
        Shape (B, V, H, W) or (B, T, V, H, W)
    variable_names : list of str
        Names of variables.
    threshold : float
        Threshold to define a "Yes" event (e.g., precipitation > 0.1mm).
    """
    assert predict.shape == truth.shape, "predict and truth must have the same shape."
    assert predict.ndim in (4, 5), "Input must be 4D or 5D."

    if predict.ndim == 4:
        predict = predict[:, None, ...]  # (B, 1, V, H, W)
        truth = truth[:, None, ...]
        is_temporal = False
    else:
        is_temporal = True

    B, T, V, H, W = predict.shape
    assert len(variable_names) == V, "Length of variable_names must match V dimension."

    hss_dict = {}

    for v in range(V):
        var_name = variable_names[v]
        hss_list = []

        for t in range(T):
            # 二值化处理
            pred_bin = (predict[:, t, v] >= threshold)
            truth_bin = (truth[:, t, v] >= threshold)

            # 计算混淆矩阵元素
            H = np.sum(pred_bin & truth_bin)          # Hits
            F = np.sum(pred_bin & ~truth_bin)         # False Alarms
            M = np.sum(~pred_bin & truth_bin)         # Misses
            CN = np.sum(~pred_bin & ~truth_bin)       # Correct Negatives
            
            total = H + F + M + CN
            
            # 计算随机期望正确数 E
            # Expected correct = [(H+F)*(H+M) + (M+CN)*(F+CN)] / total
            expected_correct = ((H + F) * (H + M) + (M + CN) * (F + CN)) / total
            
            # 计算 HSS
            denominator = total - expected_correct
            if denominator == 0:
                hss = 0.0  # 防止分母为0
            else:
                hss = (H + CN - expected_correct) / denominator
            
            hss_list.append(hss)

        hss_dict[f'HSS_{var_name}'] = np.array(hss_list) if is_temporal else hss_list[0]

    return hss_dict

def era5_crps(predict: np.ndarray,
              truth: np.ndarray,
              variable_names: list,
              **kwargs) -> dict:
    """
    Compute Continuous Ranked Probability Score (CRPS).
    
    Parameters
    ----------
    predict : np.ndarray
        Ensemble forecast data. Shape:
        - (B, M, V, H, W) for single timestep
        - (B, T, M, V, H, W) for multiple timesteps
        M is the number of ensemble members.
    
    truth : np.ndarray
        Ground truth data. Shape:
        - (B, V, H, W) or (B, T, V, H, W)
        No member dimension for truth.

    variable_names : list of str
        Names of variables.

    Returns
    -------
    crps_dict : dict
        CRPS scores per variable.
    """
    # 统一维度到 6D: (B, T, M, V, H, W)
    if predict.ndim == 5:
        predict = predict[:, None, ...]  # (B, 1, M, V, H, W)
        is_temporal = False
    else:
        is_temporal = True
        
    if truth.ndim == 4:
        truth = truth[:, None, ...]      # (B, 1, V, H, W)

    B, T, M, V, H, W = predict.shape
    assert truth.shape == (B, T, V, H, W), "Truth shape must match Predict minus Member dim."

    crps_dict = {}

    for v in range(V):
        var_name = variable_names[v]
        crps_list = []

        for t in range(T):
            # 获取当前时刻数据: (B, M, H, W) 和 (B, H, W)
            pred_t = predict[:, t, :, v, :, :]
            truth_t = truth[:, t, v, :, :]

            # CRPS 的高效计算公式 (Szunyogh et al., 2005):
            # CRPS(F, y) = E|X - y| - 0.5 * E|X - X'|
            # 其中 X, X' 是从预测分布中独立抽取的成员
            
            # 第一项: 预测成员与真值的平均绝对误差
            # mean over M dimension
            mae_truth = np.mean(np.abs(pred_t - np.expand_dims(truth_t, axis=1)), axis=1)
            
            # 第二项: 预测成员之间的平均绝对差 (Ensemble Dispersion)
            # 这是一个内存密集型操作，对于大 H, W 建议优化
            # 这里采用向量化简化计算
            # 这里的代价是 O(M^2)，如果 M 很大，建议使用随机采样估算
            mae_ensemble = 0
            for m in range(M):
                mae_ensemble += np.mean(np.abs(pred_t - pred_t[:, m:m+1, ...]), axis=1)
            mae_ensemble = mae_ensemble / (M * M)

            # 计算该时刻的空间平均值
            crps_val = np.mean(mae_truth - 0.5 * mae_ensemble)
            crps_list.append(crps_val)

        crps_dict[f'CRPS_{var_name}'] = np.array(crps_list) if is_temporal else crps_list[0]

    return crps_dict

def era5_spread_skill_ratio(predict: np.ndarray,
                            truth: np.ndarray,
                            variable_names: list,
                            eps: float = 1e-12,
                            ddof: int = 0,
                            **kwargs) -> dict:
    """
    Compute Spread/Skill Ratio (SSR) for ensemble forecasts.

    Parameters
    ----------
    predict : np.ndarray
        Ensemble forecast data. Shape:
        - (B, M, V, H, W) for single timestep
        - (B, T, M, V, H, W) for multiple timesteps
        M is the number of ensemble members.

    truth : np.ndarray
        Ground truth data. Shape:
        - (B, V, H, W) or (B, T, V, H, W)

    variable_names : list of str
        Names of variables (length V).

    eps : float
        Small constant to avoid division by zero.

    ddof : int
        Delta degrees of freedom for std (0 for population, 1 for sample std).

    Returns
    -------
    ssr_dict : dict
        Spread/Skill ratio per variable.
        - If temporal: each value is np.ndarray shape (T,)
        - Else: each value is a scalar
    """
    # ---- unify dims to 6D: (B, T, M, V, H, W) ----
    if predict.ndim == 5:
        predict = predict[:, None, ...]  # (B, 1, M, V, H, W)
        is_temporal = False
    elif predict.ndim == 6:
        is_temporal = True
    else:
        raise ValueError(f"predict must be 5D or 6D, got {predict.ndim}D")

    if truth.ndim == 4:
        truth = truth[:, None, ...]      # (B, 1, V, H, W)
    elif truth.ndim != 5:
        raise ValueError(f"truth must be 4D or 5D, got {truth.ndim}D")

    B, T, M, V, H, W = predict.shape
    assert truth.shape == (B, T, V, H, W), "Truth shape must match Predict minus Member dim."
    assert len(variable_names) == V, "variable_names length must equal V"

    ssr_dict = {}

    for v in range(V):
        var_name = variable_names[v]
        spread_list = []
        skill_list = []
        ssr_list = []

        for t in range(T):
            # pred_t: (B, M, H, W), truth_t: (B, H, W)
            pred_t = predict[:, t, :, v, :, :]
            truth_t = truth[:, t, v, :, :]

            # ---------- Spread ----------
            # ensemble std at each (B,H,W)
            ens_std = np.std(pred_t, axis=1, ddof=ddof)  # (B, H, W)
            spread = np.mean(ens_std)  # mean over B,H,W

            # ---------- Skill ----------
            # RMSE of ensemble mean vs truth
            ens_mean = np.mean(pred_t, axis=1)          # (B, H, W)
            mse = (ens_mean - truth_t) ** 2
            rmse = np.sqrt(np.mean(mse))                # scalar

            # ---------- Ratio ----------
            ssr = spread / (rmse + eps)

            spread_list.append(spread)
            skill_list.append(rmse)
            ssr_list.append(ssr)

        # 你可以只返回 SSR，也可以把 spread/skill 一起返回，方便诊断
        ssr_dict[f'SSR_{var_name}'] = np.array(ssr_list) if is_temporal else ssr_list[0]
        ssr_dict[f'Spread_{var_name}'] = np.array(spread_list) if is_temporal else spread_list[0]
        ssr_dict[f'SkillRMSE_{var_name}'] = np.array(skill_list) if is_temporal else skill_list[0]

    return ssr_dict

def era5_brier_score(predict: np.ndarray,
                     truth: np.ndarray,
                     variable_names: list,
                     thresholds: dict = None,
                     **kwargs) -> dict:
    """
    Compute Brier Score (BS) for probabilistic precipitation forecasting.
    
    Parameters
    ----------
    predict : np.ndarray
        Ensemble forecast data. Shape: (B, M, V, H, W) or (B, T, M, V, H, W)
    truth : np.ndarray
        Ground truth data. Shape: (B, V, H, W) or (B, T, V, H, W)
    variable_names : list of str
        Names of variables.
    thresholds : dict
        Thresholds for each variable. E.g., {'tp': 0.1, 'cp': 1.0}
        If None, a default 0.1 is used for all variables.

    Returns
    -------
    bs_dict : dict
        Brier Scores per variable per threshold.
    """
    # 统一维度到 6D: (B, T, M, V, H, W)
    if predict.ndim == 5:
        predict = predict[:, None, ...]  # (B, 1, M, V, H, W)
        is_temporal = False
    else:
        is_temporal = True
        
    if truth.ndim == 4:
        truth = truth[:, None, ...]      # (B, 1, V, H, W)

    B, T, M, V, H, W = predict.shape
    
    # 设置默认阈值
    if thresholds is None:
        thresholds = {var: 0.1 for var in variable_names}

    bs_dict = {}

    for v in range(V):
        var_name = variable_names[v]
        # 获取当前变量的阈值，如果没有指定则默认为 0.1
        threshold = thresholds.get(var_name, 0.1)
        bs_list = []

        for t in range(T):
            # 1. 计算预报概率 P: 集合成员中超过阈值的比例
            # pred_t shape: (B, M, H, W) -> prob_t shape: (B, H, W)
            pred_t = predict[:, t, :, v, :, :]
            prob_t = np.mean(pred_t >= threshold, axis=1)

            # 2. 计算观测二值结果 O: 超过阈值为 1，否则为 0
            # truth_t shape: (B, H, W) -> obs_binary_t shape: (B, H, W)
            truth_t = truth[:, t, v, :, :]
            obs_binary_t = (truth_t >= threshold).astype(np.float32)

            # 3. 计算 Brier Score: (P - O)^2 的均值
            # 空间平均 (H, W) 和 样本平均 (B)
            bs_val = np.mean((prob_t - obs_binary_t) ** 2)
            bs_list.append(bs_val)

        # 结果存入字典，Key 包含阈值以便区分
        key_name = f'BS_{var_name}_thr{threshold}'
        bs_dict[key_name] = np.array(bs_list) if is_temporal else bs_list[0]

    return bs_dict

def era5_acc(predict: np.ndarray,
             truth: np.ndarray,
             variable_names: list,
             threshold: float = 0.1,
             predict_avg: np.ndarray = None,
             truth_avg: np.ndarray = None,
             lat: np.ndarray = None) -> dict:
    """
    Compute Anomaly Correlation Coefficient (ACC) for ERA5 gridded data.
    Measures spatial pattern correlation between forecast and truth anomalies.

    Supports both single timestep and temporal inputs.

    Parameters
    ----------
    predict : np.ndarray
        Forecasted data. Shape:
        - (B, V, H, W) for single timestep
        - (B, T, V, H, W) for multiple timesteps

    truth : np.ndarray
        Ground truth data, same shape as predict.

    variable_names : list of str
        Names of variables (length must match V dimension).

    threshold : float, optional
        Not used here. Included for interface consistency.

    predict_avg : np.ndarray
        Climatological mean forecast. Shape must match (T, V, H, W) or (V, H, W).

    truth_avg : np.ndarray
        Climatological mean truth. Shape must match (T, V, H, W) or (V, H, W).

    Returns
    -------
    acc_dict : dict
        Dictionary with ACC scores per variable.
        - If input is 4D: float
        - If input is 5D: np.ndarray of shape (T,)
        Example:
            {
                'ACC_Precip': 0.61,
                'ACC_Temp': np.array([0.59, 0.63, ...])
            }
    """
    assert predict.shape == truth.shape, "predict and truth must have the same shape."
    assert predict.ndim in (4, 5), "Input must be 4D or 5D."
    assert predict_avg is not None and truth_avg is not None, "predict_avg and truth_avg are required for ACC."

    # Standardize to shape (B, T, V, H, W)
    if predict.ndim == 4:
        predict = predict[:, None, ...]
        truth = truth[:, None, ...]
        predict_avg = predict_avg[None]
        truth_avg = truth_avg[None]
        is_temporal = False
    else:
        is_temporal = True

    B, T, V, H, W = predict.shape
    assert len(variable_names) == V, "Length of variable_names must match V dimension."
    assert predict_avg.shape == (T, V, H, W), "predict_avg shape must match (T, V, H, W)."
    assert truth_avg.shape == (T, V, H, W), "truth_avg shape must match (T, V, H, W)."

    acc_dict = {}

    for v in range(V):
        var_name = variable_names[v]
        acc_list = []

        for t in range(T):
            pred_anom = predict[:, t, v] - predict_avg[t, v]
            truth_anom = truth[:, t, v] - truth_avg[t, v]

            numerator = np.sum(pred_anom * truth_anom)
            denominator = np.sqrt(np.sum(pred_anom ** 2) * np.sum(truth_anom ** 2))

            acc = 0.0 if denominator == 0 else numerator / denominator
            acc_list.append(acc)

        acc_dict[f'ACC_{var_name}'] = np.array(acc_list) if is_temporal else acc_list[0]

    return acc_dict

def era5_nse(predict: np.ndarray,
             truth: np.ndarray,
             variable_names: list,
             threshold: float = 0.1,
             predict_avg: np.ndarray = None,
             truth_avg: np.ndarray = None) -> dict:
    """
    Compute Nash–Sutcliffe Efficiency (NSE) for ERA5 gridded data.
    NSE measures how well forecasted time series match the observations.
    Values range from (-inf, 1], with 1 being a perfect prediction.

    Supports both single timestep and temporal inputs.

    Parameters
    ----------
    predict : np.ndarray
        Forecasted data. Shape:
        - (B, V, H, W) for single timestep
        - (B, T, V, H, W) for multiple timesteps

    truth : np.ndarray
        Ground truth data, same shape as predict.

    variable_names : list of str
        Names of variables (length must match V dimension).

    threshold : float, optional
        Not used here. Included for interface consistency.

    predict_avg : np.ndarray, optional
        Not used in NSE, included for unified API.

    truth_avg : np.ndarray, optional
        Not used in NSE, included for unified API.

    Returns
    -------
    nse_dict : dict
        Dictionary with NSE scores per variable.
        - If input is 4D: float
        - If input is 5D: np.ndarray of shape (T,)
        Example:
            {
                'NSE_Precip': 0.81,
                'NSE_Temp': np.array([0.78, 0.75, ...])
            }
    """
    assert predict.shape == truth.shape, "predict and truth must have the same shape."
    assert predict.ndim in (4, 5), "Input must be 4D or 5D."

    # Convert to shape (B, T, V, H, W)
    if predict.ndim == 4:
        predict = predict[:, None, ...]
        truth = truth[:, None, ...]
        is_temporal = False
    else:
        is_temporal = True

    B, T, V, H, W = predict.shape
    assert len(variable_names) == V, "Length of variable_names must match V dimension."

    nse_dict = {}

    for v in range(V):
        var_name = variable_names[v]
        nse_list = []

        for t in range(T):
            # Aggregate per sample (sum over H, W)
            sim = np.sum(predict[:, t, v], axis=(1, 2))  # shape (B,)
            obs = np.sum(truth[:, t, v], axis=(1, 2))    # shape (B,)
            obs_mean = obs.mean()

            numerator = np.sum((sim - obs) ** 2)
            denominator = np.sum((obs - obs_mean) ** 2)

            nse = 0.0 if denominator == 0 else 1.0 - (numerator / denominator)
            nse_list.append(nse)

        nse_dict[f'NSE_{var_name}'] = np.array(nse_list) if is_temporal else nse_list[0]

    return nse_dict
