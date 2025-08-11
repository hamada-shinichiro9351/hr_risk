import numpy as np
import pandas as pd
from typing import Tuple, Dict, List
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from .utils import clip01

# 推奨カラム（必須ではない）
PREFERRED_FEATURES: List[str] = [
    "年齢",
    "勤続年数",
    "平均残業時間h",
    "有給取得率",
    "評価(1-5)",
    "昇給回数",
    "部署異動回数",
]

# 予備：ラベルなしデータに対する簡易ルールモデル（重みは経験則）
def rule_based_score(row) -> float:
    w = {
        "勤続年数": -0.10,     # 長いほど離職低下
        "年齢": -0.05,         # 高いほど離職低下（一般化）
        "平均残業時間h": 0.18, # 多いほど離職上昇
        "有給取得率": -0.20,   # 高いほど離職低下
        "評価(1-5)": -0.25,    # 高評価ほど離職低下
        "昇給回数": -0.10,     # 多いほど離職低下
        "部署異動回数": 0.08   # 多いほど離職上昇（適応失敗の仮説）
    }
    # 正規化・安全化
    def get(col: str, default: float | None = None) -> float | None:
        try:
            val = row.get(col) if hasattr(row, 'get') else row[col]
            return float(val)
        except Exception:
            return default
    contributions = []
    tenure = get("勤続年数");
    if tenure is not None:
        contributions.append(w["勤続年数"] * min(max(tenure/10, 0), 1))
    age = get("年齢");
    if age is not None:
        contributions.append(w["年齢"] * min(max((age-20)/30, 0), 1))
    ot = get("平均残業時間h");
    if ot is not None:
        contributions.append(w["平均残業時間h"] * min(max(ot/60, 0), 1))
    pto = get("有給取得率");
    if pto is not None:
        contributions.append(w["有給取得率"] * min(max(pto/100, 0), 1))
    rating = get("評価(1-5)");
    if rating is not None:
        contributions.append(w["評価(1-5)"] * min(max((rating-1)/4, 0), 1))
    raises = get("昇給回数");
    if raises is not None:
        contributions.append(w["昇給回数"] * min(max(raises/5, 0), 1))
    moves = get("部署異動回数");
    if moves is not None:
        contributions.append(w["部署異動回数"] * min(max(moves/3, 0), 1))
    x = sum(contributions) if contributions else 0.0
    # ロジスティック変換（閾値調整）
    prob = 1 / (1 + np.exp(- (x*3)))
    return clip01(prob)

def predict_attrition(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    df = df.copy()

    has_label = "attrition" in df.columns  # 1=離職、0=在籍
    model_name = "Rule-Based"
    proba = None

    # 利用可能な特徴量のみを使用
    feature_cols = [c for c in PREFERRED_FEATURES if c in df.columns]

    if has_label and len(feature_cols) > 0:
        # ラベルがある場合は学習→予測（利用可能な特徴のみ）
        features = df[feature_cols]
        y = df["attrition"].astype(int)
        X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.2, random_state=42, stratify=y)

        pipe = Pipeline([
            ("scaler", StandardScaler(with_mean=False)),
            ("clf", LogisticRegression(max_iter=500))
        ])
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(df[feature_cols])[:,1]
        model_name = f"LogReg({len(feature_cols)}f)"
    else:
        # ラベルなし→ルールベース（利用可能な特徴のみ寄与）
        proba = df.apply(rule_based_score, axis=1).values

    # 社員IDが無い場合は連番を付与（推奨: 事前にマッピングで設定）
    if "社員ID" not in df.columns:
        df["社員ID"] = np.arange(1, len(df)+1)

    proba_series = pd.Series(proba, dtype=float)
    result = pd.DataFrame({
        "社員ID": df["社員ID"],
        "離職確率(%)": np.round(proba_series*100.0, 1)
    })
    result["リスク区分"] = pd.cut(result["離職確率(%)"],
                          bins=[-0.1, 40, 70, 100],
                          labels=["Low","Medium","High"])
    # 参照用の主要説明変数（存在するもののみ）
    ref_cols = [c for c in ["勤続年数","平均残業時間h","有給取得率","評価(1-5)"] if c in df.columns]
    if ref_cols:
        result = result.merge(df[["社員ID", *ref_cols]], on="社員ID", how="left")
    meta = {"model": model_name}
    return result.sort_values("離職確率(%)", ascending=False), meta
