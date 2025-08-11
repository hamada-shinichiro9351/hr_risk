import pandas as pd
import numpy as np
from typing import Tuple
import plotly.express as px
from plotly.graph_objs import Figure

from .utils import ensure_columns

REQ = ["社員ID","日付","勤務時間h","残業時間h"]

def detect_attendance_anomalies(
    df: pd.DataFrame,
    z_thresh: float = 2.0,
    long_hours_threshold: float = 11,
    streak_threshold: int = 12,
) -> Tuple[pd.DataFrame, Figure]:
    ensure_columns(df, REQ)
    df = df.copy()
    df["日付"] = pd.to_datetime(df["日付"])
    df.sort_values(["社員ID","日付"], inplace=True)

    # 全体概要チャート（残業の推移）
    overview = df.groupby("日付", as_index=False)["残業時間h"].sum()
    fig = px.line(overview, x="日付", y="残業時間h", title="全体残業時間（合計）の推移")

    # Zスコア異常（個人ごと）
    def z_anom(x):
        m = x.mean()
        s = x.std(ddof=0)
        if s == 0:  # 変動なし
            return pd.Series([False]*len(x), index=x.index)
        z = (x - m) / s
        return z > z_thresh

    df["異常_残業z"] = df.groupby("社員ID")["残業時間h"].transform(z_anom)

    # 長時間残業の閾値（9h超など）も併用
    df["異常_長時間"] = df["勤務時間h"] >= long_hours_threshold

    # 連続出勤（勤務時間>0の連続日数）12日以上
    def longest_streak(dates, hours):
        streaks = []
        current = 0
        prev = None
        for dt, h in zip(dates, hours):
            worked = h > 0
            if worked and (prev is None or (dt - prev).days <= 1):
                current += 1
            elif worked:
                current = 1
            else:
                current = 0
            streaks.append(current)
            prev = dt
        return pd.Series(streaks, index=dates.index)

    df["連続出勤日数"] = df.groupby("社員ID").apply(
        lambda g: longest_streak(g["日付"], g["勤務時間h"])
    ).reset_index(level=0, drop=True)
    df["異常_連続"] = df["連続出勤日数"] >= streak_threshold

    anomalies = df[(df["異常_残業z"]) | (df["異常_長時間"]) | (df["異常_連続"])].copy()
    show_cols = ["社員ID","日付","勤務時間h","残業時間h","連続出勤日数","異常_残業z","異常_長時間","異常_連続"]
    return anomalies.loc[:, show_cols].sort_values(["社員ID","日付"]), fig

def plot_attendance_overview(df: pd.DataFrame):
    overview = df.groupby("日付", as_index=False)["残業時間h"].sum()
    return px.line(overview, x="日付", y="残業時間h", title="全体残業時間（合計）の推移")
