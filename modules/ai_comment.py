import os
import json
import pandas as pd

def _rule_based_attrition(df: pd.DataFrame) -> str:
    high = int((df["離職確率(%)"] >= 70).sum())
    med = int(((df["離職確率(%)"] >= 40) & (df["離職確率(%)"] < 70)).sum())
    avg_ot = float(df["平均残業時間h"].mean()) if "平均残業時間h" in df.columns else 0.0
    lines = [
        f"所見: High {high}名 / Medium {med}名、平均残業 {avg_ot:.1f}h。",
        "提案: 高リスク者に1on1を実施し、有給取得の計画的な促進を即時着手。",
    ]
    return "\n".join(lines)

def _rule_based_attendance(df: pd.DataFrame) -> str:
    if df is None or len(df) == 0:
        return "所見: 異常なし。\n提案: 現状維持と、繁忙期前の計画的な休暇取得を周知。"
    long_series = df.get("異常_長時間")
    streak_series = df.get("異常_連続")
    long_cnt = int((pd.Series([], dtype="bool") if long_series is None else pd.Series(long_series)).sum())
    streak_cnt = int((pd.Series([], dtype="bool") if streak_series is None else pd.Series(streak_series)).sum())
    lines = [
        f"所見: 異常 {len(df)}件。長時間 {long_cnt}件 / 連続 {streak_cnt}件。",
        "提案: 業務割当の平準化と残業上限の明確化、対象者への面談実施。",
    ]
    return "\n".join(lines)

def generate_comment(dataframe: pd.DataFrame, target: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        # ルールベースfallback
        return _rule_based_attrition(dataframe) if target == "attrition" else _rule_based_attendance(dataframe)

    # OpenAIあり：短文所見を生成
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        if target == "attrition":
            k = {
                "high": int((dataframe["離職確率(%)"] >= 70).sum()),
                "median_prob": float(dataframe["離職確率(%)"].median()),
            }
        else:
            long_hours_col = dataframe.get("異常_長時間")
            streaks_col = dataframe.get("異常_連続")
            k = {
                "count": int(len(dataframe)),
                "long_hours": int(long_hours_col.sum() if long_hours_col is not None else 0),
                "streaks": int(streaks_col.sum() if streaks_col is not None else 0),
            }
        prompt = f"""あなたは人事コンサルタントです。以下の指標を基に、短く読みやすい2行構成で日本語の出力を作成してください。
1行目: 所見（ファクトに基づく要約, 60〜80文字程度）
2行目: 提案（具体的な次アクションを1つ, 40〜60文字程度）
対象: {target}
指標: {json.dumps(k, ensure_ascii=False)}
"""
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":prompt}],
            temperature=0.4,
            max_tokens=200
        )
        content = resp.choices[0].message.content
        return content.strip() if content else ""
    except Exception:
        # 失敗時はルールベース
        return _rule_based_attrition(dataframe) if target == "attrition" else _rule_based_attendance(dataframe)
