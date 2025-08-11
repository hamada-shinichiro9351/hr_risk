import os
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from io import BytesIO
from typing import Any
from modules.utils import anonymize_ids
import pandas as pd
from datetime import datetime

def _register_jp_font() -> str:
    target_name = "JPFont"
    try:
        if target_name in pdfmetrics.getRegisteredFontNames():
            return target_name
    except Exception:
        pass
    candidates = [
        (os.getenv("JP_FONT_PATH"), None),
        (os.path.join("assets", "fonts", "NotoSansCJKjp-Regular.otf"), None),
        (os.path.join("assets", "fonts", "NotoSansJP-Regular.ttf"), None),
        (r"C:\\Windows\\Fonts\\meiryo.ttc", 0),
        (r"C:\\Windows\\Fonts\\YuGothM.ttc", 0),
        (r"C:\\Windows\\Fonts\\msgothic.ttc", 0),
    ]
    for path, idx in candidates:
        try:
            if path and os.path.exists(path):
                pdfmetrics.registerFont(TTFont(target_name, path, subfontIndex=0 if idx is None else idx))
                return target_name
        except Exception:
            continue
    try:
        cid_name = "HeiseiKakuGo-W5"
        pdfmetrics.registerFont(UnicodeCIDFont(cid_name))
        return cid_name
    except Exception:
        return "Helvetica"

def _header(c: canvas.Canvas, title: str, font_name: str):
    c.setFont(font_name, 14)
    c.drawString(20*mm, 285*mm, title)
    c.setFont(font_name, 9)
    c.drawString(20*mm, 280*mm, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

def build_pdf_attrition(df: pd.DataFrame, meta: dict, high_threshold: int = 70, med_threshold: int = 40, anonymize: bool = False) -> bytes:
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    font_name = _register_jp_font()
    _header(c, "離職リスクレポート", font_name)
    c.setFont(font_name, 10)
    c.drawString(20*mm, 270*mm, f"モデル: {meta.get('model','N/A')}")
    c.drawString(20*mm, 262*mm, f"従業員数: {len(df)}")
    high = int((df['離職確率(%)']>=high_threshold).sum()); med = int(((df['離職確率(%)']>=med_threshold)&(df['離職確率(%)']<high_threshold)).sum())
    c.drawString(20*mm, 254*mm, f"High: {high} / Medium: {med}")

    y = 240
    c.setFont(font_name, 10)
    c.drawString(20*mm, y*mm, "上位リスク（Top 10）")
    c.setFont(font_name, 9)
    y -= 6
    show_df = df.copy()
    if anonymize and "社員ID" in show_df.columns:
        show_df["社員ID"] = anonymize_ids(pd.Series(show_df["社員ID"]))
    for _, r in show_df.sort_values("離職確率(%)", ascending=False).head(10).iterrows():
        line = f"社員ID:{r['社員ID']}  確率:{r['離職確率(%)']}%  勤続:{r['勤続年数']}年  残業:{r['平均残業時間h']}h  有給:{r['有給取得率']}%"
        c.drawString(20*mm, y*mm, line)
        y -= 6
        if y < 20:
            c.showPage(); y = 280
    c.showPage(); c.save()
    buf.seek(0); return buf.read()

def build_pdf_attendance(df: pd.DataFrame, anonymize: bool = False) -> bytes:
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    font_name = _register_jp_font()
    _header(c, "勤怠異常レポート", font_name)
    c.setFont(font_name, 10)
    c.drawString(20*mm, 270*mm, f"検出件数: {len(df)}")
    long_series = df.get("異常_長時間", pd.Series([0])) if len(df) > 0 else pd.Series([0])
    streak_series = df.get("異常_連続", pd.Series([0])) if len(df) > 0 else pd.Series([0])
    long_cnt = int(pd.Series(long_series).sum())
    streak_cnt = int(pd.Series(streak_series).sum())
    c.drawString(20*mm, 262*mm, f"長時間勤務フラグ: {long_cnt} / 連続勤務フラグ: {streak_cnt}")

    y = 248
    c.setFont(font_name, 10)
    c.drawString(20*mm, y*mm, "異常一覧（最大20件）")
    c.setFont(font_name, 9)
    y -= 6
    show_df = df.copy()
    if anonymize and "社員ID" in show_df.columns:
        show_df["社員ID"] = anonymize_ids(pd.Series(show_df["社員ID"]))
    for _, r in show_df.head(20).iterrows():
        date_val = r["日付"]
        date_str = date_val.split('T')[0] if isinstance(date_val, str) else getattr(date_val, 'strftime', lambda _fmt: str(date_val))('%Y-%m-%d')
        line = f"{date_str}  社員ID:{r['社員ID']}  残業:{r['残業時間h']}h  連続:{int(r['連続出勤日数'])}日"
        c.drawString(20*mm, y*mm, line)
        y -= 6
        if y < 20:
            c.showPage(); y = 280
    c.showPage(); c.save()
    buf.seek(0); return buf.read()
