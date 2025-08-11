import pandas as pd
import numpy as np
import io
from typing import Optional, Sequence, Union, Hashable, Any, cast
import hashlib
import os

def ensure_columns(df: pd.DataFrame, required: list[str]):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"必須カラムが不足しています: {missing}")

def clip01(x):
    return float(np.clip(x, 0.0, 1.0))

def read_csv_auto(src: Any, parse_dates: Union[bool, Sequence[Hashable], None] = None) -> pd.DataFrame:
    """UTF-8(BOM) と CP932 を順に試す簡易CSV読込。
    - src: パス文字列、Bytes、あるいは BytesIO
    - parse_dates: pandas の parse_dates 引数（列名のリスト等）
    """
    encodings = ["utf-8-sig", "utf-8", "cp932"]
    if isinstance(src, (bytes, bytearray, io.BytesIO)):
        raw = src.getvalue() if isinstance(src, io.BytesIO) else bytes(src)
        for enc in encodings:
            try:
                buf = io.BytesIO(raw)
                return pd.read_csv(buf, encoding=enc, parse_dates=parse_dates)
            except Exception:
                continue
        # 最後の手段：エンコード未指定でトライ
        return pd.read_csv(io.BytesIO(raw), parse_dates=parse_dates)
    else:
        # ファイルパスなど
        for enc in encodings:
            try:
                return pd.read_csv(src, encoding=enc, parse_dates=parse_dates)
            except Exception:
                continue
        return pd.read_csv(src, parse_dates=parse_dates)


def read_table_auto(src: Any, parse_dates: Union[bool, Sequence[Hashable], None] = None, sheet_name: Any = 0) -> pd.DataFrame:
    """CSV/Excelの双方を自動判別して読み込む。
    - src: パス、Bytes、BytesIO
    - Excel判定: 拡張子（.xlsx/.xls）または先頭バイト（PK / D0 CF 11 E0）
    - Excelエンジンが未導入時は明確な例外メッセージを上げる
    """
    def _is_excel_path(path: str) -> tuple[bool, str]:
        low = path.lower()
        if low.endswith(".xlsx") or low.endswith(".xlsm"):
            return True, "xlsx"
        if low.endswith(".xls"):
            return True, "xls"
        return False, ""

    def _bytes_kind(raw: bytes) -> str:
        if raw.startswith(b"PK"):
            return "xlsx"
        if raw.startswith(b"\xD0\xCF\x11\xE0"):
            return "xls"
        return "csv"

    # ファイルパス文字列の場合
    if isinstance(src, str):
        is_excel, kind = _is_excel_path(src)
        if is_excel:
            try:
                engine = "openpyxl" if kind == "xlsx" else None  # xls は古いエンジン（xlrd）が必要
                # read_excel には parse_dates を渡さず、後段で日付列があれば to_datetime する
                df = pd.read_excel(src, engine=engine, sheet_name=sheet_name)
                if parse_dates:
                    for col in cast(Sequence[Hashable], parse_dates):
                        if col in df.columns:
                            df[col] = pd.to_datetime(df[col], errors="coerce")
                return df
            except ImportError as ie:
                raise ImportError("Excel読込には openpyxl（xlsx）または xlrd（xls）が必要です。'pip install openpyxl' を実行してください。") from ie
        # CSVとして処理
        return read_csv_auto(src, parse_dates=parse_dates)

    # バイナリ/BytesIOの場合
    if isinstance(src, (bytes, bytearray, io.BytesIO)):
        raw = src.getvalue() if isinstance(src, io.BytesIO) else bytes(src)
        kind = _bytes_kind(raw)
        if kind in ("xlsx", "xls"):
            try:
                engine = "openpyxl" if kind == "xlsx" else "xlrd"
                df = pd.read_excel(io.BytesIO(raw), engine=engine, sheet_name=sheet_name)
                if parse_dates:
                    for col in cast(Sequence[Hashable], parse_dates):
                        if col in df.columns:
                            df[col] = pd.to_datetime(df[col], errors="coerce")
                return df
            except ImportError as ie:
                raise ImportError("Excel読込には openpyxl（xlsx）または xlrd（xls）が必要です。'pip install openpyxl' を実行してください。") from ie
            except Exception:
                # 最後の手段でCSVとして試す
                return read_csv_auto(io.BytesIO(raw), parse_dates=parse_dates)
        # CSVとして処理
        return read_csv_auto(io.BytesIO(raw), parse_dates=parse_dates)

    # その他はそのままCSV推定
    return read_csv_auto(src, parse_dates=parse_dates)


def validate_hr(df: pd.DataFrame) -> dict[str, list[str]]:
    """人事データの軽微な検証を行い、errors/warnings を返す。
    - 必須カラムの不足
    - 数値の範囲チェック（年齢/勤続年数/残業/有給率/評価/昇給/異動）
    - 欠損の有無
    """
    errors: list[str] = []
    warnings: list[str] = []
    required = [
        "社員ID","年齢","勤続年数","平均残業時間h","有給取得率","評価(1-5)","昇給回数","部署異動回数"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        errors.append(f"必須カラムが不足しています: {missing}")
        return {"errors": errors, "warnings": warnings}

    # 欠損
    null_counts = df[required].isna().sum()
    bad_null = {col: int(cnt) for col, cnt in null_counts.items() if cnt > 0}
    if bad_null:
        errors.append(f"必須カラムに欠損があります: {bad_null}")

    # 範囲チェック
    def count_out_of_range(series: pd.Series, min_v: float | None, max_v: float | None) -> int:
        s = pd.Series(pd.to_numeric(series, errors="coerce"))
        mask = pd.Series([False] * int(len(s)), index=s.index)
        if min_v is not None:
            mask |= s < min_v
        if max_v is not None:
            mask |= s > max_v
        return int(mask.fillna(True).sum())  # NaNは別途欠損で拾うが、ここでは不正としてカウント

    ranges = {
        "年齢": (15, 80),
        "勤続年数": (0, 50),
        "平均残業時間h": (0, 200),
        "有給取得率": (0, 100),
        "評価(1-5)": (1, 5),
        "昇給回数": (0, 50),
        "部署異動回数": (0, 50),
    }
    for col, (lo, hi) in ranges.items():
        n_bad = count_out_of_range(pd.Series(df[col]), lo, hi)
        if n_bad > 0:
            warnings.append(f"{col} の範囲外データが {n_bad} 件あります（想定範囲: {lo}〜{hi}）。")

    # 社員IDの重複
    if df["社員ID"].duplicated().any():
        warnings.append("社員IDに重複があります。集計時の解釈に注意してください。")

    return {"errors": errors, "warnings": warnings}


def validate_attendance(df: pd.DataFrame) -> dict[str, list[str]]:
    """勤怠データの軽微な検証。
    - 必須カラムの不足
    - 日付のパース可否/欠損
    - 時間の範囲チェック（勤務時間/残業時間）
    - 欠損の有無
    """
    errors: list[str] = []
    warnings: list[str] = []
    required = ["社員ID","日付","勤務時間h","残業時間h"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        errors.append(f"必須カラムが不足しています: {missing}")
        return {"errors": errors, "warnings": warnings}

    # 欠損
    null_counts = df[required].isna().sum()
    bad_null = {col: int(cnt) for col, cnt in null_counts.items() if cnt > 0}
    if bad_null:
        errors.append(f"必須カラムに欠損があります: {bad_null}")

    # 日付
    dates = pd.to_datetime(df["日付"], errors="coerce")
    nat_cnt = int(dates.isna().sum())
    if nat_cnt > 0:
        errors.append(f"日付のパースに失敗した行が {nat_cnt} 件あります（YYYY-MM-DD 形式を推奨）。")

    # 範囲チェック
    def count_out_of_range(series: pd.Series, min_v: float | None, max_v: float | None) -> int:
        s = pd.Series(pd.to_numeric(series, errors="coerce"))
        mask = pd.Series([False] * int(len(s)), index=s.index)
        if min_v is not None:
            mask |= s < min_v
        if max_v is not None:
            mask |= s > max_v
        return int(mask.fillna(True).sum())

    n_bad_work = count_out_of_range(pd.Series(df["勤務時間h"]), 0, 24*2)  # 2日相当を上限としてソフトにチェック
    n_bad_ot = count_out_of_range(pd.Series(df["残業時間h"]), 0, 24)
    if n_bad_work > 0:
        warnings.append(f"勤務時間h の異常値が {n_bad_work} 件あります。")
    if n_bad_ot > 0:
        warnings.append(f"残業時間h の異常値が {n_bad_ot} 件あります。")

    return {"errors": errors, "warnings": warnings}


def anonymize_ids(series: pd.Series, prefix: str = "ID_", length: int = 8) -> pd.Series:
    """社員IDを安定ハッシュで匿名化して返す。
    - prefix: 付与する接頭辞
    - length: 先頭から何文字切り出すか（16進）
    - 環境変数 ANON_SALT があればそれも混ぜてハッシュ化
    """
    salt = os.getenv("ANON_SALT", "hrtool")
    def _mask(x: Any) -> str:
        b = str(x).encode("utf-8")
        h = hashlib.sha256(salt.encode("utf-8") + b).hexdigest()[: max(4, length)]
        return f"{prefix}{h}"
    return series.astype(str).map(_mask)


def normalize_hr_columns(df: pd.DataFrame) -> pd.DataFrame:
    """人事データの列名・値をツールの期待に揃える軽い正規化。
    - 列名マッピング（例: 評価 -> 評価(1-5)）
    - 有給取得率が 0-1 レンジなら 0-100 にスケーリング
    """
    df2 = df.copy()
    col_map = {
        "評価": "評価(1-5)",
        # 他にも必要に応じ拡張
    }
    rename_keys = {k: v for k, v in col_map.items() if k in df2.columns and v not in df2.columns}
    if rename_keys:
        df2 = df2.rename(columns=rename_keys)

    if "有給取得率" in df2.columns:
        ser_num = pd.Series(pd.to_numeric(df2["有給取得率"], errors="coerce"), dtype=float)
        ser_max_val = float(pd.Series(ser_num).fillna(-1.0).max())
        if 0.0 <= ser_max_val <= 1.0:
            df2["有給取得率"] = (ser_num * 100.0).astype(float)
        else:
            df2["有給取得率"] = ser_num
    return df2


# ===== 列名マッピング支援 =====
def _normalize_name(name: str) -> str:
    s = str(name).strip().lower()
    # 記号・スペース・丸括弧などを削除して比較を寛容に
    for ch in [" ", "\t", "\n", "(", ")", "％", "%", "-", "_", ":", "・", "h"]:
        s = s.replace(ch, "")
    return s


def suggest_column_mapping(actual_columns: Sequence[str], required: Sequence[str], domain: str) -> dict[str, str]:
    """実データ列から標準列名への推定マッピングを返す。
    - domain: "hr" or "attendance" でシノニムを切替
    """
    norm_actual = {col: _normalize_name(col) for col in actual_columns}

    synonyms: dict[str, list[str]] = {}
    if domain == "hr":
        synonyms = {
            "社員ID": ["社員id", "従業員id", "従業員番号", "社員番号", "id"],
            "年齢": ["age"],
            "勤続年数": ["勤続", "在籍年数", "yearsinaservice"],
            "平均残業時間h": ["平均残業", "残業時間", "ot", "overtime"],
            "有給取得率": ["有給率", "有休取得率", "有休率", "pto", "vacationrate"],
            "評価(1-5)": ["評価", "レーティング", "rating", "評価スコア"],
            "昇給回数": ["昇給", "昇給回", "賃上げ回数"],
            "部署異動回数": ["異動回数", "部署異動", "ローテ回数"],
        }
    else:
        synonyms = {
            "社員ID": ["社員id", "従業員id", "従業員番号", "社員番号", "id"],
            "日付": ["年月日", "date", "日"],
            "勤務時間h": ["勤務時間", "実働時間", "労働時間", "稼働時間"],
            "残業時間h": ["残業時間", "ot", "overtime", "超過時間"],
        }

    mapping: dict[str, str] = {}
    for req in required:
        req_norm = _normalize_name(req)
        # 1) 完全一致（緩和正規化済み）
        hit = None
        for col, ncol in norm_actual.items():
            if ncol == req_norm:
                hit = col
                break
        # 2) シノニム一致
        if hit is None:
            for col, ncol in norm_actual.items():
                for syn in synonyms.get(req, []):
                    if ncol == _normalize_name(syn) or _normalize_name(syn) in ncol:
                        hit = col; break
                if hit:
                    break
        # 3) 部分一致
        if hit is None:
            for col, ncol in norm_actual.items():
                if req_norm in ncol:
                    hit = col; break
        if hit:
            mapping[req] = hit
    return mapping


def apply_column_mapping(df: pd.DataFrame, mapping_req_to_actual: dict[str, str]) -> pd.DataFrame:
    """推定/選択されたマッピング（標準名->実列名）を用いて列名を標準化する。"""
    ren = {actual: req for req, actual in mapping_req_to_actual.items() if actual in df.columns}
    if not ren:
        return df
    return df.rename(columns=ren)
