import os
import io
from typing import Any, cast
import plotly.io as pio
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

from modules.risk_model import predict_attrition
from modules.utils import (
    read_csv_auto,
    read_table_auto,
    validate_hr,
    validate_attendance,
    anonymize_ids,
    normalize_hr_columns,
    suggest_column_mapping,
    apply_column_mapping,
)
from modules.attendance_check import detect_attendance_anomalies, plot_attendance_overview
from modules.ai_comment import generate_comment
from modules.report import build_pdf_attrition, build_pdf_attendance

st.set_page_config(page_title="人事リスク・勤怠モニター", page_icon="👥", layout="wide", initial_sidebar_state="collapsed")
load_dotenv()

st.title("👥 人事リスク・勤怠モニター")
st.caption("AI＋Pythonで、離職リスク予測と勤怠異常検知を自動化（サンプルデータ同梱）")
st.caption("⚙ 設定は左側のサイドバーから開閉できます（画面上部のアイコン）。")

# 軽いテーマ調整（Plotlyとカード風スタイル）
pio.templates.default = "plotly_white"
st.markdown(
    """
    <style>
    /* metric カードの装飾 */
    div[data-testid="metric-container"] {
        background: #f7fafc;
        border: 1px solid #e6eef5;
        padding: 8px 10px;
        border-radius: 10px;
        box-shadow: 0 0 0 rgba(0,0,0,0);
    }
    /* ボタンの角丸強化 */
    .stButton>button { border-radius: 8px; }
    /* データフレーム行の行間を少し広げる */
    .stDataFrame div[role="row"] { align-items: center; }
    </style>
    """,
    unsafe_allow_html=True,
)

# 追加の配色（全体背景・タブ・ボタンなど）
st.markdown(
    """
    <style>
    /* 背景をわずかに色付け（白一色を回避） */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(180deg, #f5f7fb 0%, #fbfcff 100%);
    }
    /* サイドバーの背景と境界 */
    [data-testid="stSidebar"] {
        background: #f8fafc;
        border-right: 1px solid #e6eef5;
    }
    /* タブの見た目を柔らかく */
    button[role="tab"] {
        border-radius: 8px !important;
        background: #eef2f7 !important;
        margin-right: 6px !important;
        color: #334155 !important;
    }
    button[role="tab"][aria-selected="true"] {
        background: #dbe6f3 !important;
        color: #0f172a !important;
    }
    /* エクスパンダのヘッダー背景 */
    summary {
        background: #f1f5f9;
        border-radius: 6px;
        padding: 4px 6px;
    }
    /* ダウンロード/通常ボタンの配色とホバー */
    .stButton>button, .stDownloadButton>button {
        background: #2563eb22;
        border: 1px solid #2563eb33;
        color: #0f172a;
    }
    .stButton>button:hover, .stDownloadButton>button:hover {
        background: #2563eb33;
        border-color: #2563eb66;
    }
    /* DataFrameの行を淡くゼブラに */
    [data-testid="stDataFrame"] tbody tr:nth-child(even) {
        background-color: #f8fafc !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("設定")
    st.subheader("離職リスク")
    attr_high_thr = st.slider(
        "High閾値(%)",
        min_value=50,
        max_value=95,
        value=70,
        step=1,
        key="attr_high",
        help="この値以上の離職確率(%)を『高リスク』として判定します。警告表示やPDF集計にも反映されます。",
    )
    attr_med_thr = st.slider(
        "Medium閾値(%)",
        min_value=20,
        max_value=90,
        value=40,
        step=1,
        key="attr_med",
        help="この値以上〜High未満を『中リスク』とします。MediumがHigh以上に設定された場合は自動でHigh-1に調整します。",
    )
    st.subheader("勤怠異常")
    att_z_thr = st.slider(
        "Zスコア閾値",
        min_value=1.5,
        max_value=5.0,
        value=2.0,
        step=0.1,
        key="att_z",
        help="個人の残業時間の平常からのズレの大きさ（平均±z×標準偏差）。この値を超える日は異常候補としてフラグします。",
    )
    att_long_thr = st.slider(
        "長時間勤務(勤務時間h)閾値",
        min_value=8,
        max_value=24,
        value=11,
        step=1,
        key="att_long",
        help="1日の勤務時間がこの時間以上の場合に『長時間勤務』フラグを立てます。",
    )
    att_streak_thr = st.slider(
        "連続出勤日数閾値",
        min_value=6,
        max_value=31,
        value=12,
        step=1,
        key="att_streak",
        help="勤務時間>0の日がこの日数以上連続した場合に『連続出勤』フラグを立てます。",
    )
    st.subheader("プライバシー")
    use_anonymize = st.checkbox(
        "社員IDを匿名化して表示/出力",
        value=False,
        key="anon",
        help="表示・CSV/Excel/PDFのみ匿名化します（判定ロジックは実IDのまま）。環境変数 ANON_SALT でハッシュの塩を設定可能です。",
    )

tab1, tab2, tab3 = st.tabs(["離職リスク予測", "勤怠異常検知", "使い方"])

with tab1:
    st.subheader("離職リスク予測")
    st.markdown(
        """
推奨カラム例（不足していても動作します）

- 社員ID（または一意の識別子）
- 氏名
- 年齢
- 勤続年数
- 平均残業時間h
- 有給取得率
- 評価（1〜5など）
- 昇給回数／部署異動回数
- attrition ラベル（任意：1=離職, 0=在籍）
        """
    )
    hr_up = st.file_uploader("人事データ (CSV/Excel)", type=["csv","xlsx","xls","xlsm"], key="hr")
    use_sample_hr = st.button("サンプルを読み込む", key="load_hr_sample")

    if hr_up or use_sample_hr:
        if use_sample_hr:
            hr_df = read_table_auto(os.path.join("data", "hr_sample.csv"))
            st.info("サンプル人事データを読み込みました。")
        else:
            assert hr_up is not None
            hr_df = read_table_auto(io.BytesIO(hr_up.getvalue()))
        # 先に軽い正規化（列名・値の揃え）を実施
        hr_df = normalize_hr_columns(hr_df)

        # 正規化後に不足があればマッピングUIを提示
        required_hr = ["社員ID","年齢","勤続年数","平均残業時間h","有給取得率","評価(1-5)","昇給回数","部署異動回数"]
        missing_after_norm = [c for c in required_hr if c not in hr_df.columns]
        if missing_after_norm:
            st.warning("列名の揺れを検出しました。標準列名へマッピングしてください。")
            suggested = suggest_column_mapping(list(hr_df.columns), required_hr, domain="hr")
            mapping: dict[str, str] = {}
            with st.expander("列名マッピング（人事）", expanded=True):
                for req in required_hr:
                    options = ["(未選択)"] + list(hr_df.columns)
                    default = suggested.get(req) or "(未選択)"
                    sel = st.selectbox(f"{req} に対応する列", options=options, index=(options.index(default) if default in options else 0))
                    if isinstance(sel, str) and sel != "(未選択)":
                        mapping[req] = sel
            if mapping:
                hr_df = apply_column_mapping(hr_df, mapping)
        # 軽いバリデーション
        v = validate_hr(hr_df)
        for msg in v["warnings"]:
            st.warning(msg)
        if v["errors"]:
            for msg in v["errors"]:
                st.error(msg)
            st.stop()

        with st.spinner("予測中..."):
            result_df, meta = predict_attrition(hr_df)
        # 閾値に基づく区分再計算
        _high = int(attr_high_thr)
        _med = int(attr_med_thr)
        if _med >= _high:
            _med = _high - 1
        result_df["リスク区分"] = pd.cut(result_df["離職確率(%)"],
                                   bins=[-0.1, _med, _high, 100],
                                   labels=["Low","Medium","High"])
        # 表示の匿名化
        display_df = result_df.copy()
        if use_anonymize:
            display_df["社員ID"] = anonymize_ids(pd.Series(display_df["社員ID"]))

        # リスク表示アイコン列を追加（見た目向上）
        risk_icon_map = {"High": "🔴 高", "Medium": "🟠 中", "Low": "🟢 低"}
        display_df["リスク"] = display_df["リスク区分"].astype(str).map(lambda x: risk_icon_map.get(x, ""))
        # 列の並びを調整（社員IDの次にリスク）
        cols = list(display_df.columns)
        if "リスク" in cols:
            cols.remove("リスク")
            cols.insert(1, "リスク")
            display_df = display_df[cols]

        # KPIメトリクス
        total_emp = len(display_df)
        high_cnt = int((result_df["離職確率(%)"] >= _high).sum())
        med_cnt = int(((result_df["離職確率(%)"] >= _med) & (result_df["離職確率(%)"] < _high)).sum())
        median_prob = float(result_df["離職確率(%)"].median())
        avg_ot = float(pd.Series(result_df.get("平均残業時間h", pd.Series([0]))).mean())
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("従業員数", f"{total_emp}")
        c2.metric("High人数", f"{high_cnt}")
        c3.metric("Median確率", f"{median_prob:.1f}%")
        c4.metric("平均残業", f"{avg_ot:.1f}h")

        # 簡易フィルタ（リスク区分）
        with st.expander("表示フィルタ", expanded=False):
            selected_risks = st.multiselect(
                "表示するリスク区分",
                options=["High","Medium","Low"],
                default=["High","Medium","Low"],
                key="risk_filter",
            )
        table_df = pd.DataFrame(display_df)
        if selected_risks:
            mask_risk = table_df["リスク区分"].astype(str).isin(list(map(str, selected_risks)))
            table_df = table_df.loc[mask_risk]

        st.success(f"処理完了：{len(result_df)}名 | モデル: {meta['model']}")
        st.dataframe(
            table_df,
            use_container_width=True,
            column_config={
                "離職確率(%)": st.column_config.ProgressColumn(
                    "離職確率(%)",
                    format="%d%%",
                    min_value=0,
                    max_value=100,
                )
            },
            hide_index=True,
        )

        # ダウンロード（CSV / Excel）
        csv_bytes = table_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button("CSVをダウンロード", data=csv_bytes, file_name="attrition_results.csv", mime="text/csv")

        try:
            import openpyxl  # type: ignore  # noqa: F401
            xls_buf = io.BytesIO()
            with pd.ExcelWriter(cast(Any, xls_buf), engine="openpyxl") as writer:
                table_df.to_excel(writer, index=False, sheet_name="結果")
            xls_buf.seek(0)
            st.download_button("Excelをダウンロード", data=xls_buf.getvalue(), file_name="attrition_results.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        except Exception as ex:
            st.warning("Excel出力には openpyxl が必要です。'pip install openpyxl' で追加してください。")

        high = result_df[result_df["離職確率(%)"] >= _high]
        if not high.empty:
            st.error(f"⚠ 高リスク社員: {len(high)}名（{_high}%以上）")
        comment = generate_comment(result_df, target="attrition")
        st.subheader("AI提案")
        st.write(comment)

        # PDF出力
        pdf_bytes = build_pdf_attrition(result_df, meta, high_threshold=_high, med_threshold=_med, anonymize=bool(use_anonymize))
        st.download_button("PDFレポートをダウンロード", data=pdf_bytes,
                           file_name="離職リスクレポート.pdf", mime="application/pdf")

with tab2:
    st.subheader("勤怠異常検知")
    st.write("必須カラム例：社員ID, 日付(YYYY-MM-DD), 勤務時間h, 残業時間h")
    att_up = st.file_uploader("勤怠データ (CSV/Excel)", type=["csv","xlsx","xls","xlsm"], key="attendance")
    use_sample_att = st.button("サンプルを読み込む", key="load_att_sample")

    if att_up or use_sample_att:
        if use_sample_att:
            att_df = read_table_auto(os.path.join("data", "attendance_sample.csv"), parse_dates=["日付"]) 
            st.info("サンプル勤怠データを読み込みました。")
        else:
            assert att_up is not None
            att_df = read_table_auto(io.BytesIO(att_up.getvalue()), parse_dates=["日付"]) 
        # 正規化（勤怠データは日付型のみ事前整形済み）→ 不足があればマッピングUI
        required_att = ["社員ID","日付","勤務時間h","残業時間h"]
        missing_att = [c for c in required_att if c not in att_df.columns]
        if missing_att:
            st.warning("列名の揺れを検出しました。標準列名へマッピングしてください。")
            suggested = suggest_column_mapping(list(att_df.columns), required_att, domain="attendance")
            mapping2: dict[str, str] = {}
            with st.expander("列名マッピング（勤怠）", expanded=True):
                for req in required_att:
                    options = ["(未選択)"] + list(att_df.columns)
                    default = suggested.get(req) or "(未選択)"
                    sel = st.selectbox(f"{req} に対応する列", options=options, index=(options.index(default) if default in options else 0), key=f"att_map_{req}")
                    if isinstance(sel, str) and sel != "(未選択)":
                        mapping2[req] = sel
            if mapping2:
                att_df = apply_column_mapping(att_df, mapping2)

        # 軽いバリデーション
        v2 = validate_attendance(att_df)
        for msg in v2["warnings"]:
            st.warning(msg)
        if v2["errors"]:
            for msg in v2["errors"]:
                st.error(msg)
            st.stop()

        with st.spinner("異常検知中..."):
            anomalies, overview_fig = detect_attendance_anomalies(
                att_df,
                z_thresh=float(att_z_thr),
                long_hours_threshold=float(att_long_thr),
                streak_threshold=int(att_streak_thr),
            )
        st.plotly_chart(overview_fig, use_container_width=True)
        disp_anom = pd.DataFrame(anomalies.copy())
        if use_anonymize:
            disp_anom["社員ID"] = anonymize_ids(pd.Series(disp_anom["社員ID"]))

        # フラグのアイコン列を追加
        def _flag_row(row: pd.Series) -> str:
            icons = ""
            if bool(row.get("異常_長時間", False)):
                icons += "⏱"
            if bool(row.get("異常_連続", False)):
                icons += "🔁"
            if bool(row.get("異常_残業z", False)):
                icons += "📈"
            return icons
        disp_anom["フラグ"] = disp_anom.apply(_flag_row, axis=1)
        # 列の並び（社員IDの後ろにフラグ）
        cols2 = list(disp_anom.columns)
        if "フラグ" in cols2:
            cols2.remove("フラグ")
            if "社員ID" in cols2:
                idx = cols2.index("社員ID") + 1
            else:
                idx = 0
            cols2.insert(idx, "フラグ")
            disp_anom = disp_anom[cols2]

        # KPIメトリクス
        total_anom = len(disp_anom)
        long_cnt = int(pd.Series(disp_anom.get("異常_長時間", pd.Series([], dtype="bool"))).sum())
        streak_cnt = int(pd.Series(disp_anom.get("異常_連続", pd.Series([], dtype="bool"))).sum())
        z_cnt = int(pd.Series(disp_anom.get("異常_残業z", pd.Series([], dtype="bool"))).sum())
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("異常件数", f"{total_anom}")
        c2.metric("長時間", f"{long_cnt}")
        c3.metric("連続", f"{streak_cnt}")
        c4.metric("Zスコア", f"{z_cnt}")

        # 簡易フィルタ（異常種別）
        with st.expander("表示フィルタ", expanded=False):
            selected_types = st.multiselect(
                "表示する異常種別",
                options=["異常_長時間","異常_連続","異常_残業z"],
                default=["異常_長時間","異常_連続","異常_残業z"],
                key="anom_filter",
            )
        if selected_types:
            mask_series = pd.Series([False] * len(disp_anom))
            for colname in selected_types:
                if colname in disp_anom.columns:
                    mask_series = mask_series | disp_anom[colname].astype(bool)
            table_anom = pd.DataFrame(disp_anom.loc[mask_series])
        else:
            table_anom = pd.DataFrame(disp_anom)

        st.dataframe(table_anom, use_container_width=True, hide_index=True)

        # ダウンロード（CSV / Excel）
        att_csv_bytes = table_anom.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button("CSVをダウンロード", data=att_csv_bytes, file_name="attendance_anomalies.csv", mime="text/csv")

        try:
            import openpyxl  # type: ignore  # noqa: F401
            att_xls_buf = io.BytesIO()
            with pd.ExcelWriter(cast(Any, att_xls_buf), engine="openpyxl") as writer:
                table_anom.to_excel(writer, index=False, sheet_name="異常一覧")
            att_xls_buf.seek(0)
            st.download_button("Excelをダウンロード", data=att_xls_buf.getvalue(), file_name="attendance_anomalies.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        except Exception as ex:
            st.warning("Excel出力には openpyxl が必要です。'pip install openpyxl' で追加してください。")

        if len(anomalies) > 0:
            st.error(f"⚠ 検知した異常: {len(anomalies)}件")

        comment = generate_comment(anomalies, target="attendance")
        st.subheader("AI所見")
        st.write(comment)

        pdf_bytes = build_pdf_attendance(anomalies, anonymize=bool(use_anonymize))
        st.download_button("PDFレポートをダウンロード", data=pdf_bytes,
                           file_name="勤怠異常レポート.pdf", mime="application/pdf")

with tab3:
    st.markdown("""
### 使い方
1. 自社データのCSV/Excelファイルをアップロードしてください。
2. 列名が異なる場合は、画面の「列名マッピング」で標準名に合わせられます。
3. カラムの目安：
   - **離職リスク予測（推奨）**
     - 社員ID（または一意の識別子）
     - 氏名
     - 年齢／勤続年数／平均残業時間h／有給取得率／評価（1〜5など）
     - 昇給回数／部署異動回数
     - attrition ラベル（任意：1=離職, 0=在籍）
   - **勤怠異常検知（必須）**
     - 社員ID, 日付(YYYY-MM-DD), 勤務時間h, 残業時間h
4. OpenAIキー（任意）を `.env` に設定すると、所見がLLM生成に切り替わります（未設定時はルールベース）。
    """)

st.caption("© 人事リスク・勤怠モニター – デモ")
