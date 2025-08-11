# 人事分析ツール - HR Risk & Attendance Monitor

AI＋Pythonで、離職リスク予測と勤怠異常検知を自動化するWebアプリケーション

## 機能

- **離職リスク予測**: 機械学習による離職確率の算出
- **勤怠異常検知**: Zスコア、長時間勤務、連続出勤の異常検知
- **データ柔軟性**: CSV/Excel対応、列名マッピング機能
- **プライバシー保護**: 社員ID匿名化機能
- **レポート生成**: PDFレポート出力（日本語対応）
- **AI所見**: OpenAI API連携による自動所見生成

## ローカル実行

```bash
# 依存関係インストール
pip install -r requirements.txt

# アプリケーション起動
streamlit run app.py
```

## Streamlit.ioでの公開手順

### 1. Gitリポジトリの準備

```bash
# Gitリポジトリ初期化（まだの場合）
git init

# ファイルをステージング
git add .

# 初回コミット
git commit -m "Initial commit: HR analysis tool"

# GitHubでリポジトリ作成後、リモート追加
git remote add origin https://github.com/yourusername/hr-analysis-tool.git

# プッシュ
git branch -M main
git push -u origin main
```

### 2. 必要なファイルの確認

公開に必要なファイルが揃っているか確認：

- ✅ `app.py` - メインアプリケーション
- ✅ `requirements.txt` - 依存関係
- ✅ `modules/` - 各機能モジュール
- ✅ `data/` - サンプルデータ
- ✅ `.gitignore` - Git除外設定

### 3. Streamlit.ioでの公開

1. **Streamlit.ioにアクセス**
   - https://share.streamlit.io/ にアクセス
   - GitHubアカウントでログイン

2. **アプリケーション作成**
   - "New app" をクリック
   - Repository: `yourusername/hr-analysis-tool`
   - Branch: `main`
   - Main file path: `app.py`
   - App URL: 任意のサブドメイン名

3. **デプロイ**
   - "Deploy!" をクリック
   - 初回ビルドには数分かかります

### 4. 環境変数の設定（オプション）

AI所見機能を使用する場合：

1. Streamlit.ioの管理画面で "Secrets" を開く
2. 以下の形式で設定：

```toml
OPENAI_API_KEY = "your-openai-api-key"
ANON_SALT = "your-anonymization-salt"
```

### 5. 更新手順

```bash
# コード変更後
git add .
git commit -m "Update: 機能改善"
git push origin main
```

Streamlit.ioは自動的にGitHubの変更を検知して再デプロイします。

## ファイル構成

```
人事分析ツール/
├── app.py                 # メインアプリケーション
├── requirements.txt       # 依存関係
├── README.md             # このファイル
├── .gitignore           # Git除外設定
├── modules/
│   ├── __init__.py
│   ├── risk_model.py     # 離職リスク予測
│   ├── attendance_check.py # 勤怠異常検知
│   ├── ai_comment.py     # AI所見生成
│   ├── report.py         # PDFレポート生成
│   └── utils.py          # ユーティリティ関数
└── data/
    ├── hr_sample.csv     # 人事サンプルデータ
    └── attendance_sample.csv # 勤怠サンプルデータ
```

## 技術スタック

- **フロントエンド**: Streamlit
- **データ処理**: Pandas, NumPy
- **機械学習**: scikit-learn
- **可視化**: Plotly
- **PDF生成**: ReportLab
- **AI**: OpenAI API (オプション)

## 注意事項

- サンプルデータは公開用に調整済み
- 本番データ使用時は適切なプライバシー保護を実施
- OpenAI API使用時は適切なAPI制限とコスト管理を実施
