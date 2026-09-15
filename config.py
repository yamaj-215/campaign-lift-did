"""分析全体の設定値。

分析前に固定し、以後は結果を見てから動かさない値をここに集約する。
期間や判定基準をコード中に直書きしないことで、後付けの解釈調整を防ぐ。
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
TABLES = ROOT / "output" / "tables"
FIGURES = ROOT / "output" / "figures"

LOG_CSV = RAW / "log_data_unpivoted.csv"
MASTER_CSV = RAW / "prefecture_master.csv"
ASSIGN_CSV = RAW / "prefectures_control_test.csv"

# --- 期間定義（事前登録相当） ---
PRE_START = "2025-05-01"
PRE_END = "2025-07-31"
POST_START = "2025-08-01"
POST_END = "2025-08-31"

# --- 評価指標 ---
PRIMARY = "entries"       # 主要：エントリー数
SECONDARY = "interviews"  # 副次：面談実施数

# --- 推論の設定 ---
CLUSTER = "pref"        # 推論単位は都道府県クラスタ。行単位ではない
ALPHA = 0.05            # 有意水準（両側）
N_PERMUTATIONS = 10_000
N_BOOTSTRAP = 10_000
SEED = 20250902         # データ集計基準日。再現性のため固定

# --- 既往キャンペーン（交絡確認・作図用。両群に配信された前提） ---
PAST_CAMPAIGNS = [
    ("2025-05-12", "2025-05-31", "20代対象求人特集"),
    ("2025-06-09", "2025-06-20", "ハイクラス転職"),
    ("2025-07-01", "2025-07-14", "業界経験者向け非公開求人"),
]

# --- 期待されるデータ構造（前処理の検証に使う） ---
EXPECTED_N_PREF = 47
EXPECTED_N_DAYS = 123
EXPECTED_N_TREAT = 32
EXPECTED_N_CONTROL = 15
EXPECTED_RAW_ROWS = 134_605
