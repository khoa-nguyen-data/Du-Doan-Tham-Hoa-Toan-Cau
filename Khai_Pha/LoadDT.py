import pandas as pd
import numpy as np


def load_data(
    path: str,
    has_double_header: bool = True,
    dayfirst: bool = True
) -> pd.DataFrame:
    """
    Load và chuẩn hóa dataset thảm họa toàn cầu.

    Parameters
    ----------
    path : str
        Đường dẫn tới file CSV
    has_double_header : bool, default=True
        Dataset có 2 dòng header (Column1.. + header thật)
    dayfirst : bool, default=True
        Ngày dạng dd/mm/yyyy

    Returns
    -------
    df : pd.DataFrame
        DataFrame đã chuẩn hóa kiểu dữ liệu cơ bản
    """

    # -----------------------------
    # 1. Load raw CSV
    # -----------------------------
    if has_double_header:
        raw = pd.read_csv(path, header=None)

        # Lấy header thật ở dòng 2
        header = raw.iloc[1].tolist()
        df = raw.iloc[2:].copy()
        df.columns = header
    else:
        df = pd.read_csv(path)

    # -----------------------------
    # 2. Chuẩn hóa tên cột
    # -----------------------------
    df.columns = df.columns.str.strip()

    # -----------------------------
    # 3. Parse date
    # -----------------------------
    if "date" in df.columns:
        df["date"] = pd.to_datetime(
            df["date"],
            dayfirst=dayfirst,
            errors="coerce"
        )

    # -----------------------------
    # 4. Convert numeric columns
    # -----------------------------
    num_cols = [
        "severity_index",
        "casualties",
        "economic_loss_usd",
        "response_time_hours",
        "aid_amount_usd",
        "response_efficiency_score",
        "recovery_days",
        "latitude",
        "longitude"
    ]

    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df
