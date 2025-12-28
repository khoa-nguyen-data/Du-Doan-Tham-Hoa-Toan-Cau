"""
Data Loader - Tải và xử lý dữ liệu thảm họa
Có caching để tối ưu hiệu năng cho máy yếu
"""

import pandas as pd
import warnings
from typing import Tuple, Optional
from functools import lru_cache

warnings.filterwarnings('ignore')

# Import configuration
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.translations import DISASTER_VN


@lru_cache(maxsize=1)
def load_data(file_path: str = 'global_disaster_response_2018_2024.csv') -> pd.DataFrame:
    """
    Tải và làm sạch dữ liệu thảm họa toàn cầu
    
    Args:
        file_path: Đường dẫn đến file CSV
        
    Returns:
        DataFrame đã được làm sạch và dịch sang tiếng Việt
        
    Cache:
        Sử dụng lru_cache để tránh đọc file nhiều lần
    """
    # Đọc dữ liệu
    df = pd.read_csv(file_path)
    
    # Chuyển đổi định dạng ngày
    df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['month_name'] = df['date'].dt.strftime('%B')
    
    # Làm sạch dữ liệu - loại bỏ giá trị bất thường
    df = df[df['severity_index'] > 0].copy()
    df = df[df['casualties'] >= 0].copy()
    df = df[df['economic_loss_usd'] >= 0].copy()
    
    # Dịch tên loại thảm họa sang tiếng Việt
    df['loai_tham_hoa'] = df['disaster_type'].map(DISASTER_VN)
    
    return df


def get_data_info(df: pd.DataFrame) -> dict:
    """
    Lấy thông tin tổng quan về dữ liệu
    
    Args:
        df: DataFrame chứa dữ liệu thảm họa
        
    Returns:
        Dictionary chứa các thông tin thống kê
    """
    info = {
        'total_events': len(df),
        'num_disaster_types': df['loai_tham_hoa'].nunique(),
        'num_countries': df['country'].nunique(),
        'date_range': (
            df['date'].min().strftime('%d/%m/%Y'),
            df['date'].max().strftime('%d/%m/%Y')
        ),
        'total_casualties': df['casualties'].sum(),
        'total_economic_loss': df['economic_loss_usd'].sum(),
        'avg_severity': df['severity_index'].mean(),
        'disaster_types': df['loai_tham_hoa'].unique().tolist(),
        'countries': df['country'].unique().tolist()
    }
    
    return info


def filter_data(
    df: pd.DataFrame,
    disaster_types: Optional[list] = None,
    countries: Optional[list] = None,
    years: Optional[list] = None,
    min_severity: Optional[float] = None
) -> pd.DataFrame:
    """
    Lọc dữ liệu theo các tiêu chí
    
    Args:
        df: DataFrame gốc
        disaster_types: Danh sách loại thảm họa (tiếng Việt)
        countries: Danh sách quốc gia
        years: Danh sách năm
        min_severity: Mức độ nghiêm trọng tối thiểu
        
    Returns:
        DataFrame đã được lọc
    """
    filtered_df = df.copy()
    
    if disaster_types:
        filtered_df = filtered_df[filtered_df['loai_tham_hoa'].isin(disaster_types)]
    
    if countries:
        filtered_df = filtered_df[filtered_df['country'].isin(countries)]
    
    if years:
        filtered_df = filtered_df[filtered_df['year'].isin(years)]
    
    if min_severity is not None:
        filtered_df = filtered_df[filtered_df['severity_index'] >= min_severity]
    
    return filtered_df


def aggregate_by_disaster_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tổng hợp dữ liệu theo loại thảm họa
    
    Args:
        df: DataFrame gốc
        
    Returns:
        DataFrame tổng hợp với các chỉ số chính
    """
    agg_df = df.groupby('loai_tham_hoa').agg({
        'date': 'count',  # Đếm số sự kiện
        'casualties': 'sum',
        'economic_loss_usd': 'sum',
        'severity_index': 'mean',
        'response_time_hours': 'mean',
        'recovery_days': 'mean'
    }).reset_index()
    
    agg_df.columns = [
        'loai_tham_hoa', 'so_su_kien', 'tong_thuong_vong',
        'tong_thiet_hai', 'muc_do_nghiem_trong_tb',
        'thoi_gian_ung_pho_tb', 'thoi_gian_phuc_hoi_tb'
    ]
    
    return agg_df.sort_values('so_su_kien', ascending=False)


def aggregate_by_country(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    Tổng hợp dữ liệu theo quốc gia
    
    Args:
        df: DataFrame gốc
        top_n: Số lượng quốc gia top cần lấy
        
    Returns:
        DataFrame tổng hợp top N quốc gia
    """
    agg_df = df.groupby('country').agg({
        'date': 'count',  # Đếm số sự kiện
        'casualties': 'sum',
        'economic_loss_usd': 'sum',
        'severity_index': 'mean'
    }).reset_index()
    
    agg_df.columns = [
        'country', 'so_su_kien', 'tong_thuong_vong',
        'tong_thiet_hai', 'muc_do_nghiem_trong_tb'
    ]
    
    return agg_df.sort_values('so_su_kien', ascending=False).head(top_n)
