"""
Figure Factory - Tiện ích tạo và định dạng biểu đồ
Các hàm helper để tạo figure với config mặc định
"""

import plotly.graph_objects as go
from typing import Dict, Any, Optional

# Import configuration
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.styles import DEFAULT_LAYOUT, CHART_CONFIG, DEFAULT_SIZE
from config.translations import UI_TEXT, NUMBER_FORMAT


def create_base_figure(
    title: str,
    subtitle: Optional[str] = None,
    width: int = DEFAULT_SIZE['width'],
    height: int = DEFAULT_SIZE['height'],
    **kwargs
) -> go.Figure:
    """
    Tạo figure cơ bản với layout mặc định
    
    Args:
        title: Tiêu đề biểu đồ
        subtitle: Phụ đề (optional)
        width: Chiều rộng
        height: Chiều cao
        **kwargs: Các tham số layout bổ sung
        
    Returns:
        Plotly Figure object với layout đã cấu hình
    """
    fig = go.Figure()
    
    # Tạo tiêu đề đầy đủ
    full_title = title
    if subtitle:
        full_title = f"{title}<br><sub>{subtitle}</sub>"
    
    # Apply layout mặc định
    layout_config = DEFAULT_LAYOUT.copy()
    layout_config.update({
        'title': {
            'text': full_title,
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'weight': 'bold'}
        },
        'width': width,
        'height': height,
    })
    
    # Merge với kwargs
    layout_config.update(kwargs)
    
    fig.update_layout(**layout_config)
    
    return fig


def format_number(value: float, num_type: str = 'default') -> str:
    """
    Định dạng số theo chuẩn tiếng Việt
    
    Args:
        value: Giá trị cần định dạng
        num_type: Loại định dạng ('default', 'currency', 'percentage', 'short')
        
    Returns:
        Chuỗi đã định dạng
    """
    if num_type == 'currency':
        if value >= 1_000_000_000:
            return f"${value/1_000_000_000:.2f}B"
        elif value >= 1_000_000:
            return f"${value/1_000_000:.2f}M"
        elif value >= 1_000:
            return f"${value/1_000:.2f}K"
        return f"${value:,.0f}"
    
    elif num_type == 'percentage':
        return NUMBER_FORMAT['percentage_format'].format(value)
    
    elif num_type == 'short':
        if value >= 1_000_000:
            return f"{value/1_000_000:.1f}M"
        elif value >= 1_000:
            return f"{value/1_000:.1f}K"
        return f"{value:,.0f}"
    
    else:  # default
        return f"{value:,.0f}".replace(',', '.')


def add_watermark(fig: go.Figure, text: str = "Du-Doan-Tham-Hoa-Toan-Cau") -> go.Figure:
    """
    Thêm watermark vào biểu đồ
    
    Args:
        fig: Figure object
        text: Text watermark
        
    Returns:
        Figure với watermark
    """
    fig.add_annotation(
        text=text,
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=40, color="rgba(128,128,128,0.1)"),
        textangle=-30
    )
    return fig


def create_hover_template(fields: Dict[str, str]) -> str:
    """
    Tạo template cho hover text
    
    Args:
        fields: Dictionary {label: value_format}
        
    Returns:
        Hover template string
    """
    lines = []
    for label, value_format in fields.items():
        lines.append(f"<b>{label}:</b> {value_format}")
    
    return "<br>".join(lines) + "<extra></extra>"


def apply_color_palette(fig: go.Figure, color_map: Dict[str, str]) -> go.Figure:
    """
    Áp dụng bảng màu cho biểu đồ
    
    Args:
        fig: Figure object
        color_map: Dictionary mapping values to colors
        
    Returns:
        Figure với màu sắc đã cập nhật
    """
    for trace in fig.data:
        if hasattr(trace, 'marker') and hasattr(trace, 'name'):
            if trace.name in color_map:
                trace.marker.color = color_map[trace.name]
    
    return fig


def optimize_for_performance(fig: go.Figure, max_points: int = 1000) -> go.Figure:
    """
    Tối ưu biểu đồ cho máy yếu
    
    Args:
        fig: Figure object
        max_points: Số điểm tối đa cho mỗi trace
        
    Returns:
        Figure đã tối ưu
    """
    # Giảm số điểm nếu quá nhiều
    for trace in fig.data:
        if hasattr(trace, 'x') and len(trace.x) > max_points:
            # Sample data
            step = len(trace.x) // max_points
            if hasattr(trace, 'y'):
                trace.x = trace.x[::step]
                trace.y = trace.y[::step]
    
    # Tắt các tính năng nặng
    fig.update_layout(
        dragmode=False,
        hovermode='closest'
    )
    
    return fig


def create_custom_legend(
    fig: go.Figure,
    position: str = 'bottom',
    orientation: str = 'h'
) -> go.Figure:
    """
    Tạo legend tùy chỉnh
    
    Args:
        fig: Figure object
        position: Vị trí ('top', 'bottom', 'left', 'right')
        orientation: Hướng ('h' horizontal, 'v' vertical)
        
    Returns:
        Figure với legend tùy chỉnh
    """
    legend_config = {'orientation': orientation}
    
    if position == 'bottom':
        legend_config.update({'yanchor': 'top', 'y': -0.2, 'xanchor': 'center', 'x': 0.5})
    elif position == 'top':
        legend_config.update({'yanchor': 'bottom', 'y': 1.1, 'xanchor': 'center', 'x': 0.5})
    elif position == 'left':
        legend_config.update({'yanchor': 'middle', 'y': 0.5, 'xanchor': 'right', 'x': -0.05})
    elif position == 'right':
        legend_config.update({'yanchor': 'middle', 'y': 0.5, 'xanchor': 'left', 'x': 1.05})
    
    fig.update_layout(legend=legend_config)
    return fig
