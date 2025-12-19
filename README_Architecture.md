# 📦 Cấu Trúc Dự Án Trực Quan Dữ Liệu Thảm Họa

## 🏗️ Kiến Trúc Production-Grade

Dự án được tổ chức theo chuẩn **Plotly Dash** và **Apache Superset**, với phân tách rõ ràng giữa:
- **Config**: Cấu hình styles, translations, constants
- **Utils**: Tiện ích tái sử dụng (data loading, figure factory)
- **Components**: Các module trực quan hóa độc lập

```
visualization/
├── config/
│   ├── __init__.py
│   ├── styles.py          # Màu sắc, themes, layout config
│   └── translations.py    # Từ điển tiếng Việt, UI text
├── utils/
│   ├── __init__.py
│   ├── data_loader.py     # Load data với caching
│   └── figure_factory.py  # Helper functions cho charts
└── components/
    ├── __init__.py
    ├── disaster_distribution.py   # Sunburst + Bar
    ├── world_map.py              # Scatter geo map
    ├── trends_analysis.py        # Line charts (yearly/monthly)
    ├── economic_impact.py        # Dual bar charts
    ├── top_countries.py          # Treemap + Donut
    ├── response_efficiency.py    # Scatter + Radar
    ├── monthly_heatmap.py        # Heatmap
    └── dashboard_overview.py     # 6-panel dashboard
```

## 🎨 Đặc Điểm Nổi Bật

### 1. **Tối Ưu Hiệu Năng**
- ✅ `@lru_cache` cho data loading (tránh đọc file nhiều lần)
- ✅ Cấu hình rendering nhẹ (`plotly_white`, `staticPlot=False`)
- ✅ Giảm số điểm dữ liệu khi cần (`optimize_for_performance`)
- ✅ Tắt các tính năng nặng (scroll zoom, complex hover)

### 2. **Hoàn Toàn Tiếng Việt**
- ✅ Tất cả UI text trong `config/translations.py`
- ✅ Loại thảm họa được dịch: "Earthquake" → "Động đất"
- ✅ Tên tháng tiếng Việt: "January" → "Tháng 1"
- ✅ Hover templates và labels đều tiếng Việt

### 3. **Code Chuyên Nghiệp**
- ✅ Type hints cho tất cả functions
- ✅ Docstrings chi tiết (Args, Returns, Cache)
- ✅ Separation of concerns (config/utils/components)
- ✅ Standalone test cho mỗi component (`if __name__ == "__main__"`)
- ✅ Reusable helpers (`create_base_figure`, `format_number`)

### 4. **Dễ Bảo Trì & Mở Rộng**
- ✅ Mỗi visualization trong file riêng
- ✅ Centralized configuration (thay đổi 1 chỗ → áp dụng toàn bộ)
- ✅ Import paths rõ ràng: `from visualization.components import *`
- ✅ Consistent naming convention

## 📊 Các Component Visualization

| Component | Charts | Description |
|-----------|--------|-------------|
| **disaster_distribution** | Sunburst + Bar | Phân bố loại thảm họa |
| **world_map** | Scatter Geo | Bản đồ thảm họa toàn cầu |
| **trends_analysis** | Line (x2) | Xu hướng yearly + monthly |
| **economic_impact** | Horizontal Bar (x2) | Thiệt hại kinh tế + thương vong |
| **top_countries** | Treemap + Donut | Top 10 quốc gia |
| **response_efficiency** | Scatter + Radar | Hiệu quả ứng phó |
| **monthly_heatmap** | Heatmap | Phân bố theo tháng |
| **dashboard_overview** | 6 panels | Tổng quan toàn diện |

## 🚀 Sử Dụng

### Trong Notebook (TrucQuan.ipynb)
```python
from visualization.components.disaster_distribution import visualize_disaster_distribution

fig1, fig2 = visualize_disaster_distribution()
fig1.show(config={'displayModeBar': False})
fig2.show(config={'displayModeBar': False})
```

### Standalone Testing
```bash
python visualization/components/disaster_distribution.py
```

### Custom Configuration
```python
from visualization.config.styles import COLOR_PALETTE, DEFAULT_LAYOUT
from visualization.utils.data_loader import load_data, filter_data

# Load và lọc data
df = load_data()
filtered = filter_data(df, years=[2023, 2024], min_severity=5.0)

# Custom colors
COLOR_PALETTE['Động đất'] = '#FF0000'
```

## 🎯 Best Practices Áp Dụng

### Từ Plotly Dash Sample Apps:
- ✅ Functional components với clear inputs/outputs
- ✅ Separate `layout_helper` pattern (→ `figure_factory`)
- ✅ `utils/` folder cho reusable logic
- ✅ `config/` cho constants và settings
- ✅ Standalone testable modules

### Từ Apache Superset:
- ✅ Plugin-like architecture (mỗi viz = 1 component)
- ✅ Metadata-driven configuration (UI_TEXT dictionary)
- ✅ Separation: buildQuery → transformProps (→ load_data → visualize)

### Performance Optimization:
- ✅ Data caching với `functools.lru_cache`
- ✅ Lazy imports khi cần
- ✅ Reduced rendering complexity
- ✅ Responsive design với flexible sizing

## 📝 Ghi Chú Kỹ Thuật

1. **Data Loading**: 
   - File CSV được cache sau lần đọc đầu tiên
   - Sử dụng `@lru_cache(maxsize=1)` trong `data_loader.py`

2. **Color Management**:
   - Centralized trong `COLOR_PALETTE` (config/styles.py)
   - Consistent across tất cả components

3. **Text Management**:
   - Tất cả UI text trong `UI_TEXT` (config/translations.py)
   - Dễ dàng internationalization (i18n) trong tương lai

4. **Figure Factory**:
   - `create_base_figure()`: Template với layout mặc định
   - `format_number()`: Định dạng số tiếng Việt (1.000.000 thay vì 1,000,000)
   - `optimize_for_performance()`: Giảm complexity cho máy yếu

## 🔧 Maintenance

### Thêm Visualization Mới:
1. Tạo file trong `visualization/components/new_viz.py`
2. Import utils và config cần thiết
3. Implement function `visualize_xyz()` với docstring
4. Thêm vào `components/__init__.py`
5. Update notebook để import

### Thay Đổi Màu Sắc:
Sửa `visualization/config/styles.py`:
```python
COLOR_PALETTE['Động đất'] = '#NewColor'
```

### Thêm Text Mới:
Sửa `visualization/config/translations.py`:
```python
UI_TEXT['new_label'] = 'Text tiếng Việt'
```

## 📚 Tài Liệu Tham Khảo

- [Plotly Dash Sample Apps](https://github.com/plotly/dash-sample-apps)
- [Apache Superset Architecture](https://github.com/apache/superset)
- [Plotly Express Documentation](https://plotly.com/python/plotly-express/)
- TrucQuan_Top_Quoc_Gia.py (reference implementation)

---

**Version**: 2.0 (Production-Grade Architecture)  
**Last Updated**: December 2025  
**Tối ưu cho**: Máy yếu + Jupyter Notebook  
**Ngôn ngữ**: 100% Tiếng Việt
