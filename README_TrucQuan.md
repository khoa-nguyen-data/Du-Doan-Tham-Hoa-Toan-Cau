# 📊 Hướng Dẫn Sử Dụng Trực Quan Dữ Liệu

## 🎯 Mục đích
Hệ thống trực quan hóa dữ liệu thảm họa toàn cầu 2018-2024 với giao diện tiếng Việt, đẹp và dễ nhìn.

## 📁 Cấu trúc Files

### 1. **TrucQuan.ipynb** - Notebook chính
File notebook chứa toàn bộ trực quan tổng hợp:
- Phân bố loại thảm họa
- Bản đồ thế giới
- Xu hướng theo thời gian
- Heatmap theo tháng
- Dashboard tổng quan

### 2. **TrucQuan_PhanTich_Kinh_Te.py** - File riêng về Kinh tế
Phân tích chi tiết:
- Thiệt hại kinh tế theo loại thảm họa
- Số người thương vong
- So sánh và đối chiếu

**Chạy file:**
```bash
python TrucQuan_PhanTich_Kinh_Te.py
```

### 3. **TrucQuan_Top_Quoc_Gia.py** - File riêng về Quốc gia
Trực quan top quốc gia:
- Top 10 quốc gia bị ảnh hưởng nhiều nhất
- Treemap theo mức độ nghiêm trọng
- Donut chart phân bố %

**Chạy file:**
```bash
python TrucQuan_Top_Quoc_Gia.py
```

## 🎨 Đặc điểm Trực quan

### ✨ Giao diện
- **Tiếng Việt hoàn toàn**: Tất cả labels, titles, hover text
- **Màu sắc hài hòa**: Palette Pastel, Set2, Sunset
- **Spacing tốt**: Không bị chèn chữ lên nhau
- **Font rõ ràng**: Arial 10-12pt, dễ đọc

### 📐 Layout được tối ưu
- Margin phù hợp: `t=60-70, b=50-60, l=60-140, r=60-150`
- Spacing giữa subplots: `horizontal_spacing=0.15`
- Text angle: `-45°` cho trục x dài
- Legend: đặt bên phải hoặc góc trên

### 🔍 Hover Text
- Format số: `:,` (phân cách hàng nghìn)
- Format tiền: `$xxx.xxB` (tỷ USD)
- Thông tin đầy đủ nhưng gọn gàng

## 🚀 Cách Sử Dụng

### Chạy Notebook (Đầy đủ)
```bash
# Kích hoạt môi trường
.venv\Scripts\Activate.ps1

# Mở Jupyter
jupyter notebook TrucQuan.ipynb
```

### Chạy File Python Riêng Lẻ
```bash
# Kinh tế
python TrucQuan_PhanTich_Kinh_Te.py

# Quốc gia
python TrucQuan_Top_Quoc_Gia.py
```

## 📊 Dữ Liệu

### Đã được làm sạch
- ✅ Loại bỏ outliers (severity_index < 0)
- ✅ Kiểm tra casualties >= 0
- ✅ Kiểm tra economic_loss >= 0
- ✅ Dịch tên loại thảm họa sang tiếng Việt

### Mapping tiếng Việt
```python
'Earthquake' → 'Động đất'
'Flood' → 'Lũ lụt'
'Hurricane' → 'Bão'
'Wildfire' → 'Cháy rừng'
'Drought' → 'Hạn hán'
'Tornado' → 'Lốc xoáy'
'Landslide' → 'Sạt lở đất'
'Extreme Heat' → 'Nắng nóng cực đoan'
'Storm Surge' → 'Bão biển'
'Volcanic Eruption' → 'Núi lửa'
```

## 🎯 Tối Ưu Hiệu Năng

### Đã áp dụng
- ✅ Sampling 30% cho bản đồ (nếu > 5000 records)
- ✅ Top 5 loại thảm họa cho line chart
- ✅ Top 3 cho box plot
- ✅ Renderer: `notebook` mode
- ✅ DisplayModeBar: `False` (ẩn toolbar)

### Nếu máy vẫn yếu
Thêm vào cell đầu tiên:
```python
pio.renderers.default = 'png'  # Static image
```

## 📝 Lưu ý

1. **Text bị chèn?** → Tăng margin left/right
2. **Legend bị che?** → Thay đổi `orientation='v'` và `x=1.02`
3. **Chữ quá nhỏ?** → Tăng `font size` lên 12-13
4. **Màu không đẹp?** → Thử colorscale khác: `Viridis`, `Plasma`, `Turbo`

## 🔧 Tùy Chỉnh

### Thay đổi màu sắc
```python
color_discrete_sequence=px.colors.qualitative.Set2  # Pastel, Bold, Set3
```

### Thay đổi kích thước
```python
height=500  # Điều chỉnh theo màn hình
```

### Thay đổi font
```python
font=dict(family="Arial, sans-serif", size=12)
```

## 📞 Hỗ Trợ

Nếu gặp vấn đề:
1. Kiểm tra file CSV có đúng format không
2. Đảm bảo đã cài đặt: `plotly`, `pandas`, `numpy`
3. Chạy lại cell import thư viện
4. Clear output và chạy lại từ đầu

---
🎉 **Chúc bạn trực quan dữ liệu thành công!**
