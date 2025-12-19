# 🔧 Hướng Dẫn Fix Lỗi Heatmap

## ❌ Lỗi gặp phải:
```
ValueError: Invalid property specified for object of type plotly.graph_objs.heatmap.ColorBar: 'titleside'
```

## ✅ Đã fix:
File [visualization/config/styles.py](visualization/config/styles.py) đã được sửa:
- **Trước**: `'titleside': 'right'` (không hợp lệ)
- **Sau**: `'title': {'text': 'Số sự kiện', 'side': 'right'}` (đúng cú pháp Plotly)

## 📝 Cách chạy sau khi fix:

### Bước 1: Restart Kernel
**QUAN TRỌNG**: Jupyter đã cache code cũ, cần restart để load code mới:

1. **Cách 1**: Dùng Command Palette
   - Nhấn `Ctrl+Shift+P` (Windows/Linux) hoặc `Cmd+Shift+P` (Mac)
   - Gõ "Restart Kernel"
   - Chọn "Notebook: Restart Kernel"
   - Nhấn Enter

2. **Cách 2**: Dùng Menu
   - Click menu `Kernel` → `Restart Kernel`

3. **Cách 3**: Dùng icon
   - Tìm icon ⟲ (Restart) trên toolbar của notebook

### Bước 2: Chạy lại cells
Sau khi restart, chạy tuần tự các cells:
1. Cell 2: Import thư viện
2. Cell 3: Đọc dữ liệu
3. Cell 4-19: Các visualization (bao gồm Heatmap)

### Bước 3: Xác nhận fix thành công
Nếu heatmap chạy thành công, bạn sẽ thấy:
- Heatmap 10x12 (10 loại thảm họa × 12 tháng)
- Colorbar với title "Số sự kiện" ở bên phải
- Không có error message

## 🧪 Test độc lập (nếu cần):
Nếu vẫn gặp lỗi trong notebook, thử test bằng script:
```bash
python test_heatmap.py
```

Kết quả mong đợi:
```
Testing heatmap visualization...
✅ Heatmap created successfully!
   Shape: (10, 12)
   X labels: ('T1', 'T2', ..., 'T12')
   Y labels: ['Bão', 'Bão biển', ..., 'Động đất']
```

## 💡 Tại sao cần Restart Kernel?

Python/Jupyter **cache modules** khi import lần đầu:
- Khi bạn chạy `from visualization.config.styles import HEATMAP_CONFIG`, Python đọc file và lưu vào memory
- Khi file thay đổi, Python **không tự động reload**
- Cần restart kernel để xóa cache và đọc lại file mới

## 🔍 Chi tiết kỹ thuật:

### Cú pháp sai (cũ):
```python
HEATMAP_CONFIG = {
    'colorbar': {
        'title': 'Số sự kiện',      # String đơn giản
        'titleside': 'right',        # ❌ Property không tồn tại!
    }
}
```

### Cú pháp đúng (mới):
```python
HEATMAP_CONFIG = {
    'colorbar': {
        'title': {                   # ✅ Nested object
            'text': 'Số sự kiện',
            'side': 'right'
        },
    }
}
```

Theo [Plotly documentation](https://plotly.com/python/reference/heatmap/#heatmap-colorbar-title), `title` phải là object với properties:
- `text`: Nội dung title
- `side`: Vị trí ('top', 'bottom', 'right')
- `font`: Font settings (optional)

## ✅ Kiểm tra nhanh:
Sau khi restart kernel và chạy lại, heatmap sẽ hoạt động bình thường!
