# 🔄 FlowRevamp – Số hóa & Tái cấu trúc Flowchart từ ảnh

Hệ thống chuyển đổi ảnh flowchart (PNG/JPG) thành đồ thị có cấu trúc (JSON), sử dụng **thị giác máy tính cổ điển**: Adaptive Thresholding đa ngưỡng, phân tích Contour Hierarchy, khớp hình học kiểu RANSAC, và thuật toán dò đường **Pairwise ROI Pathfinding**. Không cần mô hình Deep Learning (ngoại trừ EasyOCR cho bước nhận dạng chữ, có thể tắt).

---

## 📋 Mục lục

- [Tổng quan](#-tổng-quan)
- [Kiến trúc pipeline](#-kiến-trúc-pipeline-4-bước)
- [Cài đặt](#-cài-đặt)
- [Hướng dẫn sử dụng](#-hướng-dẫn-sử-dụng)
- [Cấu trúc thư mục](#-cấu-trúc-thư-mục)
- [Cấu hình](#%EF%B8%8F-cấu-hình)
- [Giao diện chỉnh sửa (UI)](#-giao-diện-chỉnh-sửa-ui)
- [Định dạng JSON đầu ra](#-định-dạng-json-đầu-ra)
- [Baseline so sánh](#-baseline-so-sánh-naivepy)
- [Xử lý sự cố](#-xử-lý-sự-cố)

---

## 🎯 Tổng quan

Đầu vào là một ảnh flowchart, đầu ra gồm:

1. **File JSON** chứa cấu trúc đồ thị (`nodes` + `edges`) tại `data/output/<tên-ảnh>_graph.json`
2. **Ảnh trực quan hóa** với node và đường nối vẽ đè lên ảnh gốc tại `data/output/<tên-ảnh>_result.png`
3. **Giao diện web Streamlit** để con người hiệu chỉnh kết quả (Human-in-the-loop)

Toàn bộ pipeline chạy trên CPU. EasyOCR là thành phần ML duy nhất và mặc định **đang tắt** (`ENABLE_OCR = False`).

---

## 🧠 Kiến trúc pipeline (4 bước)

Pipeline được điều phối bởi `orchestrator.py`, chạy tuần tự 4 bước. Bước 2 và 3 mỗi bước chia thành pha thô (a) và pha tinh chỉnh (b).

### Bước 1 – Tiền xử lý (`steps/step1_preprocess.py`)

- **Adaptive Thresholding cục bộ** (`cv2.adaptiveThreshold`, Gaussian, block size 15) nhị phân hóa ảnh với **2 ngưỡng hằng số C song song**:
  - **High-C** (`C = 20`): khử bóng mờ và nhiễu nền → phục vụ phát hiện node (Bước 2).
  - **Low-C** (`C = 4`): giữ tối đa nét vẽ mảnh/mờ → phục vụ dò đường nối (Bước 3).
- Mỗi phiên bản được đóng hình thái học (`MORPH_CLOSE`, kernel 3×3) rồi **làm mảnh nét về 1-pixel** bằng `skimage.morphology.skeletonize`.
- Đầu ra: `<tên-ảnh>_preprocessed.png` (High-C) và `<tên-ảnh>_preprocessed_low_c.png` (Low-C) trong `data/step1_preprocessed/`.

### Bước 2a – Phát hiện Node (`steps/step2_detect_nodes.py`)

Chạy trên skeleton High-C:

1. **Khử đầu mũi tên kín:** quét trước các contour nhỏ (20–1500 px²) có 3–5 đỉnh; tam giác lồi hoặc chevron lõm bị tô đen để không bị nhầm thành node.
2. **Phân tích phả hệ contour:** `cv2.findContours` với `RETR_TREE`; các node flowchart xuất hiện dưới dạng **contour con** bên trong contour bao ngoài lớn nhất.
3. **Khớp hình học kiểu RANSAC** (`utils/ransac_shapes.py`): thử khớp từng contour vào 3 mô hình — hình chữ nhật (`minAreaRect`), hình thoi trục-thẳng (từ trung điểm bounding box), hình tròn (`minEnclosingCircle`) — rồi tính **tỷ lệ inlier** (điểm cách biên hình < 3 px) và chọn mô hình có tỷ lệ cao nhất.
4. **Phân loại bằng Fill Ratio:** `diện tích contour / diện tích bounding box` — hình thoi ≈ 50%, hình chữ nhật ≥ 75%. Kết hợp số đỉnh (`approxPolyDP`) và circularity để phân biệt Process / Decision / Terminal / Connector.
5. **Trích xuất lặp (Iterative Masking):** node vừa tìm được lập tức bị tô đen với **margin động theo Distance Transform** (tự thích ứng độ dày nét vẽ), rồi lặp lại quét — tối đa 20 vòng — cho đến khi không còn contour thỏa diện tích tối thiểu.

### Bước 2b – Tinh chỉnh Node (lọc chồng lấp)

Node nào chồng lấp đáng kể (giao > 10% diện tích nhỏ hơn) với **≥ 2 node khác** — điển hình là khung trang trí "khổng lồ" bao trùm nhiều node — bị gán `is_bad = true`, type `"Bad"`, và loại khỏi bước dò cạnh.

### Bước 3a – Phát hiện đường nối (`steps/step3_detect_edges.py`)

Chạy trên skeleton Low-C, thuật toán **Pairwise ROI Pathfinding**:

1. Tô đen toàn bộ vùng node hợp lệ (padding 3 px) để cô lập các nét nối.
2. Với **mỗi cặp node** cách nhau ≤ 600 px: cắt ROI là bounding box hợp của 2 node (padding 5 px).
3. Tìm contour trong ROI; một contour tạo thành cạnh nối nếu nó **chạm cả 2 node** (có điểm nằm trong vùng snap 15 px quanh mỗi bounding box).
4. **Xác định hướng:** quét trước các đầu mũi tên (contour 3–5 đỉnh, 30–800 px², nằm ngoài node); mũi tên trong ROI gần node nào hơn thì node đó là **đích** (target).
5. Quỹ đạo nét vẽ thực tế được làm mịn bằng `approxPolyDP` và giữ lại (trường `path`) để trực quan hóa.

### Bước 3b – Tinh chỉnh cạnh (nét vẽ mồ côi)

Sau khi xóa toàn bộ node hợp lệ và các contour cạnh đã dùng, những nét vẽ còn sót có bounding box > 40 px được thêm vào danh sách node dưới dạng **Bad Node** để người dùng kiểm tra thủ công.

### Bước 4 – OCR (`steps/step4_ocr.py`)

Chỉ chạy khi `ENABLE_OCR = True`. Mỗi node được cắt từ **ảnh gốc** với padding 8%, đưa qua **EasyOCR** (`["en", "vi"]`) để lấy trường `text`. Đầu ra: `<tên-ảnh>_ocr.json` và ảnh debug `<tên-ảnh>_ocr_debug.png` trong `data/step4_ocr/`.

### Hợp nhất & Trực quan hóa (`orchestrator.py`, `utils/visualization.py`)

Orchestrator gộp kết quả thành JSON cuối (`data/output/`). Trực quan hóa vẽ:
- Bounding box node: **xanh lá** (hợp lệ) / **đỏ** (Bad Node), kèm nhãn `id:type "text"`.
- Cạnh: tô màu magenta trực tiếp lên quỹ đạo nét vẽ thực tế bằng `cv2.fillPoly` (hiệu ứng "neon wire"); **chấm đỏ** đặt tại tâm node đích để chỉ hướng.

---

## 🚀 Cài đặt

### Yêu cầu

- **Python** ≥ 3.10
- **GPU** (tùy chọn) – chỉ dùng cho EasyOCR khi `OCR_GPU = True`

### Các bước

```bash
cd ~/StudioProjects/FlowRevamp

# Tạo môi trường ảo (khuyến nghị)
python -m venv .venv
source .venv/bin/activate

# Cài dependencies
pip install -r requirements.txt
```

> **Lưu ý:** OCR mặc định đang tắt. Nếu bật (`ENABLE_OCR = True` trong `config.py`), lần chạy đầu EasyOCR sẽ tự tải mô hình (~100 MB).

---

## 📖 Hướng dẫn sử dụng

### 1. Xử lý một ảnh

```bash
cp path/to/flowchart.png data/input/
python main.py data/input/flowchart.png
```

**Kết quả:**
- `data/output/flowchart_graph.json` – cấu trúc đồ thị
- `data/output/flowchart_result.png` – ảnh trực quan hóa

### 2. Xử lý hàng loạt

```bash
cp *.png data/input/
python main.py          # Xử lý mọi ảnh trong data/input/
```

Ảnh nào lỗi sẽ được log và bỏ qua, không dừng cả batch.

### 3. Các tùy chọn CLI

```bash
python main.py [IMAGE] [-v] [--max-node-size WIDTH,HEIGHT]
```

| Tùy chọn | Ý nghĩa |
|---|---|
| `IMAGE` | Đường dẫn 1 ảnh; bỏ trống để xử lý cả `data/input/` |
| `-v`, `--verbose` | Bật log DEBUG (chi tiết từng vòng lặp detect) |
| `--max-node-size W,H` | Ghi đè `MAX_NODE_WIDTH` / `MAX_NODE_HEIGHT` (vd: `200,200`) — contour lớn hơn giới hạn × tolerance bị bỏ qua |

### 4. Giao diện chỉnh sửa

```bash
streamlit run ui/app.py
```

Truy cập `http://localhost:8501`.

---

## 📁 Cấu trúc thư mục

```
FlowRevamp/
├── config.py                     # Cấu hình toàn cục (đường dẫn + tham số)
├── main.py                       # CLI entry point
├── orchestrator.py               # Điều phối 4 bước + hợp nhất JSON
├── naive.py                      # Pipeline baseline để so sánh (xem THESIS_NOTES.md)
├── requirements.txt
│
├── data/
│   ├── input/                    # ← Đặt ảnh flowchart gốc ở đây
│   ├── step1_preprocessed/       # Skeleton High-C & Low-C
│   ├── step2_nodes/              # (thư mục làm việc của Bước 2)
│   ├── step3_edges/              # (thư mục làm việc của Bước 3)
│   ├── step4_ocr/                # JSON OCR + ảnh debug (khi bật OCR)
│   └── output/                   # ← JSON cuối + ảnh kết quả
│
├── steps/
│   ├── step1_preprocess.py       # Grayscale → Adaptive Thresh (Dual-C) → Skeleton
│   ├── step2_detect_nodes.py     # ★ Iterative hierarchy contour + shape fitting
│   ├── step3_detect_edges.py     # ★ Pairwise ROI Pathfinding + leftover filter
│   └── step4_ocr.py              # EasyOCR text extraction
│
├── utils/
│   ├── ransac_shapes.py          # ★ Fit Rect/Rhombus/Circle + chấm điểm inlier
│   ├── geometry.py               # Circularity, vertex count, bbox helpers
│   ├── visualization.py          # Vẽ overlay kết quả lên ảnh gốc
│   └── io_helpers.py             # JSON I/O, liệt kê ảnh, tạo thư mục
│
└── ui/
    └── app.py                    # Streamlit Human-in-the-loop UI
```

---

## ⚙️ Cấu hình

Mọi tham số nằm trong [`config.py`](config.py). Các tham số chính đang được pipeline sử dụng:

### Bước 1 – Tiền xử lý

```python
ADAPTIVE_THRESH_BLOCK_SIZE = 15   # Kích thước block (phải lẻ)
ADAPTIVE_THRESH_C_HIGH = 20       # Ngưỡng C cao → node detection
ADAPTIVE_THRESH_C_LOW = 4         # Ngưỡng C thấp → edge detection
MORPH_KERNEL_SIZE = (3, 3)        # Kernel đóng hình thái học
```

### Bước 2 – Node detection

```python
MIN_NODE_AREA = 500               # Diện tích contour nhỏ nhất (px²)
RANSAC_N_ITER = 200               # Số vòng đánh giá mỗi mô hình hình học
RANSAC_INLIER_THRESH = 3.0        # Khoảng cách tối đa để tính inlier (px)
RANSAC_MAX_ITERATIONS = 20        # Số vòng extract-and-repeat tối đa
RANSAC_MASK_MARGIN = 5            # Margin dự phòng khi mask (px, thường bị
                                  # thay bằng margin động theo Distance Transform)
CIRCULARITY_THRESHOLD = 0.85      # Trên ngưỡng này → Connector (hình tròn)
MAX_NODE_WIDTH = 150              # Giới hạn bề ngang node tham chiếu (px)
MAX_NODE_HEIGHT = 150             # Giới hạn chiều cao node tham chiếu (px)
NODE_SIZE_TOLERANCE = 2           # Cho phép node lớn tới MAX × tolerance
```

### Bước 4 – OCR

```python
ENABLE_OCR = False                # Bật/tắt OCR (tắt giúp chạy nhanh khi debug)
OCR_LANGUAGES = ["en", "vi"]      # Ngôn ngữ EasyOCR
OCR_ROI_PADDING_RATIO = 0.08      # Padding 8% quanh mỗi crop node
OCR_GPU = True                    # Dùng GPU cho EasyOCR nếu có
```

### Trực quan hóa

```python
VIS_NODE_COLOR = (0, 255, 0)      # Bounding box node (BGR)
VIS_EDGE_COLOR = (255, 0, 255)    # Đường nối (magenta)
VIS_TEXT_COLOR = (255, 255, 0)    # Nhãn text
```

> **Ghi chú:** một số hằng số trong Bước 3 hiện được hard-code trong `steps/step3_detect_edges.py`: khoảng cách cặp node tối đa **600 px**, padding ROI **5 px**, vùng snap chạm node **15 px**, ngưỡng nét mồ côi **40 px**. Nhóm tham số `HOUGH_*` / `RANSAC_HOUGH_*` / `EDGE_NODE_SNAP_DISTANCE` trong `config.py` là di sản của phiên bản Hough cũ, hiện không còn được pipeline chính sử dụng.

---

## 🖥 Giao diện chỉnh sửa (UI)

```bash
streamlit run ui/app.py
```

Chức năng:
- **Chọn kết quả** đã xử lý từ sidebar (đọc các file `*_graph.json` trong `data/output/`)
- **Xem overlay** node/edge trên ảnh gốc; bật/tắt hiển thị Bad Node (viền đỏ)
- **Sửa Node** – đổi type (Process / Decision / Terminal / IO / Connector / Bad), sửa text OCR; Good Node và Bad Node hiển thị ở 2 nhóm riêng
- **Sửa Edge** – đổi source/target/label, thêm cạnh mới
- **Lưu** – ghi đè JSON và tự sinh lại ảnh trực quan hóa

---

## 📊 Định dạng JSON đầu ra

```json
{
  "source_image": "flowchart.png",
  "nodes": [
    {
      "id": "n1",
      "type": "Decision",
      "text": "Kiểm tra lỗi?",
      "bbox": [110, 240, 180, 120],
      "is_bad": false
    },
    {
      "id": "n2",
      "type": "Process",
      "text": "Sửa chữa",
      "bbox": [110, 400, 180, 60],
      "is_bad": false
    }
  ],
  "edges": [
    {
      "source": "n1",
      "target": "n2",
      "label": ""
    }
  ]
}
```

- `bbox` có dạng `[x, y, width, height]` (pixel, gốc trên-trái).
- `type` ∈ {`Process`, `Decision`, `Terminal`, `Connector`, `Bad`} (UI cho phép thêm `IO`).
- `is_bad = true` đánh dấu artifact (khung khổng lồ chồng lấp hoặc nét vẽ mồ côi) cần người dùng xử lý.
- `text` rỗng khi OCR tắt.
- Node được đánh số lại `n1, n2, …` theo thứ tự trên-xuống, trái-sang-phải.

---

## 🔬 Baseline so sánh (`naive.py`)

`naive.py` là pipeline đối chứng giữ nguyên cấu trúc các bước nhưng loại bỏ mọi cải tiến (Otsu toàn cục thay Adaptive Dual-C, quét contour 1 lượt thay Iterative Masking, Hough toàn cục + nối thẳng tâm thay Pairwise ROI). Dùng để đo lường đóng góp của từng cải tiến — chi tiết trong [THESIS_NOTES.md](THESIS_NOTES.md).

```bash
python naive.py data/input/flowchart.png
```

---

## 🔧 Xử lý sự cố

| Vấn đề | Giải pháp |
|--------|-----------|
| `ModuleNotFoundError: easyocr` | `pip install easyocr` (hoặc giữ `ENABLE_OCR = False`) |
| `ModuleNotFoundError: skimage` | `pip install scikit-image` |
| `FileNotFoundError: Preprocessed image ... not found` | Chạy đầy đủ pipeline qua `main.py` (Bước 2/3 cần đầu ra của Bước 1) |
| Quá ít node được detect | Giảm `MIN_NODE_AREA`; kiểm tra ảnh skeleton trong `data/step1_preprocessed/` |
| Node to bị bỏ qua | Tăng `--max-node-size` hoặc `MAX_NODE_WIDTH`/`MAX_NODE_HEIGHT`/`NODE_SIZE_TOLERANCE` |
| Hình tròn/chữ nhật bị phân loại nhầm | Điều chỉnh `CIRCULARITY_THRESHOLD` |
| Thiếu cạnh giữa 2 node xa nhau | Cạnh dài > 600 px bị bỏ (giới hạn hard-code trong `step3_detect_edges.py`) |
| Cạnh sai hướng | Đầu mũi tên không được nhận ra → sửa tay trong UI |
| OCR nhận sai chữ | Mở UI (`streamlit run ui/app.py`) để chỉnh tay |
