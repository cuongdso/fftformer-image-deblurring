# Dataset và layout dữ liệu

**Trạng thái:** layout bên dưới được audit từ `test.py` và các YAML upstream; nguồn tải, dung lượng và license dataset là **UNVERIFIED** tại M0. Không commit dataset vào Git.

## Layout chính thức của test script

`third_party/FFTformer/test.py` yêu cầu pair ground truth, không phải CLI inference một ảnh:

```text
<data_dir>/
└── test/
    ├── input/
    │   ├── 000.png
    │   └── ...
    └── target/
        ├── 000.png      # cùng filename với input
        └── ...
```

Test dataset chỉ chấp nhận extension `png`, `jpg`, `jpeg` (lowercase) và mở target cho mọi input. Tên thiếu, extension khác hoặc target không match sẽ lỗi. `test.py` không so metric trong loop, nhưng target vẫn là precondition do loader.

## Dataset được tham chiếu trong upstream pin

- `GoPro`: README/test default và `options/train/GoPro.yml`; train config dùng LMDB blur/sharp, validation dùng `sharp_crops.lmdb`/`blur_crops.lmdb`.
- `RealBlur-J`: `options/train/Realblur.yml` dùng paired LMDB blur/sharp; README liên kết pretrained RealBlur và evaluator RealBlur.
- Paper chính thức hiển thị/evaluate GoPro và RealBlur theo bản PDF. Các split, URL nguồn chuẩn, license, dung lượng, và chính xác các benchmark còn **UNVERIFIED** trong M0; WS2 phải audit primary dataset sources trước khi tải.

## Chính sách nhóm đề xuất

Lưu dữ liệu ngoài Git tại location do nhóm chọn; `data/` chỉ chứa README. WS2 phải tạo manifest có source URL, version, license/terms, checksum nếu nguồn có, split và mapping input/target. Crop, padding, normalization và color order chỉ được chốt sau khi C-03 approved.

## Kiểm tra trước khi sử dụng

1. Xác minh quyền download/use/redistribute từ nguồn dataset.
2. Kiểm tra từng input có target tương ứng khi evaluation cần GT.
3. Ghi image dimensions, channel order, dtype/range và preprocess policy vào config/metadata.
4. Chạy validator nhóm sau khi nó được triển khai; M0 chưa có validator dữ liệu.
