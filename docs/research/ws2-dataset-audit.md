# WS2-01 — nguồn dữ liệu, layout và preprocessing

Ngày audit: 18/11/2026. Owner: `phanthek2`. Trạng thái: báo cáo và kiểm chứng
local sẵn sàng review; chưa có acceptance/PR/merge của nhóm. Không sửa C-02/C-03.

## Nguồn chính

| Nguồn | Nội dung đã đối chiếu |
| --- | --- |
| [Trang GoPro của Seungjun Nah](https://seungjunnah.github.io/Datasets/gopro.html) | Nguồn phát hành, cặp mờ/rõ, biến thể gamma/linear, link Drive/Hugging Face và công bố CC BY 4.0. |
| [Repo tác giả, revision `2d5a698`](https://github.com/SeungjunNah/DeepDeblur_release/blob/2d5a698560e658718f0520e48dfb15bd52c80118/README.md) | README minh họa hai cặp Flower/Istanbul; thống kê công bố 2103 cặp train và 1111 cặp test. Chưa đếm lại toàn bộ archive. |
| [GoPro PyTorch reader](https://github.com/SeungjunNah/DeepDeblur-PyTorch/blob/e92dc7c3337127887da617ffe819b49cb022fd81/src/data/gopro_large.py), [base reader](https://github.com/SeungjunNah/DeepDeblur-PyTorch/blob/e92dc7c3337127887da617ffe819b49cb022fd81/src/data/dataset.py) | Reader của tác giả dùng `blur_gamma` và `sharp`; đây là evidence layout, không là quyết định nhóm chọn CRF cho thí nghiệm FFTformer. |
| [RealBlur README, revision `4d65bb3`](https://github.com/rimchang/RealBlur/blob/4d65bb32daa6dd89fd629311a43e69c945e631e9/README.md) | Nguồn tải chính thức, công bố CC BY 4.0 và evaluator của tác giả. Chỉ audit tài liệu; chưa tải/chạy dataset này. |
| [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | Khi dùng/chia sẻ cần ghi công, liên kết license và nêu thay đổi. Attribution cho mẫu tải local nằm cùng dữ liệu. |

Không suy diễn license dataset từ license code hoặc checkpoint. Ảnh ngoài không
được đưa vào Git của nhóm. Citation: Nah, Kim, Lee, *Deep Multi-Scale
Convolutional Neural Network for Dynamic Scene Deblurring*, CVPR 2017; Rim,
Lee, Won, Cho, *Real-World Blur Dataset for Learning and Benchmarking Deblurring
Algorithms*, ECCV 2020.

## Mapping layout sang C-03

| Layout | Cách dùng loader WS2 |
| --- | --- |
| FFTformer `test/input/<name>` + `test/target/<name>` | Truyền hai root cho `build_paired_manifest`; `sample_id` là filename đầy đủ. |
| Một sequence GoPro `<split>/<sequence>/blur_gamma/<frame>` + `sharp/<frame>` | Truyền hai root cho paired builder. Khi gộp nhiều sequence, tạo `SampleRecord` với ID `<sequence>/<frame>` để tránh trùng tên frame. |
| Hai root có cây sequence tương ứng | `build_paired_manifest(..., recursive=True)` giữ relative path làm ID. |
| Ảnh mờ không GT | `build_single_manifest`; không tự tìm hoặc tạo target. |

Ví dụ mapping nhiều sequence, không đổi layout/file nguồn:

```python
from fftformer_lab.data import ImageDataset, SampleRecord

# split_root và blur_subdir do caller/config đã chọn cung cấp.
records = [
    SampleRecord(f"{sequence.name}/{frame.name}", frame,
                 sequence / "sharp" / frame.name)
    for sequence in sorted(split_root.iterdir()) if sequence.is_dir()
    for frame in sorted((sequence / blur_subdir).glob("*.png"))
]
dataset = ImageDataset(records)
```

`blur` và `blur_gamma` không được trộn hoặc tự đổi. Việc chọn split/CRF, tạo LMDB
và đối chiếu protocol FFTformer đầy đủ phải ghi trong cấu hình thí nghiệm do nhóm
chốt; kiểm chứng local này không xác nhận chúng.

## Policy đã triển khai và kiểm chứng

- PNG/JPEG RGB 8-bit một frame → CPU float32 CHW, chia 255, giữ H/W gốc.
- Array phải khai báo RGB/BGR; normalized float32 không bị chia 255 lần nữa.
- Grayscale/alpha/16-bit, sai range/channel/shape và ảnh hỏng báo lỗi có ngữ cảnh.
- Không chỉnh orientation, resize/crop, gamma/saturation hoặc augmentation ngầm.
- Reflection padding/crop và batch dimension tiếp tục thuộc WS3. Metric/protocol
  tiếp tục thuộc WS4. Không thay đổi shared contracts.

Fixtures cực nhỏ do nhóm tự tạo tại `tests/fixtures/ws2/` là nguồn hợp pháp cho
unit tests. Test plan đã được thực thi trong `test_data_image.py` và
`test_data_dataset.py`: màu/range/shape, non-destructive decode, pairing, duplicate,
missing/orphan, mismatch, JSON portability và non-GT; xem [data guide](../modules/data.md).

## Kiểm chứng bằng ảnh tác giả công bố

Tải bốn PNG, tổng **5,016,042 bytes**, từ đường dẫn `images/Flower_*1.png` và
`images/Istanbul_*1.png` ở revision repo tác giả nói trên. README gắn nhãn rõ
ảnh mờ/rõ. Không dùng kết quả model làm target. File chỉ được đổi tên để hai root
`input`/`target` có tên trùng nhau; bytes ảnh giữ nguyên.

Verifier kiểm đúng byte count và Git blob ID trước khi ghi, rồi ghi SHA-256 từng
file vào [runtime evidence](ws2-runtime-evidence.json). Đây là checksum kiểm chứng
nguồn file riêng lẻ, không phải checksum toàn bộ archive GOPRO_Large.

- Hai pair `Flower.png`, `Istanbul.png`: RGB `[3,720,1280]`, target cùng shape.
- Paired và non-GT views có thứ tự ID ổn định; non-GT trả `target=None`.
- Hash nguồn trước/sau decode giữ nguyên. Attribution được lưu tại ignored
  `data/ws2-gopro-examples/ATTRIBUTION.md`.
- Split và CRF của các hình minh họa **không được suy đoán**. Hai cặp này không
  đại diện cho validation đầy đủ 1111 test pairs hoặc paper reproduction.

Để tái lập, xem lệnh verifier trong [handoff](ws2-handoff.md). Mặc định verifier
không tải mạng; `--download-examples` cho phép tải đúng bốn PNG đã pin.
