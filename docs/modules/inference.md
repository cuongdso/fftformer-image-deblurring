# WS3 — Inference & Demonstration

**Hiện trạng:** WS3 đã hoàn thành cả 2 task Wave C: `M1-WS3-02` (Inference pipeline core) và `M1-WS3-03` (CLI & Orchestration) đạt trạng thái `DONE`. Hai task đã merge qua [PR #9](https://github.com/cuongdso/fftformer-image-deblurring/pull/9), commit `c47d58d`. Evidence tích hợp và kết quả kiểm thử tại [project status](../project-status.md), phân công tại [ownership](../ownership.md).

Dependency và acceptance criteria chuẩn ở [backlog](../backlog.md). `M1-WS3-02` phụ thuộc vào các contract C-01, C-02, C-04 đã `APPROVED`; `M1-WS3-03` tiêu thụ trực tiếp public adapters từ WS1 (`ModelAdapter`, `load_model`) và WS2 (`read_image`, `ImageDataset`, `load_manifest`).

[Hồ sơ bàn giao WS3](../research/ws3-handoff.md) tổng hợp toàn diện các task, files bàn giao, kết quả kiểm chứng 57/57 checks và ranh giới với WS4.

## Phạm vi API WS3

Mã nguồn tại `src/fftformer_lab/inference/` cung cấp pipeline suy luận khử mờ, tính toán padding reflection tới bội 32, serialization ảnh đầu ra và công cụ dòng lệnh (CLI) theo [Contract C-02/C-04](../contracts.md).

| API | Chức năng |
| --- | --- |
| `InferencePipeline` | Bộ điều phối suy luận chính; nhận model, xác thực device, hỗ trợ `run_single` và `run_batch` tuần tự, đo thời gian thực thi chính xác. |
| `InferenceRequest` | Dataclass đóng gói tensor ảnh đầu vào `image` (RGB float32 CHW) và `sample_id`. |
| `InferenceResult` | Dataclass immutable chứa ảnh đã khôi phục `restored_image`, `sample_id`, `padding`, các shape trước/sau pad, `device`, `elapsed_time_seconds`, `is_mock` và `to_dict()`. |
| `PaddingMetadata` | Lưu trữ siêu dữ liệu padding: `top`, `bottom`, `left`, `right`, `mode="reflect"`, `original_shape` và `padded_shape`. |
| `calculate_reflection_padding` | Tính toán lượng pad đáy và phải để $H, W$ chia hết cho 32; phát hiện sớm và chặn lỗi nếu kích thước ảnh quá nhỏ cho reflection pad ($H \le \text{pad\_h}$ hoặc $W \le \text{pad\_w}$). |
| `pad_image` | Áp dụng reflection padding lên tensor 3D CHW hoặc 4D NCHW. |
| `unpad_image` | Cắt (crop) tensor về lại spatial extent ban đầu $H \times W$. |
| `write_image` | Lưu tensor RGB float32 CHW `[3, H, W]` thành ảnh 8-bit PNG hoặc JPEG theo C-02 (clamp $[0.0, 1.0]$, lượng tử hóa uint8). |
| `run_cli` | Điều phối CLI toàn diện cho ảnh đơn, thư mục ảnh và dataset manifest; xử lý lỗi và exit code chuẩn. |
| `validate_input_tensor` | Kiểm tra tensor đầu vào tuân thủ nghiêm ngặt C-02: 3D, RGB 3 kênh, float32, giá trị hữu hạn không NaN/Inf, range $[0.0, 1.0]$. |
| `MockModel` | Test double độc lập hỗ trợ các mode `identity`, `invert`, `shift`; gán nhãn `is_mock = True` và kiểm tra đầu vào NCHW chia hết cho 32. |

## Cơ chế Padding & Cropping (C-02/C-04)

Theo baseline official và Contract C-02, mô hình FFTformer yêu cầu spatial dimensions $H$ và $W$ phải là bội số của 32 (do cấu trúc chia patch 8x8 ở cấp sâu nhất $H/4$).
Pipeline thực hiện padding ở góc phải và cạnh đáy bằng phép phản xạ (*reflection padding*):

$$\Delta H = (32 - H \pmod{32}) \pmod{32}, \quad \Delta W = (32 - W \pmod{32}) \pmod{32}$$

- Bộ pad tương ứng: `(left=0, right=ΔW, top=0, bottom=ΔH)`.
- Khi kích thước đã chia hết cho 32, padding bằng 0 và ảnh được giữ nguyên.
- Sau khi model forward, tensor output có kích thước $[1, 3, H + \Delta H, W + \Delta W]$ được squeeze bỏ chiều batch, sau đó crop lại `[:H, :W]` và clamp trong khoảng $[0.0, 1.0]$.

## Sử dụng CLI Orchestrator (`M1-WS3-03`)

Từ repository root:

```powershell
$env:PYTHONPATH = (Resolve-Path src).Path
```

### 1. Kiểm tra luồng xử lý một ảnh bằng mock
```powershell
python -m fftformer_lab.inference --input anh_mo.png --output anh_ro.png --mock
```
*Tùy chọn: Thêm cờ `--save-metadata` để lưu file `anh_ro_metadata.json` chứa thông tin chi tiết về kích thước, padding và thời gian chạy.*

### 2. Kiểm tra luồng xử lý một thư mục bằng mock
```powershell
python -m fftformer_lab.inference --input-dir data/test_blur/ --output-dir results/deblurred/ --mock
```

### 3. Kiểm tra luồng xử lý theo manifest WS2 bằng mock
```powershell
python -m fftformer_lab.inference --manifest data/gopro_test_manifest.json --output-dir results/gopro/ --mock
```

### 4. Khử mờ bằng checkpoint FFTformer

Cần submodule đã khởi tạo, PyTorch, NumPy, Pillow, einops và checkpoint tương thích. Các lượt chạy mock ở trên mặc định dùng phép biến đổi `identity`.
```powershell
python -m fftformer_lab.inference `
    --input anh_mo.png `
    --output results/anh_ro.png `
    --checkpoint checkpoints/fftformer_GoPro.pth `
    --architecture-file third_party/FFTformer/basicsr/models/archs/fftformer_arch.py `
    --device cpu
```

## Kiểm thử WS3 và bằng chứng runtime evidence

Script kiểm chứng `scripts/ws3_verify.py` tự động chạy toàn bộ 57 tests của WS3 và cập nhật [JSON evidence](../research/ws3-02-runtime-evidence.json):

```powershell
python -B scripts/ws3_verify.py --repo . --output docs/research/ws3-02-runtime-evidence.json
# Summary: 57/57 passed
```

Chi tiết 57 tests gồm 4 nhóm:
- [10 unit tests thuần](../../tests/unit/test_inference_pure.py): Cấu trúc dữ liệu `PaddingMetadata`, `InferenceResult.to_dict()`, công thức padding với các kích thước khác nhau, kiểm tra điều kiện biên kích thước quá nhỏ, phân cấp ngoại lệ, từ chối mock mode lạ. Hoàn toàn không cần PyTorch.
- [10 pipeline integration tests](../../tests/unit/test_inference_pipeline.py): Xác thực tensor đầu vào, round-trip pad/unpad bảo toàn nội dung, các mode mock model (identity, invert, shift), từ chối kích thước không chia hết 32, chạy single với đo thời gian, hỗ trợ `InferenceRequest` và duck-typed `Sample`, chạy batch tuần tự bảo toàn thứ tự, và truyền lỗi model.
- [22 contract tests](../../tests/unit/test_inference_contracts.py): Kiểm tra chặt chẽ các ràng buộc C-01/C-02/C-04 (từ chối model ở `training` mode, bảo vệ device cố định của model, từ chối device sai, xử lý sample ID fallback, cô lập metadata, kiểm tra output dtype/shape/non-finite, batch rỗng, batch báo lỗi kèm index).
- [15 CLI & IO tests](../../tests/unit/test_inference_cli.py): Kiểm tra lưu ảnh PNG/JPEG (`write_image`), lượng tử hóa và clamp, từ chối định dạng lạ, CLI `--help`, bắt lỗi thiếu đối số, chạy ảnh đơn lẻ, chạy thư mục batch, chạy qua manifest của WS2, xuất metadata và xử lý lỗi input không tồn tại.

Toàn bộ 57 tests chạy độc lập trên CPU trong ~0.25s.

## Ranh giới bàn giao cho WS4

WS3 cung cấp `InferencePipeline`, `InferenceResult` và `write_image` để WS4 sử dụng trực tiếp trong module đánh giá chất lượng (C-05). Kết quả suy luận cho từng sample luôn chứa tensor ảnh đã khôi phục đúng kích thước gốc, nhãn `sample_id` tương ứng và metadata thời gian chạy.
