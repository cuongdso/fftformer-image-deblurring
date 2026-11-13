# ADR-0002: Chốt M1.0 interface contract baseline v1.0

- **Status:** Accepted by Tech Lead
- **Date:** 2026-11-13
- **Scope:** C-01 đến C-06 trong `docs/contracts.md`; không thay đổi upstream baseline và không xác nhận runtime result.

## Context và authority

M1.0 đã chuẩn bị candidate v0.1 nhưng chờ cross-workstream review. Tech Lead đã chỉ thị dùng official source code làm evidence chính và trực tiếp quyết định các điểm không có trong source; implementation của các thành viên chỉ được làm trong scope Tech Lead cho phép. Vì vậy contract được freeze thành v1.0, dù tên thành viên/GitHub username vẫn `TBD`.

## Evidence upstream tại pin `01306b5d4072146a011c8c6e9d4927cf4607f1d5`

| Evidence | Kết luận dùng cho contract |
| --- | --- |
| `test.py`: PIL + `torchvision.transforms.functional.to_tensor`; `fftformer()` mặc định 3 input/output channels | `to_tensor` đưa ảnh PIL thông thường vào tensor `float32` `[0,1]`; DataLoader tạo batch NCHW và model mặc định ba kênh. Script không gọi `convert('RGB')`, nên RGB là chuẩn hóa có chủ đích của group contract, không phải conversion upstream đã xác minh. |
| `test.py`: `pad(input_img, (0,w_n,0,h_n), mode='reflect')`, crop `:h,:w` | Baseline pad cạnh phải/dưới reflection tới bội 32, rồi khôi phục H×W gốc. |
| `test.py`: `torch.load`, `load_state_dict(..., strict=True)`, `torch.device('cuda')` | Baseline strict-load và CUDA-centric; group adapter phải không che mismatch, CPU support không được suy diễn. |
| `test.py`: `<data_dir>/test/input` và `target` theo filename | Đây là convention adapter official test, không phải layout bắt buộc của mọi dataset nhóm. |
| `options/train/{GoPro,Realblur}.yml` | Validation đặt `crop_border: 0`, `test_y_channel: false` cho PSNR/SSIM. |
| `basicsr/models/image_restoration_model.py` + `basicsr/utils/img_util.py` | Nhánh validation `use_image=True` chuyển tensor RGB qua `tensor2img`; `rgb2bgr=True` tạo `ndarray` HWC BGR `uint8` trước khi gọi metric. |
| `basicsr/metrics/psnr_ssim.py` | Metric khai báo `input_order` CHW/HWC; SSIM mặc định đi qua CUDA (`ssim3d=True`). Tensor không được coi là input interchange nguyên trạng ngoài adapter đã xác định. |

## Decisions

1. C-01 strict loading là default; `strict=False` chỉ được dùng bởi task Tech Lead nêu rõ lý do. Group adapter resolve `map_location` theo device request để portability, tách biệt behavior `test.py`.
2. C-02/C-03/C-04 chuẩn hóa RGB `float32` `[0,1]`, sample `CHW [3,H,W]`, model batch `NCHW [N,3,H,W]`; conversion và padding metadata luôn explicit. Không tự convert grayscale/alpha hay BGR ngầm.
3. Dataset nhóm dùng manifest/sample ID deterministic và có thể không có GT. Official `test/input` + `test/target` chỉ là one adapter profile.
4. C-04 trả một result mỗi sample; batch orchestration trả sequence theo thứ tự để không che sample metadata/padding.
5. C-05 tách hai evaluator profile: `upstream-basicsr` cho config/code audit và `realblur-official` cho claim RealBlur. Với nhánh `use_image=True` của upstream, WS4 chuyển RGB CHW/NCHW qua `tensor2img`, ghi `rgb2bgr` và dùng cùng HWC representation cho prediction/GT. Chúng không thay thế nhau; không khẳng định reproduction khi chưa chạy protocol tương ứng.
6. C-06 dùng config versioned và runtime path resolution; không commit path máy cá nhân/secret. Đây là quyết định nhóm, không phải claim upstream hỗ trợ cross-platform.

## Consequences

- Các contract C-01..C-06 là `APPROVED` specification v1.0; không có test/model/dataset availability nào được ngụ ý.
- Wave C chỉ bắt đầu theo task scope Tech Lead giao, với test/evidence riêng. Thành viên phải dừng và báo nếu source/evidence mới mâu thuẫn v1.0.
- Mọi thay đổi cross-module phải do Tech Lead cho phép, cập nhật ADR và contract trước hoặc cùng PR implementation.
