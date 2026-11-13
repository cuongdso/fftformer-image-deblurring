# Pretrained models

**Trạng thái:** static audit; không có checkpoint nào được nhóm tải hoặc chạy trong M0.

## Artifact được quan sát trong upstream checkout

`third_party/FFTformer/pretrain_model/fftformer_GoPro.pth` tồn tại trong submodule pin và có kích thước 66,441,171 bytes trong checkout audit. Artifact này là file upstream-tracked; superproject chỉ commit submodule SHA, không sao chép weight vào lịch sử của nhóm. WS1 phải xác minh kiến trúc/config/state-dict trước khi gọi nó compatible.

README upstream còn liên kết một Google Drive folder cho **RealBlur pre-trained model**, nhưng static audit không xác minh filename, checksum, dataset huấn luyện, license hoặc điều khoản phân phối của artifact trong link đó. Không tải hoặc phân phối artifact này trước khi kiểm tra nguồn.

## Cách test script upstream load checkpoint

`test.py` gọi `torch.load(args.test_model)` rồi `model.load_state_dict(state_dict, strict=True)` trên `fftformer()` default. Default argument trong script là `./pretrain_model/net_g_GoPro_HIDE.pth`, nhưng file đó không được quan sát trong upstream checkout audit. Đây là mismatch cần được làm rõ, không được tự thay tên rồi tuyên bố tái hiện thành công.

## Quy trình quản lý checkpoint của nhóm (PLANNED)

1. Đặt weight ngoài Git, ví dụ location local được cấu hình; không dùng path cá nhân trong config commit.
2. Ghi origin URL, version, architecture/config, SHA-256 nếu có/thực hiện được, access date và terms trong experiment metadata.
3. Chạy compatibility check strict; lưu exact error nếu mismatch.
4. Chỉ sau đó mới ghi mapping checkpoint↔architecture trong tài liệu này và C-01.

Không suy ra hai checkpoint tương đương chỉ từ tên file/dataset.
