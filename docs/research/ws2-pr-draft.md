# WS2 — hồ sơ review trước khi push

Owner: `phanthek2`. GitHub: `phanthek2-arch`. Branch: `feat/data-image-validator`.
Base: `f06de9432052af873aeb8b5ff8c85217d24ed222`.
Người dùng đã cho phép commit/push. Code WS2-02/03 được tách thành hai commit
`a24c62b` và `0c108a6`; audit/hồ sơ là commit riêng. Chưa mở PR hoặc merge;
không có xác nhận review từ thành viên khác.

## Đọc trước khi xem code

1. [Handoff](ws2-handoff.md): kết quả, lệnh tái lập và giới hạn.
2. [Data guide](../modules/data.md): API, policy và ví dụ sử dụng.
3. [Dataset audit](ws2-dataset-audit.md): nguồn/terms, native layout và ảnh đã kiểm.
4. [Evidence JSON](ws2-runtime-evidence.json): logs, exit codes và hashes.

## Nội dung đề xuất cho PR implementation

**Tiêu đề:** `feat(data): add validated image I/O and deterministic dataset samples`

Ảnh đọc từ file hiện được kiểm RGB/8-bit và chuẩn hóa thành float32 CHW `[0,1]`
trước khi cung cấp cho WS3/WS4. Paired manifests kiểm mapping 1:1, thứ tự duyệt
xác định và lỗi cặp/size/channel; non-GT trả target None, không suy diễn GT.

- Issue/task IDs: `M1-WS2-02`, `M1-WS2-03`; audit evidence: `M1-WS2-01`.
- Workstream/owner: WS2, `phanthek2`.
- Files/modules: `src/fftformer_lab/data/`, hai `tests/unit/test_data_*.py`,
  `tests/fixtures/ws2/`, optional `data` extra, verifier và docs WS2.
- API/contract: public API WS2 mới theo C-02/C-03; shared contracts không đổi.
- Upstream compatibility: không sửa source/pin, checkpoint hoặc WS1/WS3/WS4.

**Xác minh đã chạy:** 40/40 WS2 tests; 137/137 repository tests khi bật checkpoint;
standard-library suite 50 qua/87 skip; compile, pip check, whitespace qua. Hai cặp
ảnh tác giả 1280x720 qua paired/non-GT loader; synthetic CPU smoke dùng strict
GoPro checkpoint thật qua WS1/WS3. Logs/source hashes ở evidence JSON.

**Giới hạn:** chỉ validate hai ảnh ví dụ ngoài và fixture nhóm, chưa full dataset
benchmark/RealBlur/CUDA/LMDB/training. Reader chủ đích từ chối non-RGB/16-bit/animation;
không đổi crop/resize/padding policy, không tự thêm augmentation hoặc collation.

**Reviewer đề xuất:** WS1 (`DucHai27`) và WS3 (`Alex`) cho image/sample boundary;
WS4/Tech Lead (`cuongdso`) cho GT/provenance/scope. Đây là đề xuất, không gửi mention
hay thông báo tới các tài khoản đó.

## Checklist trước khi xuất bản

- [x] Chức năng khớp C-02/C-03; thử cả paired và non-GT.
- [x] Test errors, source immutability, ordering và consumer boundary có evidence.
- [x] Có nguồn ảnh/attribution/checksum; bốn PNG ngoài nằm dưới ignored `data/`.
- [x] Candidate binary của task chỉ có ba fixture tự tạo tổng 224 bytes.
- [x] Không thay shared contracts hoặc source/module WS1/WS3/WS4.
- [x] Người dùng đã cho phép commit/push feature branch; không merge hoặc tự nhận team review.
- [x] Tách code WS2-02 và WS2-03 thành các commit riêng, có thể review theo task.
- [ ] Mở PR theo một task mỗi PR như CONTRIBUTING/INSTRUCTION yêu cầu.
- [ ] Reviewer chấp nhận trước khi người có thẩm quyền merge.

Branch chứa trọn bản tích hợp để review. Commit WS2-02 chỉ export image/errors;
commit WS2-03 bổ sung dataset exports, loader và tests. Có thể chọn các commit
theo task để chuẩn bị PR mà không tạo image-only snapshot thiếu `dataset.py`.
Audit/verifier/docs đi cùng task phù hợp; JSON evidence mô tả bản tích hợp cuối,
cần tái tạo cho đúng snapshot khi xuất bản từng PR.

## Xem và chạy lại trên máy hiện tại

```powershell
git status --short --branch
git log --oneline origin/main..HEAD
git diff origin/main...HEAD --stat
git diff origin/main...HEAD -- src/fftformer_lab/data tests/unit/test_data_image.py tests/unit/test_data_dataset.py
.\.venv\Scripts\python.exe -B scripts/ws2_verify.py --repo . --examples-dir data/ws2-gopro-examples --checkpoint third_party/FFTformer/pretrain_model/fftformer_GoPro.pth --output docs/research/ws2-runtime-evidence.json
```

Chỉ push feature branch theo yêu cầu người dùng. Không dùng force-push hoặc tự
merge main. GitHub PR/review vẫn là bước riêng; kiểm chứng local không thay thế
approval của nhóm.
