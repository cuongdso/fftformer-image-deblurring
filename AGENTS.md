# Hướng dẫn cho contributors và AI agents

## Mục tiêu

Repository này hỗ trợ nghiên cứu, tái hiện có kiểm chứng và phát triển công cụ xung quanh FFTformer cho image deblurring. Đây không phải official implementation. Trạng thái chuẩn của dự án nằm tại `docs/project-status.md`.

## Đọc trước khi thay đổi

1. `INSTRUCTION.md`, `README.md` và `docs/architecture.md`.
2. `docs/project-status.md`, `docs/ownership.md`, `docs/backlog.md` và `CONTRIBUTING.md`.
3. Tài liệu module trong `docs/modules/` và contract tại `docs/contracts.md` tương ứng với task.
4. Khi làm baseline/reproduction: `docs/upstream-audit.md` và `docs/reproduction/`.

## Repository map và nguồn sự thật

- `third_party/FFTformer/`: official baseline, Git submodule pin; paper và upstream audit là nguồn sự thật cho hành vi upstream.
- `src/fftformer_lab/`: code do nhóm sở hữu; docs contracts là nguồn sự thật cho interface trước khi code tồn tại.
- `configs/`: cấu hình do nhóm sở hữu; chưa có config thực thi ở M0.
- `tests/`: test cho code nhóm. Không thêm test giả định baseline đã chạy.
- `experiments/`: metadata, không commit dataset, checkpoint hoặc output lớn.

Không ghi mâu thuẫn với trạng thái chuẩn. Thông tin chưa kiểm chứng phải mang nhãn `UNVERIFIED`; ý tưởng chưa làm phải mang `PLANNED` hoặc `PROPOSED`.

## Quy ước kỹ thuật

- Python: UTF-8, 4 spaces, type hints cho public API mới, docstring ngắn cho public entry point.
- Tên package: `fftformer_lab`; không import/mutate baseline như một dependency ngầm.
- RGB/BGR, shape, range và padding phải tuân thủ `docs/contracts.md`; không tự đoán.
- Dữ liệu, checkpoint, secrets, log/output lớn không được commit.
- Không thay đổi API chung, kiến trúc baseline hay submodule pointer mà không cập nhật docs/ADR và có review.

## Ownership và Git workflow

Ownership theo workstream nằm ở `docs/ownership.md`; cách nhận task và handoff chuẩn nằm ở `INSTRUCTION.md`. Tech Lead là authority cuối cùng cho contract chung, dependency liên module và scope task; agent không đợi consensus để phủ định một ADR/contract đã Tech Lead chốt, nhưng phải báo bằng evidence nếu source mới mâu thuẫn. Làm việc trên feature branch ngắn hạn từ `main`: `feat/model-*`, `feat/data-*`, `feat/inference-*`, `feat/evaluation-*`, `fix/*`, `docs/*`, `chore/*`, `experiment/*`. Không force-push `main`, không rewrite/xóa branch của người khác.

## Kiểm tra

M0 có các lệnh không cần GPU/dependency ngoài standard library:

```bash
python scripts/check_env.py
python -m unittest discover -s tests -p "test_*.py"
python -m compileall -q scripts src
git diff --check
```

Chỉ báo cáo lệnh đã thực sự chạy và kết quả thực tế. Test model/checkpoint/GPU phải có marker riêng khi được bổ sung.

## Hoàn thành task

Đọc code, task trong backlog và tài liệu liên quan trước khi sửa; giữ phạm vi nhỏ. Cập nhật documentation/contract bị ảnh hưởng, chạy test phù hợp, rồi mở PR bằng template. Handoff phải nêu files, interface thay đổi, commands/results, hạn chế, commit/PR và việc còn lại. Chi tiết tại `docs/onboarding.md`.

## Cấm

Không training, tải dataset/weights lớn, tuyên bố benchmark PASS, hoặc triển khai kiến trúc mới ngoài task được giao. Không copy upstream vào `src/`, không sửa `third_party/FFTformer/` tùy tiện, không commit credential, và không biến kế hoạch thành trạng thái đã hoàn thành.
