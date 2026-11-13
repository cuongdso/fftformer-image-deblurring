# Hướng dẫn nhận việc M1 cho thành viên và coding agent

## Project overview

Đây là repository học thuật cho bài tập lớn Xử lý ảnh: nghiên cứu, tái hiện có kiểm chứng và xây dựng pipeline nhóm xoay quanh **FFTformer** — *Efficient Frequency Domain-Based Transformers for High-Quality Image Deblurring* (CVPR 2023). Paper là nguồn chuẩn cho phương pháp; [official source `kkkls/FFTformer`](https://github.com/kkkls/FFTformer) là nguồn đối chiếu implementation.

`third_party/FFTformer/` là official baseline được pin bằng submodule và chỉ dùng để đối chiếu/reproduce. `src/fftformer_lab/` là group implementation; không phải bản sao hay bản sửa trực tiếp của baseline.

M1.0 đã chốt six interface contracts C-01 đến C-06 ở baseline v1.0, giải vòng phụ thuộc giữa workstream và ghi quyết định tại ADR-0002. Điều đó **không** có nghĩa model adapter, data loader, inference, metric, checkpoint compatibility, CPU/GPU support, benchmark hay reproduction đã được triển khai/chạy. Trạng thái chính thức luôn ở [docs/project-status.md](docs/project-status.md).

## Repository orientation

Trước khi nhận hoặc giao code task, đọc theo thứ tự sau:

1. [AGENTS.md](AGENTS.md) — quy tắc thường trực của repository.
2. File này — cách chọn workstream, task và vận hành agent.
3. [README.md](README.md) — mục tiêu học thuật và baseline boundary.
4. [Project status](docs/project-status.md) — những gì thật sự đã/chưa làm.
5. [Architecture](docs/architecture.md) — ranh giới official baseline và code nhóm.
6. [Contracts](docs/contracts.md) — **nguồn chuẩn duy nhất** cho interface liên module.
7. [Ownership](docs/ownership.md) — trạng thái nhận workstream và file boundary.
8. [Backlog](docs/backlog.md) — task ID, dependency và acceptance criteria.
9. `docs/modules/<workstream>.md` tương ứng: [model](docs/modules/model.md), [data](docs/modules/data.md), [inference](docs/modules/inference.md), hoặc [evaluation](docs/modules/evaluation.md).
10. Khi task liên quan baseline/reproduction: [upstream audit](docs/upstream-audit.md) và `docs/reproduction/` liên quan.

| Khu vực | Mục đích |
| --- | --- |
| `third_party/FFTformer/` | Baseline upstream đã pin; không sửa trong task nhóm. |
| `src/fftformer_lab/` | Mã Python do nhóm sở hữu, tách theo `model`, `data`, `inference`, `evaluation`, `app`, `common`. |
| `tests/` | Unit/integration fixture và test của **nhóm**. |
| `configs/`, `experiments/` | Config versioned và metadata có khả năng truy vết; không chứa secret/artifact lớn. |
| `data/`, `checkpoints/`, `results/` | Chỉ hướng dẫn/đường dẫn local; dataset, weights và output lớn bị ignore. |
| `docs/` | Kiến trúc, contract, ownership, backlog, audit và reproduction evidence. |

Không sao chép C-01..C-06 vào task description hay source code như một nguồn chuẩn thứ hai. Link đến [contracts](docs/contracts.md), nêu contract ID và chỉ đề xuất thay đổi interface qua Tech Lead.

## Chọn workstream và nhận task

| Workstream | Trạng thái nhận | Entry tasks Wave C | Scope chính |
| --- | --- | --- | --- |
| WS1 Model & Architecture | `OPEN` | `M1-WS1-02`, `M1-WS1-03` | `src/fftformer_lab/model/` |
| WS2 Dataset & Preprocessing | `OPEN` | `M1-WS2-02`, `M1-WS2-03` | `src/fftformer_lab/data/`, test fixtures liên quan |
| WS3 Inference & Demo | `OPEN` | `M1-WS3-02`, `M1-WS3-03` | `src/fftformer_lab/inference/`, `src/fftformer_lab/app/` |
| WS4 Evaluation & Integration | `RESERVED` — ưu tiên Tech Lead `cuongdso` | `M1-WS4-02` đến `M1-WS4-04` | `src/fftformer_lab/evaluation/`, `configs/`, `experiments/`, integration tests |

`OPEN` không có nghĩa hai người được làm cùng một workstream. Thành viên chưa có phân công chọn **một** workstream OPEN, đọc issue/backlog, rồi comment nhận task trên GitHub Issue hoặc báo Tech Lead. Chỉ sau xác nhận mới cập nhật Owner trong [ownership](docs/ownership.md) và/hoặc issue. WS4 là reservation ưu tiên của Tech Lead, không là tuyên bố rằng thành viên khác đã đồng ý hay đã được gán việc.

Trước khi code, người nhận phải xác định: task ID; contracts; dependency đã hoàn thành; file/module được phép sửa; reviewer role; acceptance criteria; out-of-scope. Nếu dependency chưa sẵn sàng, dùng fixture/mock chỉ khi contract cho phép và ghi rõ đó không phải FFTformer inference thật; nếu không thể làm an toàn, ghi blocker trong issue/PR cho Tech Lead.

Handoff tối thiểu gồm: task ID, files/module đã đổi, interface thay đổi (`Không` nếu không có), commands và kết quả test thực tế, limitation/`UNVERIFIED`, commit SHA, PR URL và dependency/next step. Không nghiệm thu chỉ bằng mô tả hoặc README.

## Workstream instructions and Definition of Done

### WS1 — Model & Architecture

- **Scope:** `src/fftformer_lab/model/`, unit tests của module, `docs/modules/model.md` khi boundary/evidence thay đổi.
- **Tasks:** `M1-WS1-02` tạo adapter/model specification và validation metadata; `M1-WS1-03` thêm strict checkpoint loader cùng test mismatch. Xem dependency/acceptance criteria chính xác trong [backlog](docs/backlog.md).
- **Tuân thủ:** C-01 và C-02. Đọc `fftformer_arch.py`, config upstream và audit trước khi mapping constructor/checkpoint.
- **Deliverables/DoD:** public API có type hints/docstring ngắn; `ModelSpec`/metadata xác định implementation và checkpoint; lỗi file/device/schema/key/shape có ngữ cảnh; unit tests cho constructor/config/strict mismatch; handoff nêu checkpoint artifact nào đã hoặc chưa dùng.
- **Integration boundary:** cung cấp model evaluation mode + metadata cho WS3/WS4; không ép consumer phải import/mutate `third_party` ngầm.
- **Reviewer:** WS3 cho consumer impact, Tech Lead cho scope/interface.
- **Cấm:** không sửa `third_party/FFTformer/`, không đổi architecture default, không dùng `strict=False` để che mismatch, không claim checkpoint compatible nếu chưa có evidence.

### WS2 — Dataset & Preprocessing

- **Scope:** `src/fftformer_lab/data/`, fixture nhỏ hợp pháp và test liên quan; cập nhật `docs/modules/data.md` khi cần.
- **Tasks:** `M1-WS2-02` image validation/normalization; `M1-WS2-03` deterministic paired manifest/loader và non-GT samples.
- **Tuân thủ:** C-02 và C-03: RGB `float32` CHW `[3,H,W]`, `[0,1]`, `target | None`, sample ID deterministic; conversion BGR→RGB phải explicit.
- **Deliverables/DoD:** đọc ảnh không phá file nguồn; pair/missing/duplicate/size/channel/corrupt-image errors có message; fixture cực nhỏ trong Git; unit tests range/shape/pair/non-GT/failure cases; không resize/crop ngầm ngoài policy config versioned.
- **Integration boundary:** cung cấp `Sample` theo contract cho WS3/WS4, không gắn trực tiếp vào filesystem layout của official test.
- **Reviewer:** WS1/WS3 cho image boundary, WS4 cho supervision/evaluation impact; Tech Lead xác nhận scope.
- **Cấm:** không commit dataset lớn, không tự coi grayscale/alpha/BGR là compatible, không tự thay padding policy.

### WS3 — Inference & Demo

- **Scope:** `src/fftformer_lab/inference/`, `src/fftformer_lab/app/`, test và docs module liên quan. Demo UI chỉ là `PLANNED`; không tạo ngoài task scope.
- **Tasks:** `M1-WS3-02` inference request/result, pad/crop và mock pipeline tests; `M1-WS3-03` CLI/orchestration sau khi interface WS1/WS2 cần thiết ổn định.
- **Tuân thủ:** C-01, C-02, C-04. Input/output phải giữ RGB/range/shape; padding reflection/crop phải có metadata.
- **Deliverables/DoD:** một result mỗi sample, ordered batch orchestration, model/device/input/output errors có context, CLI help/error tests, pad/crop/metadata tests. Mock model được phép để test pipeline nhưng phải được đặt tên/ghi nhận là mock, không là FFTformer benchmark.
- **Integration boundary:** nhận model adapter WS1 và `Sample` WS2; trả `InferenceResult` cho WS4. Không đổi shared fields mà không có approval.
- **Reviewer:** WS1 và WS2 cho provider boundary, WS4 cho result/evaluation impact; Tech Lead cho scope.
- **Cấm:** không tuyên bố CPU/GPU inference thật đã hỗ trợ nếu chưa chạy; không sửa model để né memory; không triển khai demo độc lập vượt task.

### WS4 — Evaluation & Integration

- **Scope:** `src/fftformer_lab/evaluation/`, `configs/`, `experiments/`, integration tests và light CI khi được scope.
- **Tasks:** `M1-WS4-02` metric/protocol adapter cùng deterministic fixtures; `M1-WS4-03` config + experiment metadata/evaluation CLI; `M1-WS4-04` integration tests sau khi providers công bố interfaces ổn định.
- **Tuân thủ:** C-02 đến C-06. Tách `upstream-basicsr` khỏi `realblur-official`; C-05 yêu cầu RGB tensor được adapter chuyển HWC và ghi `rgb2bgr` trước metric upstream.
- **Deliverables/DoD:** PSNR/SSIM fixture reference có evidence, GT mismatch/non-finite/unknown protocol errors, config validation không GPU, metadata đủ contract/config/checkpoint/dataset identity, integration tests chỉ claim phần đã chạy.
- **Integration boundary:** không sửa nghiệp vụ WS1–WS3; dùng public contracts/mock fixture cho test độc lập; end-to-end chỉ bắt đầu khi dependency đã được merge hoặc Tech Lead scope rõ.
- **Reviewer:** WS2 cho data/GT boundary, WS3 cho result boundary; Tech Lead là reviewer/owner ưu tiên WS4.
- **Cấm:** không gọi metric tự tính là paper reproduction, không thay official RealBlur protocol bằng BasicSR, không commit results/checkpoints lớn.

## Git collaboration workflow (PowerShell)

```powershell
# Clone lần đầu
git clone --recurse-submodules https://github.com/cuongdso/fftformer-image-deblurring.git
Set-Location fftformer-image-deblurring

# Checkout đã có sẵn: đồng bộ main và submodule
git status --short --branch
git fetch origin
git switch main
git pull --ff-only origin main
git submodule update --init --recursive
python scripts/check_env.py

# Một task, một branch ngắn hạn
git switch -c feat/model-model-spec       # WS1
# git switch -c feat/data-image-validator # WS2
# git switch -c feat/inference-pad-crop   # WS3
# git switch -c feat/evaluation-metrics   # WS4
```

Chỉ sửa scope đã giao, chạy test liên quan rồi commit/push/PR:

```powershell
python -m unittest discover -s tests -p "test_*.py"
python -m compileall -q scripts src
git diff --check
git status --short
git add <files-in-scope>
git commit -m "feat(<module>): <mô tả ngắn>"
git push -u origin <branch>
```

Mở PR vào `main` bằng template, nêu task ID, contracts, dependency, reviewer, tests và limitations. Trước khi rebase/merge giải quyết conflict, fetch `main` mới nhất và trao đổi với owner của file bị chồng lấn. Không force-push vào `main`, tự merge PR, rewrite/xóa branch người khác hoặc sửa module workstream khác khi chưa có thỏa thuận.

## AI coding agent operating rules

1. Đọc `AGENTS.md`, file này, contract và module guide trước khi sửa code.
2. Xác định workstream, task ID, scope và dependency; không tự nhận module khác.
3. Không sửa C-01..C-06, `third_party/FFTformer/` hay upstream submodule pin.
4. Không bịa benchmark/test result, không tải/commit dataset hoặc model weights lớn.
5. Giữ diff hẹp; không refactor diện rộng hoặc thêm API/config/dataset/metric chưa được xác minh.
6. Khi dependency thiếu, dùng mock/fixture theo contract hoặc nêu blocker; không tự đổi interface để unblock.
7. Chỉ báo cáo test đã thực sự chạy; đánh dấu `UNVERIFIED` với lý do.
8. Không merge `main`. Handoff luôn có commit, files, tests, limitations và PR URL.

## Quick-start prompt for a coding agent

```text
Repository: D:\xla\fftformer-image-deblurring
Workstream: <WS_ID>
Task ID: <TASK_ID>
Branch: <FEATURE_BRANCH>

Đọc AGENTS.md, INSTRUCTION.md, README.md, docs/project-status.md,
docs/architecture.md, docs/contracts.md, docs/ownership.md, docs/backlog.md
và docs/modules/<workstream>.md trước khi code. Khi liên quan upstream,
đọc docs/upstream-audit.md và reproduction docs phù hợp.

Thực hiện duy nhất <TASK_ID> trong phạm vi <WS_ID>. Tuân C-01..C-06 liên quan;
không tự sửa contracts, third_party/FFTformer, submodule pin, module workstream
khác, dataset/weights lớn hoặc lịch sử main. Nếu dependency chưa có, dùng mock/fixture
đúng contract hoặc ghi blocker, không tự mở rộng scope.

Chạy test phù hợp, git diff --check và các smoke check cần thiết. Commit và push
feature branch, tạo PR vào main nhưng không tự merge. Bàn giao: mục tiêu đạt được,
files changed, interface changes hoặc "Không", commands/kết quả thực tế,
limitations/UNVERIFIED, commit SHA, PR URL và next dependency.
```
