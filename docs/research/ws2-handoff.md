# Bàn giao WS2 — Dataset & Preprocessing

Ngày kiểm chứng: 18/11/2026. Owner: `phanthek2`. Branch:
`feat/data-image-validator`, base `f06de9432052af873aeb8b5ff8c85217d24ed222`.
Task IDs: `M1-WS2-01` (audit), `M1-WS2-02` (image boundary), `M1-WS2-03` (loader).

**Trạng thái:** đã kiểm chứng local; người dùng đã cho phép commit/push feature
branch ngày 18/11/2026. Chờ PR/review, chưa merge vào `main`.
Tài khoản xuất bản GitHub: `phanthek2-arch`; tên owner trong phân công: `phanthek2`.
Reviewer: TBD, theo vai trò WS1/WS3/WS4 và Tech Lead tại
[ownership](../ownership.md). Không coi kiểm chứng local là acceptance/merge của nhóm.

Commits triển khai:

- WS2-02: `a24c62beeb4e069d6338a1327584cafee8fb0841` — image validation/normalization.
- WS2-03: `0c108a67b882d5ccb934cfa89c14a4938d35f26b` — paired/non-GT dataset.
- Audit/verifier/hồ sơ được commit riêng sau hai commit code; xem `git log` của
  `feat/data-image-validator` để lấy SHA. JSON ghi lần kiểm chứng trước publication,
  source được đối chiếu theo SHA-256 LF khi chuẩn bị commit.

## Nội dung bàn giao

| Task | Phần đã làm | Files chính |
| --- | --- | --- |
| M1-WS2-01 | Đối chiếu nguồn/terms, native layout, preprocessing, fixture plan và hai cặp ảnh tác giả công bố; không đổi contract đã duyệt. | [Dataset audit](ws2-dataset-audit.md), `scripts/ws2_verify.py` |
| M1-WS2-02 | Image decode/validation, uint8 hoặc normalized float32 array → RGB float32 CHW, BGR conversion explicit, không sửa nguồn hoặc thay geometry. | `data/image.py`, `data/errors.py`, `tests/unit/test_data_image.py` |
| M1-WS2-03 | Sorted paired/single manifest, relative-path JSON schema v1, immutable records, lazy Sample sequence với target/GT flag rõ và preprocessing metadata. | `data/dataset.py`, `tests/unit/test_data_dataset.py` |
| Hỗ trợ | Public exports, optional `data` dependency extra, fixtures synthetic 224 bytes, docs/owner/status. | `data/__init__.py`, `pyproject.toml`, `tests/fixtures/ws2/`, [data guide](../modules/data.md) |

Paths `data/*.py` trong bảng thuộc `src/fftformer_lab/`. Mã nhóm được viết độc lập
theo [C-02/C-03](../contracts.md), đối chiếu `test.py`, image utils và paired/single
dataset classes ở official pin `01306b5d4072146a011c8c6e9d4927cf4607f1d5`.
Không copy hoặc chỉnh upstream, không thêm shared contract mới và không sửa WS1/3/4.

API Python mới thuộc WS2 được mô tả tại [data guide](../modules/data.md). Consumer
WS3 hiện nhận trực tiếp `ImageDataset` qua `run_batch`, dùng `image` và `sample_id`.
Consumer WS4 có `target`, `has_ground_truth` và `preprocessing_metadata.to_dict()`;
WS4 tiếp tục sở hữu evaluator/protocol/config thí nghiệm. Manifest JSON của WS2
chỉ lưu inventory, không tự xác nhận license/split/provenance dataset bên ngoài.

Checklist review theo [bản nháp PR](ws2-pr-draft.md). Đây là self-review có evidence,
không thay review/acceptance của WS1/WS3/WS4 hoặc Tech Lead.

## Commands và kết quả thực tế

Từ repository root, Windows 11, Python 3.13.0, torch 2.15.0+cpu, NumPy 2.5.5,
Pillow 13.0.0, einops 0.8.2; CPU threads giới hạn 2 cho full suite/smoke:

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests/unit -p 'test_data_*.py'
# Ran 40 tests; OK; không skip.

$env:FFTFORMER_RUN_MODEL_TESTS = '1'
$env:FFTFORMER_GOPRO_CHECKPOINT = (Resolve-Path 'third_party/FFTformer/pretrain_model/fftformer_GoPro.pth').Path
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -p 'test_*.py'
# Ran 137 tests; OK; không skip.

python -B -S -m unittest discover -s tests -p 'test_*.py'
# Ran 137 tests; OK (skipped=87): 50 chạy qua, 87 opt-in/dependency tests skip.

.\.venv\Scripts\python.exe -m compileall -q scripts src tests
git diff --check
# Exit 0.
```

Verifier tái lập toàn bộ các bước trên, kiểm dependency, ghi stdout/stderr/exit code,
SHA-256 nguồn và kết quả paired/non-GT trên ảnh tác giả:

```powershell
.\.venv\Scripts\python.exe -B scripts/ws2_verify.py --repo . --examples-dir data/ws2-gopro-examples --checkpoint third_party/FFTformer/pretrain_model/fftformer_GoPro.pth --output docs/research/ws2-runtime-evidence.json
```

Máy khác chưa có các ảnh ví dụ có thể thêm `--download-examples`: chỉ tải bốn PNG
đã pin, tổng 5,016,042 bytes, kiểm Git blob ID, giữ attribution và không ghi đè file
khác checksum. Không tải toàn dataset/weights. Không có flag này thì không gọi mạng.
Để chạy đầy đủ model/checkpoint suite còn cần `einops` và upstream đã init.
Mặc định không truyền `--checkpoint` sẽ skip các tests model opt-in và không chạy
model smoke; trường hợp đó được ghi rõ trong JSON. Runtime WS2 tests bị skip sẽ
làm verifier báo FAIL thay vì tuyên bố nghiệm thu khi thiếu image dependencies.

Tests mới gồm 18 tests dùng standard library và 22 tests cần image/tensor deps.
Có kiểm màu RGB chính xác, uint8 roundtrip, storage isolation, normalized float
không bị chia 255 lần nữa, caller dtype/device defaults, grayscale/alpha/16-bit/
animation/corrupt rejection, deterministic ordering, missing/orphan/duplicate,
JSON duplicate keys, relocation, GT flag, size/channel mismatch và WS3 mock boundary.
Tests chỉ dùng fixture nhóm tự sinh; CI không cần tải dataset hoặc cài model deps.

Smoke model thực đã chạy qua file reader/`ImageDataset`,
`load_model(ModelSpec(..., device="cpu"))` và `InferencePipeline(adapter).run_batch(dataset)`:

| Synthetic sample | Shape input và restored (CHW) | Padding (top,bottom,left,right) |
| --- | --- | --- |
| `sample_001.png` | `[3,64,64]` | `[0,0,0,0]` |
| `sample_002.png` | `[3,64,96]` | `[0,0,0,0]` |
| `sample_003.png` | `[3,50,70]` | `[0,14,0,26]` |

Verifier tự tạo ba ảnh RGB trong thư mục tạm, đọc theo non-GT mode và kiểm ID,
shape, float32, finite/range, `is_mock=False` và SHA-256 nguồn trước/sau. Strict checkpoint
GoPro đã có trong submodule, hash
`336717b7c4e97b66af112cc7878106cb185b698bd872c2a8247911507540b2a3`.
Không dùng output synthetic để đo hoặc tuyên bố chất lượng khử mờ.

## Dữ liệu thật đã kiểm tra

- Hai cặp Flower/Istanbul trong README tác giả GoPro, revision
  `2d5a698560e658718f0520e48dfb15bd52c80118`, đã tải và đọc ở paired/non-GT modes.
- Mỗi ảnh RGB uint8 1280x720 → float32 `[3,720,1280]`; paired target cùng shape,
  non-GT target/target_path là None và flag false; ID order ổn định.
- Bốn file tổng khoảng 5 MB; source blob ID/byte count và SHA-256 được kiểm tra.
  Nguồn và attribution nằm tại [audit](ws2-dataset-audit.md) và ignored
  `data/ws2-gopro-examples/ATTRIBUTION.md`; không đưa ảnh ngoài vào Git.
- Đây là ảnh ví dụ công bố, không phải validation toàn bộ test split; không
  suy đoán CRF/split và không chạy quality benchmark. Model smoke bên trên dùng
  ảnh synthetic riêng để giữ kiểm tra CPU nhỏ và tái lập độc lập dataset lớn.

## Phạm vi self-review trước bàn giao

- Các public APIs, error context, conversion/mutation và paired/non-GT invariants
  đã đối chiếu C-02/C-03 bằng code và tests. Không thay shared fields WS1/WS3/WS4.
- Import/manifest vẫn dùng được với standard library; image deps là extra `data`.
- Source files, fixtures và verifier có fingerprint raw/LF trong runtime JSON;
  hash LF dùng để đối chiếu khi Git Windows chuyển line endings.
- Data/checkpoints/.venv và ảnh tải ngoài vẫn ignored; candidate PNG trong Git
  chỉ gồm ba fixture tự tạo tổng 224 bytes. Push feature branch được người dùng
  cho phép; không tự merge hoặc gửi thông báo tới reviewer.

## Giới hạn và bước tiếp theo

- Chờ PR/review C-02/C-03 consumer boundary theo các commit task đã tách; không tự
  merge `main` hoặc tuyên bố team review đã hoàn tất.
- Báo cáo WS2-01 đã có local, chờ nhóm review; RealBlur mới được audit nguồn/terms,
  chưa tải hoặc chạy. Chưa kiểm chứng toàn bộ GoPro/RealBlur split.
- Reader hiện chỉ hỗ trợ RGB 8-bit PNG/JPEG một frame; không tự convert grayscale,
  alpha, palette, 16-bit, EXIF orientation hoặc profile màu.
- Không có LMDB, random crop/augmentation/training loader, image writer/CLI,
  evaluator hoặc CUDA validation. Không đổi padding policy để ảnh nhỏ qua inference.
- Manifest check chỉ xác nhận inventory và files tồn tại; muốn kiểm mọi ảnh phải
  iterate dataset. Default PyTorch collation không hỗ trợ dataclass Sample;
  consumer hiện dùng sequence như API WS3 đã có.

[Runtime evidence](ws2-runtime-evidence.json) ghi commands/results đã chạy và
SHA-256 code/tests/fixtures của bản local để reviewer đối chiếu.
