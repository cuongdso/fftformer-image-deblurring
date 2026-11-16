# M1-WS1-01 — Đối chiếu kiến trúc và kiểm tra tương thích checkpoint FFTformer

- Workstream: WS1 — Model & Architecture
- Reviewer theo backlog: WS3
- Ngày lập: 16/11/2026
- Trạng thái: Hoàn thành phần đối chiếu và kiểm tra sơ bộ; chờ review qua Pull Request

## 1. Mục tiêu và phạm vi

Nhiệm vụ `M1-WS1-01` trong `docs/backlog.md` yêu cầu kiểm tra cấu trúc các khối mạng, cấu hình và khả năng tương thích checkpoint của FFTformer. Hai đầu ra chính là bảng đối chiếu giữa bài báo và mã nguồn (*paper–code mapping*), cùng kế hoạch kiểm thử kích thước tensor và checkpoint. Kết quả được dùng làm cơ sở triển khai `M1-WS1-02` (model adapter) và `M1-WS1-03` (checkpoint loader).

Phạm vi kiểm tra gồm kiến trúc encoder–decoder, FSAS, DFFN, các tham số khởi tạo, đường đi của tensor trong `forward()` và khả năng nạp checkpoint GoPro vào model tương ứng. Phần kiểm thử chỉ thực hiện trên mã FFTformer gốc và trọng số sẵn có; chưa có model adapter hoặc checkpoint loader do nhóm tự phát triển. Việc huấn luyện lại, đo PSNR/SSIM và benchmark GPU không thuộc kết quả của nhiệm vụ này.

### 1.1. Nguồn tham chiếu

| Thành phần | Định danh |
| --- | --- |
| Bài báo | L. Kong và cộng sự, *Efficient Frequency Domain-based Transformers for High-Quality Image Deblurring*, CVPR 2023; chủ yếu §3.1–3.3, Hình 2, công thức (4)–(8) và §4.1 |
| Mã nguồn gốc | Git submodule `third_party/FFTformer`, commit `01306b5d4072146a011c8c6e9d4927cf4607f1d5` |
| File kiến trúc | `third_party/FFTformer/basicsr/models/archs/fftformer_arch.py` |
| SHA-256 của file kiến trúc | `dc0e1775eed3c3358f637084e9e345ec0246b66084487dc4220aad0a5cb66ddf` |
| Cấu hình | `third_party/FFTformer/options/train/GoPro.yml`, `third_party/FFTformer/options/train/Realblur.yml` |
| Mã kiểm thử chính thức | `third_party/FFTformer/test.py` |
| Checkpoint được kiểm tra | `third_party/FFTformer/pretrain_model/fftformer_GoPro.pth` |

SHA commit xác định phiên bản mã nguồn, còn SHA-256 checkpoint tại Mục 5 xác định riêng file trọng số. Checkpoint này được Git của submodule theo dõi tại commit đã pin; repository nhóm chỉ tham chiếu submodule và không sao chép checkpoint vào lịch sử Git của nhóm.

Bản audit sử dụng hai loại bằng chứng: kết quả đọc mã nguồn/cấu hình và kết quả chạy kiểm thử. Các nhận xét không được kiểm chứng bằng thực nghiệm được ghi rõ ở phần giới hạn. C-01 và C-02 trong `docs/contracts.md` (v1.0, `APPROVED`) là ranh giới kỹ thuật áp dụng cho các bước triển khai tiếp theo; báo cáo này không thay đổi nội dung contract.

## 2. Đối chiếu bài báo với mã nguồn

Các số dòng trong Mục 2 và Mục 3 đều tham chiếu `fftformer_arch.py` tại commit nêu ở Mục 1.1.

| Nội dung trong bài báo | Vị trí mã nguồn | Kết quả đối chiếu |
| --- | --- | --- |
| Kiến trúc encoder–decoder bất đối xứng; đầu ra có residual với ảnh mờ (§3.3, Hình 2a, công thức 8) | `fftformer`, dòng 218–297 | `forward()` trả kết quả qua `self.output(...) + inp_img` ở dòng 295. |
| Trích xuất đặc trưng đầu vào (Hình 2a) | `OverlapPatchEmbed`, dòng 179–189 | Conv 3×3, stride 1; tạo 48 kênh đặc trưng ở độ phân giải ban đầu. |
| Encoder chỉ sử dụng DFFN (§3.3) | `encoder_level1/2/3`, dòng 232–244; `TransformerBlock`, dòng 138–156 | Các block encoder dùng `att=False`, không gọi FSAS. |
| Decoder kết hợp FSAS và DFFN (§3.3) | `decoder_level3/2/1`, dòng 246–260; `refinement`, dòng 262–264 | Các block dùng `att=True`, có nhánh attention và nhánh FFN với residual riêng. |
| Thay đổi độ phân giải và kết hợp skip features (Hình 2a) | `Downsample`, dòng 194–203; `Upsample`, dòng 205–213; `Fuse`, dòng 159–175 | Resize bilinear kết hợp convolution; `Fuse` ghép đặc trưng, xử lý bằng block DFFN, sau đó tách và cộng các nhánh. |
| FSAS: giải attention trong miền tần số (§3.1, Hình 2b, công thức 4–6) | `FSAS`, dòng 99–134 | Q/K/V qua conv 1×1 và depth-wise conv 3×3; FFT trên patch, nhân phần tử, IFFT, chuẩn hóa, kết hợp V và chiếu kênh. |
| DFFN: chọn lọc thông tin tần số (§3.2, Hình 2c, công thức 7) | `DFFN`, dòng 65–96 | FFT theo patch 8×8, nhân với tham số tần số học được, IFFT, depth-wise conv và cổng GELU. |
| Chuẩn hóa đặc trưng (Hình 2b–c) | `LayerNorm`, dòng 16–62 | Mặc định dùng biến thể `WithBias`; chuẩn hóa theo kênh tại mỗi vị trí không gian. |
| Patch tần số 8×8 (§4.1) | `DFFN` dòng 72; `FSAS` dòng 110 | Cả hai khối dùng `patch_size=8`; kích thước không gian của feature phải phù hợp khi tách patch. |

### 2.1. FSAS

Đầu vào của một khối FSAS có shape `[B,C,H,W]`. Trong `FSAS.forward()`, `to_hidden` và depth-wise convolution tạo tensor có `6C` kênh (dòng 103–115). Tensor này được tách bằng `chunk(3, dim=1)` thành `q`, `k`, `v`, mỗi nhánh có `2C` kênh.

Hai nhánh `q` và `k` được sắp xếp thành các patch 8×8 (dòng 117–120). Code áp dụng `torch.fft.rfft2` cho từng nhánh, nhân `q_fft * k_fft`, sau đó sử dụng `torch.fft.irfft2` để trở lại miền không gian (dòng 121–125). Các patch được ghép về feature map, chuẩn hóa bằng `LayerNorm(2C)`, nhân với `v` và chiếu từ `2C` kênh về `C` kênh (dòng 126–134). `TransformerBlock` cộng kết quả với đầu vào trước khi chuyển sang DFFN.

Có một khác biệt giữa công thức và code cần lưu ý. Công thức (4) trong bài báo mô tả tương quan trong miền Fourier với phép liên hợp phức, trong khi mã nguồn tính trực tiếp `q_fft * k_fft` mà không gọi phép liên hợp. Theo quy ước Fourier thông thường, phép nhân hai phổ như trong code tương ứng với tích chập vòng, còn tương quan chéo cần liên hợp một phổ. Báo cáo ghi nhận khác biệt này ở mức đối chiếu toán học, chưa có bằng chứng về lý do tác giả chọn cách triển khai như vậy.

FSAS trong code cũng không tạo ma trận attention không gian kích thước `N×N` và không thực hiện `softmax` trên ma trận đó. Khi trình bày, cần phân biệt bộ giải attention trong miền tần số của FFTformer với phép self-attention đầy đủ trong Transformer thông thường. Mã nguồn upstream được giữ nguyên để bảo đảm khả năng đối chiếu và tương thích checkpoint.

### 2.2. DFFN và ngoại lệ trong `Fuse`

DFFN bắt đầu bằng convolution 1×1 để mở rộng số kênh. Sau đó đặc trưng được chia thành patch 8×8, biến đổi bằng `rfft2`, nhân với tham số `self.fft` học được và biến đổi ngược bằng `irfft2` (dòng 70–91). Một depth-wise convolution được áp dụng trước khi tách đặc trưng thành hai phần; đầu ra cổng là `gelu(x1) * x2`, rồi được chiếu về số kênh ban đầu (dòng 92–96).

Số kênh ở phần mở rộng phụ thuộc `ffn_expansion_factor`:

```text
hidden_features = int(C * ffn_expansion_factor)
project_in output channels = 2 * hidden_features
self.fft shape = [2 * hidden_features, 1, 1, 8, 5]
```

Encoder, Decoder và Refinement nhận `ffn_expansion_factor=3` từ constructor `fftformer`, nên `project_in` có `6C` kênh. Hai khối `Fuse` là ngoại lệ: ở dòng 163, `Fuse` tạo `TransformerBlock(dim=n_feat * 2)` nhưng không truyền `ffn_expansion_factor`. Vì vậy, chúng sử dụng giá trị mặc định `2.66` của `TransformerBlock` tại dòng 139.

| Khối | Số kênh `C` | Hệ số mở rộng | Shape của `self.fft` |
| --- | ---: | ---: | --- |
| `fuse1.att_channel.ffn` | 96 | 2.66 | `[510,1,1,8,5]` |
| `fuse2.att_channel.ffn` | 192 | 2.66 | `[1020,1,1,8,5]` |

Do đó, thay đổi tham số `ffn_expansion_factor` ở constructor `fftformer` không tự động thay đổi hệ số của hai DFFN trong `Fuse`.

Về phương pháp, bài báo lấy cảm hứng từ cơ chế chọn lọc hệ số tần số của JPEG. Triển khai DFFN dùng FFT và trọng số nhân có thể học; code không thực hiện trực tiếp chuỗi thao tác DCT, làm tròn và lượng tử hóa cứng của bộ mã hóa JPEG. Đây là một cơ chế học trọng số tần số, không phải bộ nén JPEG.

## 3. Cấu trúc mạng và kích thước tensor

Đầu vào model là tensor ảnh RGB kiểu `float32`, shape `[B,3,H,W]` theo C-02. Với kiến trúc hiện tại, điều kiện đủ để các feature ở cả ba mức chia được thành patch 8×8 là `H` và `W` cùng chia hết cho 32: mạng giảm độ phân giải hai lần, tức feature sâu nhất có kích thước `H/4 × W/4`.

| Bước trong `forward()` | Dòng code | Shape tổng quát | Ví dụ với đầu vào `[1,3,32,32]` |
| --- | --- | --- | --- |
| `inp_img` | 270 | `[B,3,H,W]` | `[1,3,32,32]` |
| `patch_embed`, `encoder_level1` | 271–272 | `[B,48,H,W]` | `[1,48,32,32]` |
| `down1_2`, `encoder_level2` | 274–275 | `[B,96,H/2,W/2]` | `[1,96,16,16]` |
| `down2_3`, `encoder_level3` | 277–278 | `[B,192,H/4,W/4]` | `[1,192,8,8]` |
| `decoder_level3` | 280 | `[B,192,H/4,W/4]` | `[1,192,8,8]` |
| `up3_2`, `fuse2`, `decoder_level2` | 282–286 | `[B,96,H/2,W/2]` | `[1,96,16,16]` |
| `up2_1`, `fuse1`, `decoder_level1` | 288–291 | `[B,48,H,W]` | `[1,48,32,32]` |
| `refinement` | 293 | `[B,48,H,W]` | `[1,48,32,32]` |
| `output` và residual với `inp_img` | 295–297 | `[B,3,H,W]` | `[1,3,32,32]` |

Khi khởi tạo model trên CPU từ cấu hình GoPro, số lượng module và tham số thu được như sau:

| Thành phần | Số lượng |
| --- | ---: |
| TransformerBlock ở encoder, không có FSAS | `6 + 6 + 12 = 24` |
| TransformerBlock ở decoder, có FSAS | `12 + 6 + 6 = 24` |
| TransformerBlock ở refinement, có FSAS | 4 |
| TransformerBlock trong hai khối `Fuse`, không có FSAS | 2 |
| Tổng module FSAS | 28 |
| Tổng module DFFN | 54 (52 dùng hệ số 3; 2 trong `Fuse` dùng 2.66) |
| Tham số có thể học | 16.560.474 (~16,6 triệu) |
| Số mục `state_dict` | 535 |

Hai chi tiết trong code cần được giữ khi xây dựng adapter. Thứ nhất, `reduce_chan_level2` được khai báo tại dòng 251 nhưng không được gọi trong `forward()`; các tham số của nó vẫn có trong `state_dict`. Thứ hai, mặc định constructor khai báo `num_blocks=[6,6,12,8]`, nhưng phần triển khai chỉ sử dụng ba phần tử đầu, trong khi `GoPro.yml` đặt `[6,6,12]`. Hai cách khởi tạo cho số tham số và bộ key giống nhau trong phép kiểm tra đã thực hiện. Không có cơ sở để loại bỏ tham số hoặc sửa constructor chỉ vì một thành phần chưa xuất hiện trong `forward()`.

## 4. Đối chiếu cấu hình

### 4.1. Cấu hình kiến trúc

`options/train/GoPro.yml:38–46` và `options/train/Realblur.yml:38–46` khai báo cùng nhóm tham số `network_g`:

```yaml
network_g:
  type: fftformer
  inp_channels: 3
  out_channels: 3
  dim: 48
  num_blocks: [6, 6, 12]
  num_refinement_blocks: 4
  ffn_expansion_factor: 3
  bias: false
```

Phần lớn giá trị này cũng là mặc định trong constructor `fftformer` (`fftformer_arch.py:218–227`), ngoại trừ danh sách `num_blocks` có thêm phần tử thứ tư không được sử dụng. Trong `test.py:67`, mô hình được tạo trực tiếp bằng `fftformer()` thay vì đọc `network_g` từ YAML.

Kết quả kiểm tra trên CPU cho thấy cả cách khởi tạo mặc định và cách khởi tạo từ GoPro YAML đều tạo ra 16.560.474 tham số, 535 mục `state_dict`, 28 module FSAS và 54 module DFFN. Phép nạp checkpoint với `strict=True` cũng thành công. Kết quả này xác nhận tính tương thích của hai cách khởi tạo với checkpoint đang kiểm tra; nó không xác nhận các thiết lập huấn luyện là giống nhau.

### 4.2. Thiết lập huấn luyện trong bài báo và YAML

| Nội dung | Bài báo, §4.1 | `GoPro.yml` |
| --- | --- | --- |
| Optimizer | Adam với tham số mặc định | AdamW; `weight_decay=1e-3`, `betas=[0.9,0.9]` |
| Learning rate ban đầu | `1e-3` | `1e-3` |
| Lịch learning rate | Mô tả cosine annealing và nêu mốc 600.000 iterations; câu văn chưa chỉ rõ đó có phải tổng số bước huấn luyện | `total_iter=300000`; scheduler `T_max=300000` |
| Crop/patch huấn luyện | 256×256 | `gt_size=128` |
| Batch size | 16 | `batch_size_per_gpu=8`, `num_gpu=8`; batch thực tế còn phụ thuộc cấu hình chạy |
| Patch dùng cho FSAS/DFFN | 8×8 | Cố định 8×8 trong model |
| Hàm loss | Dẫn chiếu phương pháp ở tài liệu [4] | `L1Loss` và `FFTLoss` với trọng số 1,0 và 0,1 |

Mốc 600.000 iterations trong §4.1 không nên được diễn giải thành tổng số bước huấn luyện khi chưa có căn cứ bổ sung. Ngược lại, giá trị `total_iter=300000` là thông số cụ thể của YAML. Các khác biệt về optimizer, crop và batch size cho thấy cấu hình phát hành cần được phân biệt với phần mô tả trong bài báo khi lập kế hoạch reproduction.

`Realblur.yml` sử dụng cùng kiến trúc nhưng có `gt_size=256`, `batch_size_per_gpu=2` và `lr=4e-4` (`Realblur.yml:18–25,58–59`). Chưa thực hiện kiểm tra trọng số RealBlur nên không có kết luận về khả năng tương thích checkpoint của tập này.

## 5. Kiểm tra checkpoint và chạy thử model

### 5.1. Artifact và môi trường

Checkpoint được kiểm tra là `third_party/FFTformer/pretrain_model/fftformer_GoPro.pth`.

| Thuộc tính | Giá trị |
| --- | --- |
| Dung lượng | 66.441.171 byte |
| SHA-256 | `336717b7c4e97b66af112cc7878106cb185b698bd872c2a8247911507540b2a3` |
| Kiểu dữ liệu sau `torch.load` | `collections.OrderedDict` |
| Số tensor | 535 |
| Kiểu tensor | 535 tensor `torch.float32` |
| Python sử dụng trong lần chạy có evidence | 3.14.10 |
| PyTorch | 2.15.0+cpu |
| `einops` | 0.8.2 |
| `PyYAML` | 6.0.3 |
| Hệ điều hành / kiến trúc máy | Windows / AMD64 |
| Thiết bị | CPU; `torch.set_num_threads(2)` |

Các giá trị trên và kết quả ở Mục 5.2 được ghi nhận trong cùng lần chạy [script kiểm tra WS1-01](../../scripts/ws1_01_verify.py), với bằng chứng [ws1-01-runtime-evidence.json](ws1-01-runtime-evidence.json) mang nhãn `group-measured`, loại `integration-cpu-checkpoint`. JSON lưu thời điểm UTC tại `timestamp_utc`, phiên bản thư viện, số luồng CPU, upstream commit và SHA-256 của source kiến trúc, cấu hình, script, checkpoint. Đây là môi trường của lần chạy này, không mặc định là cấu hình trên máy của mọi thành viên; mỗi lần chạy lại cần lưu evidence của chính lần chạy đó.

### 5.2. Kết quả kiểm tra

| Phép thử | Kết quả | Nhận xét | Mã trong `checks` của JSON |
| --- | --- | --- | --- |
| Đọc checkpoint | PASS | `torch.load(..., map_location='cpu', weights_only=True)` đọc được mapping chứa tensor. | `checkpoint_raw_tensor_state_dict` |
| Tạo model từ GoPro YAML | PASS | Model có 16.560.474 tham số; metadata ghi 535 mục `state_dict`. | `parameter_count_16560474` |
| Nạp checkpoint nghiêm ngặt | PASS | `load_state_dict(strict=True)` không có `missing_keys` hoặc `unexpected_keys`; không phát sinh lỗi shape. | `strict_checkpoint_load` |
| Nạp vào constructor mặc định | PASS | Checkpoint tương thích với model tạo bằng `fftformer()`; ghi kèm số tham số, mục `state_dict` và module FSAS/DFFN. | `default_constructor_strict_checkpoint_load` |
| Forward với `[1,3,32,32]` | PASS | Đầu ra `[1,3,32,32]`, toàn bộ giá trị hữu hạn. | `forward_32x32` |
| Forward với `[1,3,64,96]` | PASS | Đầu ra `[1,3,64,96]`, toàn bộ giá trị hữu hạn. | `forward_64x96` |
| Forward với `[1,3,40,40]` | PASS — từ chối đúng dự kiến | Bắt đúng `EinopsError` và thông báo feature trung gian có chiều 20 không chia được thành patch 8. | `forward_40x40_rejected` |
| Checkpoint thiếu `patch_embed.proj.weight` | PASS — từ chối đúng dự kiến | `load_state_dict(strict=True)` báo `Missing key(s)`. | `missing_key_rejected` |
| Checkpoint có thêm `audit.unexpected.weight` | PASS — từ chối đúng dự kiến | Báo `Unexpected key(s)`. | `unexpected_key_rejected` |
| Checkpoint có weight shape `[47,3,3,3]` thay cho `[48,3,3,3]` | PASS — từ chối đúng dự kiến | Báo `size mismatch`. | `wrong_shape_rejected` |

Các trường hợp checkpoint lỗi được tạo bằng bản sao `state_dict` trong bộ nhớ; file `.pth` gốc không bị sửa. Hai phép thử forward hợp lệ dùng tensor `float32` toàn số 0 trong `torch.inference_mode()`, không dùng ảnh thật. Kết quả chứng minh model chạy được với các shape đã thử và checkpoint khớp cấu trúc; chưa phản ánh chất lượng khử mờ.

Script kiểm tra lại gồm mười phép thử tự động tương ứng mười dòng trong bảng: số tham số, định dạng checkpoint, strict load từ cấu hình YAML, strict load với constructor mặc định, hai forward hợp lệ, một forward không hợp lệ và ba trường hợp checkpoint lỗi. Kết quả thực tế là `10/10 PASS`. Với các trường hợp đầu vào/checkpoint sai, `PASS` nghĩa là lỗi được phát hiện đúng dự kiến, không phải model chấp nhận dữ liệu sai. [JSON evidence](ws1-01-runtime-evidence.json) lưu trạng thái từng phép thử, shape và trích thông báo lỗi; cần đọc cùng [mã kiểm thử](../../scripts/ws1_01_verify.py) để hiểu điều kiện kiểm tra.

### 5.3. Đường dẫn checkpoint trong `test.py`

`test.py:117` sử dụng đường dẫn mặc định `./pretrain_model/net_g_GoPro_HIDE.pth`, khác tên checkpoint đã kiểm tra là `pretrain_model/fftformer_GoPro.pth`. Trong tập artifact hiện có tại thời điểm audit, không tìm thấy file mang tên mặc định. Do đó chỉ có thể kết luận checkpoint `fftformer_GoPro.pth` tương thích ở mức `state_dict`; chưa xác định hai tên file có cùng trọng số hay không.

Script `test.py` chính thức dùng CUDA (`test.py:67–98`). Thành công trên CPU với việc gọi trực tiếp model không đồng nghĩa toàn bộ luồng `test.py` đã được xác minh chạy trên CPU.

## 6. Ma trận kiểm thử cho WS1-02 và WS1-03

Bảng dưới đây là kế hoạch kiểm thử cho mã adapter và checkpoint loader của nhóm. Kết quả chạy trực tiếp trên upstream tại Mục 5 chỉ cung cấp trường hợp đối chiếu, không được tính là kết quả kiểm thử hai module chưa triển khai. Các phép thử cần checkpoint thật được phân loại là integration và dùng artifact bên ngoài Git; unit test ưu tiên fixture nhỏ, có thể phân phối hợp pháp.

### 6.1. Model adapter và tensor shape (`M1-WS1-02`)

| ID | Trường hợp | Kết quả mong đợi |
| --- | --- | --- |
| `S01` | Khởi tạo model bằng `network_g` GoPro; đọc tham số và metadata | Đúng architecture ID, upstream pin, cấu hình; số tham số 16.560.474. |
| `S02` | Đầu vào RGB `float32`, NCHW `[1,3,32,32]` | Đầu ra cùng shape và hữu hạn; không mặc định clamp giá trị ảnh. |
| `S03` | Đầu vào `[1,3,64,96]` | Đầu ra `[1,3,64,96]` và hữu hạn. |
| `S04` | Đầu vào thiếu batch dimension, sai số kênh, chứa NaN/Inf hoặc ngoài range hợp đồng | Adapter báo lỗi đầu vào rõ ràng; không tự động chuyển đổi dữ liệu mà không được yêu cầu. |
| `S05` | Đầu vào chưa padding, chẳng hạn `[1,3,40,40]` | Adapter kiểm tra/nêu rõ tiền điều kiện shape. Padding và crop thuộc WS3 theo C-02/C-04. |
| `S06` | Architecture ID hoặc tham số constructor không hợp lệ | Báo lỗi cấu hình có ngữ cảnh, không để lỗi nội bộ khó xác định nguyên nhân. |

### 6.2. Checkpoint loader và device (`M1-WS1-03`)

| ID | Trường hợp | Kết quả mong đợi |
| --- | --- | --- |
| `C01` | Không tồn tại đường dẫn checkpoint | Báo file không tồn tại trước bước nạp. |
| `C02` | Checkpoint có schema không được hỗ trợ, ví dụ dictionary chứa key bao ngoài | Báo rõ schema không phù hợp; không tự đoán định dạng khác. |
| `C03` | Thiếu một key trong fixture | Strict load thất bại và chỉ ra key thiếu. |
| `C04` | Có key thừa trong fixture | Strict load thất bại và chỉ ra key thừa. |
| `C05` | Tên key đúng nhưng tensor shape sai | Báo tên tham số và shape không tương thích; không chuyển sang `strict=False`. |
| `C06` | Fixture checkpoint hợp lệ, có thể chia sẻ | Strict load thành công; model ở chế độ `eval()`. |
| `C07` | Checkpoint GoPro thật và kiến trúc tương ứng | Nạp nghiêm ngặt thành công; ghi checkpoint SHA-256, upstream commit, device và `map_location`. |
| `C08` | Yêu cầu CUDA khi không khả dụng hoặc device không hợp lệ | Báo rõ device không hỗ trợ/không khả dụng, không chuyển thiết bị âm thầm. |
| `C09` | Kiến trúc lệch checkpoint, ví dụ đổi `dim=32` | Báo lỗi key/shape khi strict load; không tự đổi cấu hình để ép nạp. |

Đã quan sát được các trường hợp tương đương `S02`, `S03`, `C03`, `C04`, `C05` và `C07` khi làm việc trực tiếp với mã gốc. Việc kiểm tra lại chúng qua API do nhóm viết vẫn thuộc WS1-02/03.

### 6.3. Cách chạy lại kiểm tra sơ bộ

Lệnh đã chạy từ thư mục gốc repository, với Python đã cài PyTorch, `einops`, `PyYAML`, Git khả dụng và checkpoint GoPro đặt tại đường dẫn ở Mục 5:

```powershell
python -B scripts/ws1_01_verify.py --repo . --output docs/research/ws1-01-runtime-evidence.json
```

Thông báo tổng kết của lần chạy:

```text
[PASS] default_constructor_strict_checkpoint_load
[PASS] forward_40x40_rejected
Wrote JSON evidence: docs/research/ws1-01-runtime-evidence.json
Summary: 10/10 passed
```

Hai dòng `PASS` trên là trích đoạn cho hai phép thử bổ sung; toàn bộ mười kết quả nằm trong [JSON evidence](ws1-01-runtime-evidence.json). Tùy chọn `-B` và việc tắt ghi bytecode khi import trong script tránh tạo thêm `.pyc` trong submodule. Script tự ghi các đường dẫn trong JSON tương đối với thư mục gốc repository, dùng dấu `/`, nên không cần sửa tay kết quả sau khi chạy. Nếu dùng `--checkpoint` để chỉ file ngoài repository, trường đường dẫn được ghi thành `<external>/<tên-file>` để ẩn thư mục máy cá nhân; SHA-256 vẫn định danh artifact thực đã nạp. Khi chạy lại để so sánh và muốn giữ evidence hiện tại, chọn một đường dẫn `--output` khác.

## 7. Giới hạn và nội dung bàn giao

Việc kiểm tra WS1-01 đưa ra ba kết luận có thể sử dụng cho giai đoạn triển khai. Thứ nhất, source FFTformer tại commit đã xác định có cấu trúc tương ứng với mô hình trong bài báo ở cấp block và luồng đặc trưng. Thứ hai, checkpoint GoPro mang SHA-256 nêu ở Mục 5 nạp được bằng `strict=True` vào model từ GoPro YAML và constructor mặc định. Thứ ba, forward trên CPU thành công với hai kích thước hợp lệ; các tình huống thiếu/thừa key và sai shape đều được phát hiện.

Các giới hạn còn lại cần được giữ rõ trong tài liệu và PR:

1. Công thức FSAS trong bài báo và phép nhân FFT trong code không hoàn toàn trùng nhau. Chưa xác định nguyên nhân của khác biệt này, và không sửa mã nguồn upstream trong phạm vi WS1-01.
2. Không có kết quả đo PSNR/SSIM trên GoPro, HIDE hoặc RealBlur; không có benchmark GPU hay tái hiện quá trình huấn luyện.
3. Chưa kiểm tra checkpoint RealBlur. File GoPro đã thử không được dùng để suy ra tính tương thích của các checkpoint khác.
4. Điều kiện chia patch khiến một số kích thước đầu vào cần padding. `test.py:91–98` dùng reflect-padding ở cạnh phải/dưới đến bội số 32 rồi crop về kích thước gốc. WS3 phụ trách pad/crop; WS1 cần công bố và kiểm tra tiền điều kiện khi tích hợp theo C-01/C-02.
5. `reduce_chan_level2` hiện có tham số nhưng không tham gia `forward()`. Thành phần này vẫn cần được giữ để checkpoint khớp cấu trúc.
6. Các kết quả được ghi nhận trên cấu hình CPU nêu tại Mục 5, không đại diện cho mọi phiên bản PyTorch hoặc mọi thiết bị.

Đối với `M1-WS1-02`, cần triển khai `ModelSpec` và adapter ở `src/fftformer_lab/model/` theo C-01/C-02, cung cấp architecture/config/device metadata và kiểm tra đầu vào. `M1-WS1-03` tiếp tục triển khai strict checkpoint loader và các trường hợp không tương thích ở Mục 6. Không thay đổi `third_party/FFTformer/` hoặc hợp đồng liên module trong phạm vi hai task này nếu chưa có quy trình phê duyệt tương ứng.

Bản audit đã có mapping, đường dẫn nguồn, bảng shape, thông tin checkpoint và ma trận kiểm thử theo tiêu chí của `M1-WS1-01`. Trạng thái `DONE` của task chỉ được cập nhật sau khi PR hoàn tất review và được phê duyệt theo quy trình nhóm.
