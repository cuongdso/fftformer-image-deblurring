# Upstream audit — FFTformer official baseline

## Identity và phạm vi audit

- **Source:** <https://github.com/kkkls/FFTformer>
- **Pinned SHA:** `01306b5d4072146a011c8c6e9d4927cf4607f1d5` (`main` khi khảo sát 2026-11-13)
- **Local path:** `third_party/FFTformer/` (Git submodule)
- **License:** MIT, copyright (c) 2023 kkkls, đã đọc file `license` trong pin.
- **Method:** static audit file/code; **không** cài dependencies, tải artifact mới, chạy test, benchmark hay training ở M0.

## File quan trọng và vai trò

| Path (trong submodule) | Vai trò đã xác minh |
| --- | --- |
| `README.md` | mô tả paper, dependency khái quát, `train.sh`, `test.sh`, link checkpoint RealBlur/evaluation |
| `requirements.txt` | `addict`, `future`, `lmdb`, `numpy`, `opencv-python`, `Pillow`, `pyyaml`, `requests`, `scikit-image`, `scipy`, `tb-nightly`, `tqdm`, `einops` |
| `setup.py` | package `basicsr`; import PyTorch; `get_requirements()` hiện return `[]`; optional C++/CUDA extension, `--no_cuda_ext` tắt build |
| `test.sh` / `test.py` | `setup.py develop --no_cuda_ext`; load `fftformer()` + strict state dict, paired test data, save output |
| `train.sh` | `torch.distributed.launch --nproc_per_node=8 … GoPro.yml` |
| `basicsr/models/archs/fftformer_arch.py` | `DFFN`, `FSAS`, `TransformerBlock`, encoder/decoder `fftformer` |
| `options/train/GoPro.yml`, `Realblur.yml` | PairedImageDataset LMDB, 8 GPU, config training/validation metrics |
| `pretrain_model/fftformer_GoPro.pth` | upstream-tracked file, 66,441,171 bytes at this checkout; no group copy is committed |

## Luồng inference upstream

`test.sh` chạy `python setup.py develop --no_cuda_ext` rồi `python test.py`. `test.py` mặc định tạo `fftformer()`; `torch.load(args.test_model)` và `load_state_dict(..., strict=True)`. Dataset code đọc `<data_dir>/test/input/` và `<data_dir>/test/target/` với tên file tương ứng, chuyển bằng `torchvision.transforms.functional.to_tensor`, pad reflection ở cạnh phải/dưới đến bội số 32, predict, crop về kích thước ban đầu, clamp `[0,1]` và lưu PIL RGB tại `results/<model_name>/GoPro/`.

Script test **không** tính PSNR/SSIM trong loop. README nói PSNR RealBlur được đo theo official RealBlur test code. Vì `test.py` luôn đặt `torch.device('cuda')` và gọi CUDA cache/synchronize, không được mô tả là CPU-compatible.

## Kiến trúc đã đối chiếu với code

`fftformer` dùng `OverlapPatchEmbed`, ba mức encoder với DFFN-only blocks, ba mức decoder và refinement với `att=True`, sau đó residual add với input. `FSAS` tách Q/K/V, chia patch 8×8, `rfft2`, nhân Q/K theo phần tử, `irfft2`, normalization và gating với V. `DFFN` dùng patch 8×8, learned frequency tensor, `rfft2/irfft2`, depth-wise convolution và GELU gate. Cấu hình `GoPro.yml` khai báo `dim: 48`, `num_blocks: [6,6,12]`, refinement 4, expansion 3; constructor default có `num_blocks=[6,6,12,8]` nhưng chỉ ba phần tử đầu được index. Compatibility checkpoint/config vẫn cần WS1 xác minh.

## Dependency, dữ liệu, training và metric

- README nêu Python, PyTorch 1.11, scikit-image, OpenCV, TensorBoard, einops. `requirements.txt` không pin version và không liệt kê PyTorch/torchvision; `setup.py` cũng không đưa requirements vào `install_requires`. Đây là rủi ro cài đặt đã xác minh.
- `GoPro.yml` và `Realblur.yml` dùng `PairedImageDataset` + LMDB; đều đặt `num_gpu: 8`, seed 42. GoPro train path là `/data0/.../GoPro_lmdb/{sharp,blur}.lmdb`, val dùng `./datasets/GoPro/test/{sharp_crops,blur_crops}.lmdb`; RealBlur config dùng `/data0/.../Realblur-J/train/{sharp,blur}.lmdb`.
- Training: 300,000 iterations, AdamW, L1 + FFT loss, DDP/NCCL qua entrypoint deprecated `torch.distributed.launch`. Đây là configuration upstream, chưa chạy.
- YAML validation khai báo `calculate_psnr` và `calculate_ssim` với crop 0, không Y-channel. BasicSR có metric code; README riêng chỉ dẫn official RealBlur evaluator cho kết quả họ báo cáo. Protocol benchmark chính xác cần WS4 đối chiếu trước khi claim reproduction.

## Không portable / rủi ro

1. Test script có absolute path mặc định, bắt buộc target image và CUDA hard-code.
2. Mọi kích thước được pad 32 cho test nhưng FSAS/DFFN rearrange theo patch 8; wrapper nhóm phải giữ policy có test, không tự đơn giản hóa.
3. `setup.py develop`/`torch.distributed.launch` cũ có thể không phù hợp toolchain PyTorch mới; chưa chạy nên không kết luận compatibility.
4. `setup.py` metadata nêu BasicSR/Apache classifier trong khi upstream top-level license là MIT; nhóm chỉ dựa top-level MIT cho submodule, không diễn giải license dependency.
5. README link checkpoint RealBlur là Google Drive folder, không công bố checksum/tên mapping đầy đủ trong static audit; không tự coi mọi checkpoint cùng kiến trúc là tương thích.

## Chưa xác minh

- Exact Python/CUDA/driver/OS combo đã chạy upstream.
- Download source, integrity checksum, redistribution right của pretrained weights; license từng dataset.
- Dataset sizes, split, crop generation và mapping filename thực tế.
- Full parameter count, checkpoint schema của từng artifact, benchmark result và paper reproduction.
- Các claim dataset trong paper ngoài những config/README đã audit trực tiếp.
