# Inference official baseline

**Trạng thái:** lệnh được audit từ upstream pin; **NOT RUN** ở M0. Không mô tả `test.py` là tool single-image hoặc CPU inference.

## Preconditions đã xác minh

- Checkout submodule tại SHA pin.
- PyTorch/CUDA environment hoạt động; `test.py` hard-code `torch.device('cuda')`.
- Checkpoint state-dict tương thích strict với `fftformer()`.
- Dữ liệu theo paired layout `test/input` + `test/target`; xem [datasets](datasets.md).

## Entrypoint upstream

Từ `third_party/FFTformer/`, upstream cung cấp:

```bash
python setup.py develop --no_cuda_ext
python test.py --data_dir <paired-data-root> --test_model <checkpoint-path> --model_name <run-name>
```

`test.sh` chỉ gọi hai lệnh đầu với argument mặc định. Output ảnh được lưu tại `results/<model_name>/GoPro/`; code padding reflection đến multiple 32, crop trở lại H×W gốc, clamp `[0,1]`, thêm `0.5/255` trước khi serialize PIL RGB.

## Kiểm tra kết quả

Chỉ kiểm tra rằng output được tạo/số filename khớp sau khi command thật sự hoàn thành. Điều đó **không** chứng minh chất lượng hay reproduction benchmark. Lưu command, commit upstream, checkpoint identity, data identity và environment vào `experiments/<id>/metadata.md`.

## Lỗi đã biết từ audit

- CUDA unavailable: script không CPU fallback do `torch.device('cuda')`, `empty_cache`, `synchronize`.
- Path/target thiếu: loader luôn mở `<data_dir>/test/target/<name>`.
- Checkpoint mismatch: strict `load_state_dict` fail.
- Invalid image dimensions: reflection padding hoặc model patch/rearrange có thể fail; policy group phải được test trước khi wrapper khác baseline.

Wrapper nhóm là `PLANNED` trong WS3 và phải giải thích mọi khác biệt với script này.
