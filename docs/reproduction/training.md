# Training official baseline

**Trạng thái:** audit-only; M0 không training, fine-tuning, tải dataset hay tạo checkpoint.

## Entrypoint và config đã xác minh

Upstream `train.sh` chạy:

```bash
python setup.py develop --no_cuda_ext
python -m torch.distributed.launch --nproc_per_node=8 --master_port=4321 basicsr/train.py -opt options/train/GoPro.yml --launcher pytorch
```

`GoPro.yml` khai báo 8 GPU, paired LMDB, crop 128, batch/GPU 8, 300,000 iterations, AdamW, L1 + FFT loss, TensorBoard logging và DDP backend `nccl`. `Realblur.yml` tương tự nhưng train crop 256 và data roots RealBlur-J. Các absolute data paths trong upstream không portable.

## Phân biệt scope

- **Full reproduction:** audit đủ dataset/terms, environment, distributed resources, config, checkpoint/evaluation protocol để đối chiếu paper; chưa bắt đầu.
- **Fine-tuning:** bắt đầu từ weight phù hợp với dataset/config mới; cần task/metadata riêng, không gọi là full reproduction.
- **Training thử nghiệm nhỏ:** resource/config thay đổi; chỉ dùng để test pipeline, không dùng khẳng định paper result.

## Resume và artifact

YAML có `path.resume_state`; BasicSR source có logic save/resume state. Chỉ ghi là đã hỗ trợ sau khi run thật sự xác minh. Với mọi run, version config, seed, hardware, checkpoint/log paths và evaluator phải nằm trong metadata. Không commit checkpoint/log lớn.

## Rủi ro hiện có

`torch.distributed.launch` là entrypoint legacy với PyTorch mới; NCCL/8 GPU là assumption code/config. Không thay baseline để xử lý resource yếu mà không tạo experiment/implementation tách riêng và ghi deviation.
