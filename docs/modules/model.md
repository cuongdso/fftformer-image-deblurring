# WS1 — Model & Architecture

**Owner/reviewer:** TBD. **M0:** không có code model nhóm.

Nghiên cứu `third_party/FFTformer/basicsr/models/archs/fftformer_arch.py` theo pin, đối chiếu paper trước khi viết adapter. Phạm vi gồm mapping FSAS/DFFN/encoder/decoder, shape checks, parameter inspection và checkpoint compatibility. Không thay default architecture hoặc load non-strict để che mismatch.

Boundary: cung cấp C-01 cho WS3/WS4, nhận tensor theo C-02. Test dự kiến: shape, deterministic metadata, fixture checkpoint compatibility (không có fixture/implementation ở M0).
