# Phương pháp ghi nhận thí nghiệm

Mỗi kết quả nhóm phải liên kết group commit, upstream SHA, checkpoint identity, dataset/split/license, config, environment/hardware, seed (nếu có), command, protocol, artifacts ngoài Git và failure cases. Template ở `experiments/README.md`.

Phân loại bắt buộc:

- **Paper-reported:** chỉ trích nguồn, không nhận là nhóm đo.
- **Group-measured:** command/evidence và protocol có thể truy vết.
- **UNVERIFIED:** không dùng để so sánh/kết luận.

Thay đổi architecture, data protocol hoặc evaluation protocol phải được ghi thành experiment/task riêng và không ghi đè baseline.
