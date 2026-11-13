# ADR-0001: Pin official baseline bằng Git submodule

- **Status:** Accepted for M0
- **Date:** 2026-11-13
- **Decision:** `third_party/FFTformer/` là Git submodule URL `https://github.com/kkkls/FFTformer.git`, pin tại `01306b5d4072146a011c8c6e9d4927cf4607f1d5`.

## Context

Nhóm cần đối chiếu được với implementation chính thức nhưng không được sửa baseline tùy tiện hoặc copy mã không có traceability. Upstream công bố MIT License tại commit đã pin.

## Consequences

- Clone cần `--recurse-submodules` hoặc `git submodule update --init --recursive`.
- Superproject chỉ version submodule pointer; không vendor/copy source upstream vào `src/`.
- Thay đổi pin phải được PR review, cập nhật `docs/upstream-audit.md`, reproduction docs và experiment metadata.
- Khi tái sử dụng source upstream, giữ MIT copyright/license. Dataset và checkpoint có điều khoản riêng, hiện `UNVERIFIED` trừ khi nguồn của chúng được audit riêng.

## Alternatives

Vendor snapshot bị loại ở M0 vì tăng nguy cơ drift và phải đồng bộ history/license thủ công. Không thấy trở ngại kỹ thuật nghiêm trọng nào của submodule trong khảo sát M0.
