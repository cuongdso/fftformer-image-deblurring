# WS2 fixtures

These 2x2 PNGs were generated locally from explicit pixel values using Python's
standard-library `struct` and `zlib`; they contain no external dataset content.

- `rgb.png`: red/green on the top row, blue/white on the bottom row, RGB uint8.
- `grayscale.png`: grayscale values 0/128 and 192/255; intentionally invalid for C-02.
- `rgba.png`: RGBA with alpha values 255/0 and 128/255; intentionally invalid for C-02.

The fixtures are 224 bytes total. Tests copy them into temporary directories and
create additional images there. They are unit-test fixtures, not GoPro/RealBlur
samples or evidence of model quality. Larger development samples in `data/` stay
outside Git.
