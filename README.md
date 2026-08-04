# lcevc-enc

An LCEVC (Low Complexity Enhancement Video Coding, ISO/IEC 23094-2 / MPEG-5
Part 2) enhancement-layer encoder written in Rust, designed to work with a
VVC (H.266) base codec (decodable with vvdec).

The encoder produces enhancement-layer NAL units in the bitstream format
decoded by the reference decoder (`lcevcdec` from
[v-novaltd/LCEVCdec](https://github.com/v-novaltd/LCEVCdec), bitstream
version "AlignWithSpec"). Where the DIS draft `spec.pdf` and the reference
decoder disagree, the decoder is authoritative.

## Verification

Every configuration the encoder supports has been verified end-to-end
against the **actual reference decoder** (lcevcdec 4.2.0,
`liblcevc_dec_api.so`, via the C harness in `verify/`): the decoder output is
**bit-exact** with the encoder's reconstruction (PSNR = inf, max diff = 0) for:

- scaling modes: 2D/2D, 1D/2D, 0D/2D, 2D/1D, 0D/1D, 1D/1D, 0D/0D
- transform types: 4x4 (DDS, 16 layers) and 2x2 (DD, 4 layers)
- upsamplers: nearest, bilinear, cubic, modified-cubic (with and without
  predicted average)
- temporal prediction (multi-frame, inter/intra per TU with the temporal
  buffer)
- tiled pictures (512x256, 1024x512) with all three tile-size compression
  modes (none, prefix, prefix-on-diff)
- temporal + tiled combined
- VVC base (encoded with vvenc via ffmpeg, decoded with the native VVC
  decoder; the `base.266` output is decodable with vvdec)

## Building

```sh
cargo build --release
```

## Usage

```sh
# Raw base (lossless base for testing the enhancement layer):
lcevc_enc -i input.yuv -s 1280x720 -f 30 --frames 10 \
          --base-mode raw -o out.lcevc

# VVC base (encodes the base with libvvenc, decodes it back, and saves the
# VVC bitstream):
lcevc_enc -i input.yuv -s 1280x720 -f 30 --frames 10 \
          --base-mode vvc --base-out base.266 -o out.lcevc

# Diagnostic dumps (base pictures and the encoder reconstruction, yuv420p):
lcevc_enc -i input.yuv -s 1280x720 -f 30 --frames 1 -o out.lcevc \
          --dump-base --dump-recon
```

`input.yuv` is yuv420p (luma then chroma planes), `WxH` must satisfy the
decoder's alignment rules (multiples of 16 for 4:2:0 with the 4x4 transform
and 2D scaling).

Options include `--step-width-l1`, `--step-width-l2` (quantizer step widths
in s8.7 units; smaller = better quality), `--transform 2x2|4x4`,
`--upsampler nearest|bilinear|cubic|modified-cubic`,
`--scaling-l1 0|1|2`, `--scaling-l2 0|1|2`, `--predicted-average on|off`,
`--temporal on|off`, `--tiles none|512x256|1024x512|WxH` and
`--tile-size-compression 0|1|2`.

## Self-check

`selfcheck` encodes two frames, decodes the second frame's NAL unit with the
built-in bit-exact mirror of the reference decoder (in `src/decoder.rs`) and
compares:

```sh
cargo run --release --bin selfcheck -- input.yuv 512 128 2 2 0
```

## Verification harness

`verify/lcevc_verify.c` links against the reference decoder
(`liblcevc_dec_api.so.4`) and decodes the enhancement stream together with
the dumped base pictures, comparing against the encoder's reconstruction:

```sh
gcc -o verify/lcevc_verify verify/lcevc_verify.c \
    -I <lcevcdec>/include -I <lcevcdec>/src/api/include \
    -L <lcevcdec-lib> -llcevc_dec_api -lm
LD_LIBRARY_PATH=<lcevcdec-lib> ./verify/lcevc_verify out.lcevc \
    base_dump.yuv recon_dump.yuv <baseW> <baseH> <W> <H>
```

## Bitstream summary

Per access unit: one NAL unit (`00 00 01`, header byte `0x7B` for IDR /
`0x79` otherwise, second byte `0xFF`) containing process blocks, with
emulation prevention and an `0x80` RBSP stop byte. Blocks:

1. **sequence_config** (first IDR): profile, level, sublevel, conformance
   window.
2. **global_config** (IDR): chroma, bit depths, transform, upsampler,
   scaling modes, tiling, temporal settings, resolution.
3. **picture_config** (every frame): no-enhancement bit, quant-matrix mode,
   step widths (15 bits each), dither control, temporal refresh.
4. **encoded_data** (or **encoded_data_tiled**): per-chunk flags
   (entropy-enabled, rle-only), chunk data with Huffman tables
   (LSB/MSB/zero-run contexts), RLE coefficient streams, and optional
   per-tile size compression and temporal signal chunks.

The entropy coding uses the reference decoder's "AlignWithSpec" format:
canonical Huffman tables signalled by code lengths, RLE value bytes with
`run`/`msb` flag bits, zero-run and temporal run counts as MSB-first 7-bit
chunks. The reconstruction mirrors the decoder's S16 fixed-point pipeline
(samples as `(v - 128) * 128`), dequant with the U12.4 fixed-point
logarithm, the exact inverse/forward Hadamard-based transforms, and the
2x2-tap kernel upsamplers.
