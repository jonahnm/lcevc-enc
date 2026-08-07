#!/usr/bin/env bash
# transcode_lcevc.sh — transcode any ffmpeg-readable input to LCEVC + VVC.
#
# Pipeline:  ffmpeg (decode + scale to even YUV420) -> yuv4mpegpipe ->
#            lcevc_enc (VVC base @ QP 24 + LCEVC enhancement) -> MP4 mux
#
# usage:
#   transcode_lcevc.sh INPUT [OUTPUT_BASE] [--target-kbps N] [extra lcevc_enc args...]
#
# Produces:
#   OUTPUT_BASE.mp4        dual-track MP4 (VVC base + LCEVC enhancement)
#   OUTPUT_BASE.lcevc      raw LCEVC enhancement stream
#   OUTPUT_BASE.base.266   raw VVC base bitstream
#
# Options:
#   --target-kbps N       rate-control the enhancement toward N kbps
#                         (default: rate-control the total toward 5 Mbps,
#                          7 Mbps at 4K)
#   --keyframe-interval N keyframe (GOP) interval in seconds; the base is
#                         encoded per GOP (one vvenc invocation per group)
#   --gop N               GOP size in frames (overrides --keyframe-interval)
#   --scale WxH           downscale the video before encoding (e.g. 3840x2160
#                         for an 8K source; an 8K encode is ~4x slower)
#   --base-scale N        base downscale factor per dimension: 2 = half-res
#                         (default), 4 = quarter-res, 1 = full-res base
#   --transform 2x2|4x4   residual transform: 4x4 (DDS, 16 layers,
#                         default, faster) or 2x2 (DD, 4 layers, slightly
#                         better quality at low bitrates)
#                         4x4 (DDS, 16 layers). 2x2 is more efficient at
#                         low bitrates.
#   --upsampler MODE [K0,K1,K2,K3]
#                         upsampler: nearest|bilinear|cubic|modified-cubic
#                         |adaptive. "adaptive" signals a custom 4-tap
#                         kernel (spec 8.6.7); the taps must follow, e.g.
#                         --upsampler adaptive -1023,9214,9214,-1023.
#                         Default: adaptive with the Lanczos-2 kernel
#                         {-1023, 9214, 9214, -1023}.
#   --kernel K0,K1,K2,K3  shorthand for --upsampler adaptive K0,K1,K2,K3
#   --rdoq-lambda-div N   RDOQ rate penalty divisor (LCEVC_RDOQ_LAMBDA_DIV);
#                         larger = weaker penalty = more coefficients kept.
#                         Default 4 (tuned for ~7 Mbps 4K).
#   --vvc-preset NAME     vvenc preset for the VVC base (faster|fast|medium|
#                         slow; default faster)
#   --audio               remux the source audio into the output MP4 (needs
#                         the system ffmpeg; uses stream copy, no re-encode)
#   anything else         passed through to lcevc_enc (e.g. --frames N,
#                         --temporal on --temporal-sw-modifier 24 for temporal
#                         prediction, --scaling-l1 2 for a quarter-res base)
#
# The total frame count is probed (ffprobe) so the progress line shows
# N/M with ETA. Temporal prediction is off by default (costs ~50%
# ~20-50% encode time for ~+0.1 dB); enable them by passing
# --temporal on --temporal-sw-modifier 24 or removing --no-rdoq.
#
# The default encode is tuned for ~7 Mbps 4K30: VVC base QP 24 at 1080p,
# 4x4 transform, signalled Lanczos-2 upsampler kernel, L1 residual disabled
# (SW1 = 32767, matching the LTM 8.1 reference operating points in comp.pdf),
# per-TU adaptive residual and RDOQ lambda 4, with difficulty-weighted rate
# control toward the total budget.
#
# Environment:
#   LCEVC_ENC   path to the lcevc_enc binary (default: target/release)
#   FFMPEG      ffmpeg binary (default: ffmpeg from PATH)
#   FFPROBE     ffprobe binary (default: ffprobe from PATH)

set -euo pipefail

LCEVC_ENC="${LCEVC_ENC:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/target/release/lcevc_enc}"
FFMPEG="${FFMPEG:-ffmpeg}"
FFPROBE="${FFPROBE:-ffprobe}"

INPUT="${1:?usage: transcode_lcevc.sh INPUT [OUTPUT_BASE] [--target-kbps N]}"
shift
OUT="${1:-}"
if [ -n "$OUT" ]; then
    case "$OUT" in
        --*) OUT="" ;;
        *) shift ;;
    esac
fi
if [ -z "$OUT" ]; then
    OUT="${INPUT##*/}"
    OUT="${OUT%.*}"
fi
# strip a trailing .mp4 if the user passed the full output name
case "$OUT" in
    *.mp4) OUT="${OUT%.mp4}" ;;
esac

TARGET=""
TOTAL=""
AUDIO=0
KEYFRAME=""
GOP=""
SCALE=""
BASE_SCALE=2
TRANSFORM="4x4"
UPSAMPLER_ARGS=(--upsampler adaptive -1023,9214,9214,-1023)
LAMBDA_DIV=4
PRESET="faster"
FAST=0
ENCODE_ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --target-kbps)
            TARGET="$2"
            shift 2
            ;;
        --total-kbps)
            TOTAL="$2"
            shift 2
            ;;
        --audio)
            AUDIO=1
            shift
            ;;
        --keyframe-interval)
            KEYFRAME="$2"
            shift 2
            ;;
        --gop)
            GOP="$2"
            shift 2
            ;;
        --scale)
            SCALE="$2"
            shift 2
            ;;
        --base-scale)
            BASE_SCALE="$2"
            shift 2
            ;;
        --transform)
            case "$2" in
                2x2|4x4) TRANSFORM="$2"; shift 2 ;;
                *) echo "--transform must be 2x2 or 4x4, got $2" >&2; exit 1 ;;
            esac
            ;;
        --upsampler)
            case "$2" in
                nearest|bilinear|linear|cubic|modified-cubic)
                    UPSAMPLER_ARGS=(--upsampler "$2")
                    shift 2
                    ;;
                adaptive)
                    if [ -z "${3:-}" ]; then
                        echo "--upsampler adaptive needs the 4 kernel taps, e.g. --upsampler adaptive -1023,9214,9214,-1023" >&2
                        exit 1
                    fi
                    UPSAMPLER_ARGS=(--upsampler adaptive "$3")
                    shift 3
                    ;;
                *)
                    echo "--upsampler must be nearest|bilinear|cubic|modified-cubic|adaptive" >&2
                    exit 1
                    ;;
            esac
            ;;
        --kernel)
            UPSAMPLER_ARGS=(--upsampler adaptive "$2")
            shift 2
            ;;
        --rdoq-lambda-div)
            LAMBDA_DIV="$2"
            shift 2
            ;;
        --vvc-preset)
            PRESET="$2"
            shift 2
            ;;
        --fast)
            FAST=1
            shift
            ;;
        *)
            ENCODE_ARGS+=("$1")
            shift
            ;;
    esac
done

case "$BASE_SCALE" in
    1) SCALING_L1=0; SCALING_L2=0 ;;
    2) SCALING_L1=0; SCALING_L2=2 ;;
    4) SCALING_L1=2; SCALING_L2=2 ;;
    *)
        echo "--base-scale must be 1 (full-res), 2 (half-res) or 4 (quarter-res), got $BASE_SCALE" >&2
        exit 1
        ;;
esac

if [ ! -x "$LCEVC_ENC" ]; then
    echo "lcevc_enc not found at $LCEVC_ENC (cargo build --release first)" >&2
    exit 1
fi

# --- detect the source bit depth (10-bit HDR keeps 10 bits; anything else 8) ---
PF="$("$FFPROBE" -v error -select_streams v:0 -show_entries stream=pix_fmt -of default=noprint_wrappers=1:nokey=1 "$INPUT" 2>/dev/null || true)"
BITS="$("$FFPROBE" -v error -select_streams v:0 -show_entries stream=bits_per_raw_sample -of default=noprint_wrappers=1:nokey=1 "$INPUT" 2>/dev/null | head -1 || true)"
DEPTH=8
PIXFMT="yuv420p"
FORMAT="yuv420p"
set_depth() {
    DEPTH=10
    PIXFMT="yuv420p10le"
    FORMAT="yuv420p10le"
}
case "$BITS" in
    ''|0|'N/A'|8) : ;;
    *) if [ "$BITS" -gt 8 ] 2>/dev/null; then set_depth; fi ;;
esac
# Fallback when bits_per_raw_sample is unavailable: match the pixel format
# name (avoid 8-bit NV12/NV16 whose names contain "12"/"16").
if [ "$DEPTH" -eq 8 ]; then
    case "$PF" in
        *p10*|*p12*|*p14*|*p16*|*010*|*012*|*014*|*016*|*210*|*216*|*410*|\
        *10le*|*12le*|*14le*|*16le*|*rgb4[89]*|*rgb64*|*ayuv64*|*gray1[0246]*)
            set_depth ;;
    esac
fi

echo "== detected: pix_fmt=${PF:-unknown}, bits_per_raw_sample=${BITS:-N/A} -> ${DEPTH}-bit pipeline =="

# --- detect the total frame count (for the N/M progress + ETA) ---
TOTAL_FRAMES=""
case "${INPUT##*.}" in
    mp4|mov|m4v)
        TOTAL_FRAMES="$("$FFPROBE" -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of default=noprint_wrappers=1:nokey=1 "$INPUT" 2>/dev/null | head -1 || true)"
        ;;
esac
if [ -z "$TOTAL_FRAMES" ] || [ "$TOTAL_FRAMES" = "N/A" ] || ! [ "$TOTAL_FRAMES" -gt 0 ] 2>/dev/null; then
    DUR="$("$FFPROBE" -v error -select_streams v:0 -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$INPUT" 2>/dev/null | head -1 || true)"
    FPS="$("$FFPROBE" -v error -select_streams v:0 -show_entries stream=avg_frame_rate -of default=noprint_wrappers=1:nokey=1 "$INPUT" 2>/dev/null | head -1 || true)"
    if [ -n "$DUR" ] && [ -n "$FPS" ]; then
        case "$FPS" in
            */*) FN="${FPS%%/*}"; FD="${FPS##*/}"; TOTAL_FRAMES=$(awk -v d="$DUR" -v n="$FN" -v m="$FD" 'BEGIN { printf "%d", d*n/m }') ;;
            *)   TOTAL_FRAMES=$(awk -v d="$DUR" -v f="$FPS" 'BEGIN { printf "%d", d*f }') ;;
        esac
    fi
fi
FRAMES_ARG=()
if [ -n "$TOTAL_FRAMES" ] && [ "$TOTAL_FRAMES" -gt 0 ] 2>/dev/null; then
    FRAMES_ARG+=(--frames "$TOTAL_FRAMES")
fi

VFILTER="scale=trunc(iw/2)*2:trunc(ih/2)*2:flags=lanczos,format=${FORMAT}"
if [ -n "$SCALE" ]; then
    VFILTER="scale=${SCALE}:flags=lanczos,format=${FORMAT}"
fi

# Default total-bitrate budget: ~0.84 bpp scaled so 4K caps at 7 Mbps
# TOTAL (VVC base + enhancement); the encoder measures the base's actual
# bitrate per GOP and gives the rest to the enhancement. Pass
# --total-kbps to override, or --target-kbps for an enhancement-only
# target.
if [ -z "$TOTAL" ] && [ -z "$TARGET" ]; then
    DW="$("$FFPROBE" -v error -select_streams v:0 -show_entries stream=width -of default=noprint_wrappers=1:nokey=1 "$INPUT" 2>/dev/null | head -1)"
    DH="$("$FFPROBE" -v error -select_streams v:0 -show_entries stream=height -of default=noprint_wrappers=1:nokey=1 "$INPUT" 2>/dev/null | head -1)"
    if [ -n "$DW" ] && [ -n "$DH" ] && [ "$DW" -gt 0 ] 2>/dev/null && [ "$DH" -gt 0 ] 2>/dev/null; then
        TOTAL=$(awk -v w="$DW" -v h="$DH" 'BEGIN { t = w*h/8294400*7000; if (t < 800) t = 800; if (t > 30000) t = 30000; printf "%d", t }')
    fi
fi

# Carry the source colour metadata (HDR) into the base VUI and the
# output MP4's colr box.
COLOR_ARGS=()
CP="$("$FFPROBE" -v error -select_streams v:0 -show_entries stream=color_primaries -of default=noprint_wrappers=1:nokey=1 "$INPUT" 2>/dev/null | head -1)"
CT="$("$FFPROBE" -v error -select_streams v:0 -show_entries stream=color_transfer -of default=noprint_wrappers=1:nokey=1 "$INPUT" 2>/dev/null | head -1)"
CS="$("$FFPROBE" -v error -select_streams v:0 -show_entries stream=colorspace -of default=noprint_wrappers=1:nokey=1 "$INPUT" 2>/dev/null | head -1)"
if [ -n "$CP$CT$CS" ]; then
    CP="${CP:-bt709}"; CT="${CT:-bt709}"; CS="${CS:-bt709}"
    [ "$CP" = "unknown" ] && CP="bt709"
    [ "$CT" = "unknown" ] && CT="bt709"
    [ "$CS" = "unknown" ] && CS="bt709"
    COLOR_ARGS+=(--color "$CP:$CT:$CS")
fi

GOP_ARGS=()
if [ -n "$GOP" ]; then
    GOP_ARGS+=(--base-gop "$GOP")
elif [ -n "$KEYFRAME" ]; then
    GOP_ARGS+=(--base-gop-seconds "$KEYFRAME")
fi

echo "== transcode $INPUT -> ${OUT}.mp4 (base QP 24, ${DEPTH}-bit, ${BASE_SCALE}x base downscale, ${TOTAL_FRAMES:-?} frames, transform ${TRANSFORM}) =="

TARGET_ARGS=()
if [ -n "$TOTAL" ]; then
    TARGET_ARGS+=(--total-kbps "$TOTAL")
elif [ -n "$TARGET" ]; then
    TARGET_ARGS+=(--target-kbps "$TARGET")
fi

set -o pipefail
FAST_ENV=()
FAST_ARGS=()
if [ "$FAST" -eq 1 ]; then
    FAST_ENV=(LCEVC_FAST=1)
    FAST_ARGS=(--no-rdoq)
fi
"$FFMPEG" -hide_banner -loglevel error \
    -i "$INPUT" \
    -map 0:v:0 -an \
    -vf "$VFILTER" \
    -fps_mode cfr \
    -f yuv4mpegpipe -strict -1 -pix_fmt "$PIXFMT" - |
    env LCEVC_RDOQ_LAMBDA_DIV="$LAMBDA_DIV" "${FAST_ENV[@]}" "$LCEVC_ENC" -i - --input-format y4m --bit-depth "$DEPTH" \
        --base-mode vvc \
        --vvc-qp 24 --vvc-preset "$PRESET" \
        --base-gop 30 \
        --scaling-l1 $SCALING_L1 --scaling-l2 $SCALING_L2 \
        --transform "$TRANSFORM" \
        "${UPSAMPLER_ARGS[@]}" \
        --qm-beta 0.3 \
        --step-width-l1 32767 --step-width-l2 1000 \
        --no-psnr \
        "${FRAMES_ARG[@]}" \
        "${GOP_ARGS[@]}" \
        "${TARGET_ARGS[@]}" \
        "${COLOR_ARGS[@]}" \
        "${FAST_ARGS[@]}" \
        --base-out "${OUT}.base.266" -o "${OUT}.lcevc" \
        --mux "${OUT}.mp4" \
        "${ENCODE_ARGS[@]}"

if [ "$AUDIO" -eq 1 ]; then
    if "$FFMPEG" -hide_banner -loglevel error -i "${OUT}.mp4" -i "$INPUT" \
        -map 0:v -map 1:a? -c copy -y "${OUT}.audio.mp4" 2>/dev/null; then
        mv "${OUT}.audio.mp4" "${OUT}.mp4"
        echo "== audio copied into ${OUT}.mp4 =="
    else
        echo "!! audio remux failed, keeping video-only ${OUT}.mp4" >&2
        rm -f "${OUT}.audio.mp4"
    fi
fi

echo "== done: ${OUT}.mp4 (${OUT}.lcevc + ${OUT}.base.266) =="
