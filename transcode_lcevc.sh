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
#                         (default: fixed step widths 1024/256)
#   --keyframe-interval N keyframe (GOP) interval in seconds; the base is
#                         encoded per GOP (one vvenc invocation per group)
#   --scale WxH           downscale the video before encoding (e.g. 3840x2160
#                         for an 8K source; an 8K encode is ~4x slower)
#   --audio               remux the source audio into the output MP4 (needs
#                         the system ffmpeg; uses stream copy, no re-encode)
#   anything else         passed through to lcevc_enc (e.g. --frames N,
#                         --temporal on --temporal-sw-modifier 24 for temporal
#                         prediction, --scaling-l1 2 for a quarter-res base)
#
# The total frame count is probed (ffprobe) so the progress line shows
# N/M with ETA. Temporal prediction is off by default; it costs ~50% encode
# time for ~+0.1 dB and can be enabled by passing
# --temporal on --temporal-sw-modifier 24 as extra args.
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

TARGET=""
AUDIO=0
KEYFRAME=""
SCALE=""
ENCODE_ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --target-kbps)
            TARGET="$2"
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
        --scale)
            SCALE="$2"
            shift 2
            ;;
        *)
            ENCODE_ARGS+=("$1")
            shift
            ;;
    esac
done

if [ ! -x "$LCEVC_ENC" ]; then
    echo "lcevc_enc not found at $LCEVC_ENC (cargo build --release first)" >&2
    exit 1
fi

# --- detect the source bit depth (10-bit HDR keeps 10 bits; anything else 8) ---
PF="$("$FFPROBE" -v error -select_streams v:0 -show_entries stream=pix_fmt -of csv=p=0 "$INPUT" 2>/dev/null || true)"
case "$PF" in
    *10*|*12*|*16*) DEPTH=10; PIXFMT="yuv420p10le"; FORMAT="yuv420p10le" ;;
    *)              DEPTH=8;  PIXFMT="yuv420p";     FORMAT="yuv420p" ;;
esac

# --- detect the total frame count (for the N/M progress + ETA) ---
TOTAL_FRAMES="$("$FFPROBE" -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of csv=p=0 "$INPUT" 2>/dev/null | head -1 || true)"
if [ -z "$TOTAL_FRAMES" ] || [ "$TOTAL_FRAMES" = "N/A" ] || ! [ "$TOTAL_FRAMES" -gt 0 ] 2>/dev/null; then
    DUR="$("$FFPROBE" -v error -select_streams v:0 -show_entries format=duration -of csv=p=0 "$INPUT" 2>/dev/null | head -1 || true)"
    FPS="$("$FFPROBE" -v error -select_streams v:0 -show_entries stream=avg_frame_rate -of csv=p=0 "$INPUT" 2>/dev/null | head -1 || true)"
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

GOP_ARGS=()
[ -n "$KEYFRAME" ] && GOP_ARGS+=(--base-gop-seconds "$KEYFRAME")

echo "== transcode $INPUT -> ${OUT}.mp4 (base QP 24, ${DEPTH}-bit, half-res pyramid, ${TOTAL_FRAMES:-?} frames) =="

TARGET_ARGS=()
[ -n "$TARGET" ] && TARGET_ARGS+=(--target-kbps "$TARGET")

set -o pipefail
"$FFMPEG" -hide_banner -loglevel error \
    -i "$INPUT" \
    -map 0:v:0 -an \
    -vf "$VFILTER" \
    -fps_mode cfr \
    -f yuv4mpegpipe -strict -1 -pix_fmt "$PIXFMT" - |
    "$LCEVC_ENC" -i - --input-format y4m --bit-depth "$DEPTH" \
        --base-mode vvc \
        --vvc-qp 24 --vvc-preset faster \
        --base-gop 30 \
        --scaling-l1 0 --scaling-l2 2 \
        --upsampler modified-cubic \
        --qm-beta 0.3 \
        --step-width-l1 1024 --step-width-l2 256 \
        --no-psnr \
        "${FRAMES_ARG[@]}" \
        "${GOP_ARGS[@]}" \
        "${TARGET_ARGS[@]}" \
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
