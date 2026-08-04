@echo off
setlocal enabledelayedexpansion
rem transcode_lcevc.bat - transcode any ffmpeg-readable input to LCEVC + VVC.
rem
rem Pipeline:  ffmpeg (decode + scale to even YUV420) -> yuv4mpegpipe ->
rem            lcevc_enc (VVC base @ QP 24 + LCEVC enhancement) -> MP4 mux
rem
rem usage:
rem   transcode_lcevc.bat INPUT [OUTPUT_BASE] [--target-kbps N] [--audio] [extra lcevc_enc args...]
rem
rem Produces:
rem   OUTPUT_BASE.mp4        dual-track MP4 (VVC base + LCEVC enhancement)
rem   OUTPUT_BASE.lcevc      raw LCEVC enhancement stream
rem   OUTPUT_BASE.base.266   raw VVC base bitstream
rem
rem Options:
rem   --target-kbps N       rate-control the enhancement toward N kbps
rem   --keyframe-interval N keyframe (GOP) interval in seconds
rem   --scale WxH           downscale the video before encoding
rem   --audio               remux the source audio into the output MP4
rem                         (stream copy, no re-encode)
rem   anything else      passed through to lcevc_enc (e.g. --frames N,
rem                      --temporal on --temporal-sw-modifier 24 for temporal prediction)
rem
rem Environment:
rem   LCEVC_ENC   path to the lcevc_enc.exe binary
rem   FFMPEG      ffmpeg binary
rem   FFPROBE     ffprobe binary

if defined LCEVC_ENC set "LCEVC_ENC=%LCEVC_ENC%"
if not defined LCEVC_ENC set "LCEVC_ENC=%~dp0target\release\lcevc_enc.exe"
if not exist "%LCEVC_ENC%" if exist "%~dp0target\release\lcevc_enc" set "LCEVC_ENC=%~dp0target\release\lcevc_enc"
if not defined FFMPEG set "FFMPEG=ffmpeg"
if not defined FFPROBE set "FFPROBE=ffprobe"

set "INPUT=%~1"
if "%INPUT%"=="" goto usage
shift

set "OUT="
if not "%~1"=="" (
    set "A1=%~1"
    if not "!A1:~0,2!"=="--" (
        set "OUT=%~1"
        shift
    )
)
if not defined OUT set "OUT=%~n1"

set "TARGET="
set "AUDIO=0"
set "KEYFRAME="
set "SCALE="
set "EXTRA="
:argloop
if "%~1"=="" goto doneargs
if "%~1"=="--target-kbps" (
    set "TARGET=%~2"
    shift
    shift
    goto argloop
)
if "%~1"=="--audio" (
    set "AUDIO=1"
    shift
    goto argloop
)
if "%~1"=="--keyframe-interval" (
    set "KEYFRAME=%~2"
    shift
    shift
    goto argloop
)
if "%~1"=="--scale" (
    set "SCALE=%~2"
    shift
    shift
    goto argloop
)
set "EXTRA=!EXTRA! %~1"
shift
goto argloop
:doneargs

if not exist "%LCEVC_ENC%" (
    echo lcevc_enc not found at %LCEVC_ENC% ^(cargo build --release first^) 1>&2
    exit /b 1
)

rem --- detect the source bit depth (10-bit HDR keeps 10 bits; anything else 8) ---
set "DEPTH=8"
set "PIXFMT=yuv420p"
set "FORMAT=yuv420p"
set "PF="
set "BITS="
rem Use temp files instead of for /f: cmd's for /f parsing breaks on
rem filenames containing parentheses/brackets.
"%FFPROBE%" -v error -select_streams v:0 -show_entries stream=pix_fmt -of csv=p=0 "%INPUT%" > "%TEMP%\lcevc_pf.txt" 2>nul
if exist "%TEMP%\lcevc_pf.txt" set /p PF=<"%TEMP%\lcevc_pf.txt"
"%FFPROBE%" -v error -select_streams v:0 -show_entries stream=bits_per_raw_sample -of csv=p=0 "%INPUT%" > "%TEMP%\lcevc_bits.txt" 2>nul
if exist "%TEMP%\lcevc_bits.txt" set /p BITS=<"%TEMP%\lcevc_bits.txt"
del /q "%TEMP%\lcevc_pf.txt" "%TEMP%\lcevc_bits.txt" 2>nul
echo !BITS! | findstr /r "10 12 14 16" >nul && (
    set "DEPTH=10"
    set "PIXFMT=yuv420p10le"
    set "FORMAT=yuv420p10le"
)
if "!DEPTH!"=="8" (
    rem Fallback when bits_per_raw_sample is unavailable: match the pixel
    rem format name (avoid 8-bit NV12/NV16 whose names contain "12"/"16").
    echo %PF% | findstr /r "p10 p12 p14 p16 010 012 014 016 210 216 410 10le 12le 14le 16le" >nul && (
        set "DEPTH=10"
        set "PIXFMT=yuv420p10le"
        set "FORMAT=yuv420p10le"
    )
)

set "VFILTER=scale=trunc(iw/2)*2:trunc(ih/2)*2:flags=lanczos,format=!FORMAT!"
if defined SCALE set "VFILTER=scale=!SCALE!:flags=lanczos,format=!FORMAT!"

echo == detected: pix_fmt=%PF%, bits_per_raw_sample=%BITS% -^> !DEPTH!-bit pipeline
echo == transcode %INPUT% -^> %OUT%.mp4 ^(base QP 24, !DEPTH!-bit, half-res pyramid^)

set "TARGET_ARGS="
if defined TARGET set "TARGET_ARGS=--target-kbps !TARGET!"
set "GOP_ARGS="
if defined KEYFRAME set "GOP_ARGS=--base-gop-seconds !KEYFRAME!"

"%FFMPEG%" -hide_banner -loglevel error -i "%INPUT%" -map 0:v:0 -an -vf "!VFILTER!" -fps_mode cfr -f yuv4mpegpipe -strict -1 -pix_fmt !PIXFMT! - | "%LCEVC_ENC%" -i - --input-format y4m --bit-depth !DEPTH! --base-mode vvc --vvc-qp 24 --vvc-preset faster --base-gop 30 --scaling-l1 0 --scaling-l2 2 --upsampler modified-cubic --qm-beta 0.3 --step-width-l1 1024 --step-width-l2 256 --no-psnr --no-rdoq !GOP_ARGS! !TARGET_ARGS! --base-out "%OUT%.base.266" -o "%OUT%.lcevc" --mux "%OUT%.mp4"!EXTRA!
if errorlevel 1 (
    echo !! encode failed 1>&2
    exit /b 1
)

if "!AUDIO!"=="1" (
    "%FFMPEG%" -hide_banner -loglevel error -i "%OUT%.mp4" -i "%INPUT%" -map 0:v -map 1:a? -c copy -y "%OUT%.audio.mp4" 2>nul
    if not errorlevel 1 (
        move /y "%OUT%.audio.mp4" "%OUT%.mp4" >nul
        echo == audio copied into %OUT%.mp4
    ) else (
        echo !! audio remux failed, keeping video-only %OUT%.mp4 1>&2
        del /q "%OUT%.audio.mp4" 2>nul
    )
)

echo == done: %OUT%.mp4 ^(%OUT%.lcevc + %OUT%.base.266^)
exit /b 0

:usage
echo usage: transcode_lcevc.bat INPUT [OUTPUT_BASE] [--target-kbps N] [--audio] [extra args...]
exit /b 2
