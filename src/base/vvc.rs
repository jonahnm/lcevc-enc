//! VVC base codec integration via ffmpeg subprocesses.
//!
//! Base encode: ffmpeg with `libvvenc` (the Fraunhofer VVC encoder).
//! Base decode: ffmpeg with the native `vvc` decoder (bitstream-compatible
//! with vvdec; the VVC bitstream produced can be decoded with vvdecapp).

use crate::config::LcevcConfig;
use crate::frame::Picture;
use std::process::{Command, Stdio};

/// Encode a sequence (GOP) of base pictures with vvenc in a single
/// invocation (inter prediction between frames) and decode the result.
pub fn encode_decode_vvc_gop(
    cfg: &LcevcConfig,
    frames: &[crate::frame::Picture],
    base_out: Option<&str>,
) -> Result<Vec<crate::frame::Picture>, String> {
    if frames.is_empty() {
        return Ok(Vec::new());
    }
    let depth = cfg.enhancement_depth;
    let width = frames[0].width as u32;
    let height = frames[0].height as u32;
    if width % 2 != 0 || height % 2 != 0 {
        return Err(format!("VVC base requires even dimensions, got {width}x{height}"));
    }
    for f in frames {
        if f.width != width as usize || f.height != height as usize {
            return Err("base GOP frames have inconsistent dimensions".into());
        }
    }

    let mut preset = "faster".to_string();
    let mut qp = "30".to_string();
    {
        let state = crate::base::BASE_STATE.lock().unwrap();
        let extra = state.extra.clone();
        drop(state);
        for opt in extra.split(',') {
            if let Some(v) = opt.strip_prefix("preset=") {
                preset = v.to_string();
            } else if let Some(v) = opt.strip_prefix("qp=") {
                qp = v.to_string();
            }
        }
    }
    // libvvenc only accepts yuv420p10le input, so the base codec path always
    // runs at 10-bit: 8-bit pipeline samples (0-255) pass through as raw u16
    // values unchanged. NOTE: ffmpeg's -vvenc-params is a DICT option: the
    // pairs are COLON separated with case-sensitive lowercase keys;
    // mtprofile=3 enables wavefront (WPP) + tiles (2 columns x 1 row,
    // resolution dependent; degrades to WPP-only for small bases).
    let pix_fmt = "yuv420p10le";
    let ncpu = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    let run_ffmpeg = |params: &str, input: &[u8], log: &mut String| -> Result<Option<Vec<u8>>, String> {
        let mut child = std::process::Command::new("ffmpeg")
            .args([
                "-hide_banner", "-loglevel", "error", "-y",
                "-f", "rawvideo", "-pix_fmt", pix_fmt,
                "-s", &format!("{width}x{height}"),
                "-r", "25",
                "-i", "-",
                "-c:v", "libvvenc", "-preset", &preset, "-qp", &qp,
                "-vvenc-params", params,
                "-f", "vvc", "pipe:1",
            ])
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .map_err(|e| format!("failed to run ffmpeg (libvvenc gop): {e}"))?;
        // Write the YUV on a thread so the encode output pipe cannot deadlock.
        let mut stdin = child.stdin.take().unwrap();
        let input_owned = input.to_vec();
        let writer = std::thread::spawn(move || {
            use std::io::Write;
            let _ = stdin.write_all(&input_owned);
        });
        let output = child
            .wait_with_output()
            .map_err(|e| format!("ffmpeg libvvenc gop failed: {e}"))?;
        writer.join().unwrap();
        if !output.status.success() {
            *log = String::from_utf8_lossy(&output.stderr).into_owned();
            Ok(None)
        } else {
            Ok(Some(output.stdout))
        }
    };
    // mtprofile=3 enables wavefront (WPP) + tiles; older libvvenc builds
    // reject the key, so retry with the basic params in that case.
    let mut yuv = Vec::with_capacity(frames.len() * (width as usize * height as usize * 3 / 2) * 2);
    for frame in frames {
        crate::base::write_yuv420(frame, 10, &mut yuv).map_err(|e| e.to_string())?;
    }
    let mut ffmpeg_log = String::new();
    let mtp_params = format!(
        "internalbitdepth=10:inputbitdepth=10:threads={}:mtprofile=3",
        ncpu,
    );
    let base_params = format!("internalbitdepth=10:inputbitdepth=10:threads={}", ncpu);
    let vvc = run_ffmpeg(&mtp_params, &yuv, &mut ffmpeg_log)?
        .or_else(|| run_ffmpeg(&base_params, &yuv, &mut ffmpeg_log).unwrap_or(None))
        .ok_or_else(|| format!("ffmpeg libvvenc gop encode failed:\n{ffmpeg_log}"))?;

    if let Some(path) = base_out {
        use std::io::Write;
        let mut out = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .map_err(|e| format!("failed to open base output: {e}"))?;
        out.write_all(&vvc).map_err(|e| format!("failed to write base output: {e}"))?;
    }

    // Decode the VVC bitstream back to YUV (piped, no temp files).
    let mut child = std::process::Command::new("ffmpeg")
        .args([
            "-hide_banner", "-loglevel", "error", "-y",
            "-i", "-",
            "-f", "rawvideo", "-pix_fmt", pix_fmt,
            "pipe:1",
        ])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| format!("failed to run ffmpeg (vvc gop decode): {e}"))?;
    let data = {
        use std::io::Write;
        let mut stdin = child.stdin.take().unwrap();
        let vvc2 = vvc.clone();
        let writer = std::thread::spawn(move || {
            let _ = stdin.write_all(&vvc2);
        });
        let output = child
            .wait_with_output()
            .map_err(|e| format!("ffmpeg vvc gop decode failed: {e}"))?;
        writer.join().unwrap();
        if !output.status.success() {
            return Err(format!(
                "ffmpeg vvc gop decode failed:\n{}",
                String::from_utf8_lossy(&output.stderr)
            ));
        }
        output.stdout
    };
    // The decode always outputs 10-bit raw (2 bytes per sample).
    let bytes_per = 2usize;
    let n = width as usize * height as usize;
    let frame_bytes = (n + 2 * ((width as usize / 2) * (height as usize / 2))) * bytes_per;
    let mut out = Vec::with_capacity(frames.len());
    for (fi, _) in frames.iter().enumerate() {
        let chunk = &data[fi * frame_bytes..(fi + 1) * frame_bytes];
        let mut pic = crate::frame::Picture::new(width as usize, height as usize, cfg.chroma);
        let mut off = 0usize;
        for plane in &mut pic.planes {
            let n = plane.data.len();
            for (dst, pair) in plane
                .data
                .iter_mut()
                .zip(chunk[off..off + n * 2].chunks_exact(2))
            {
                *dst = u16::from_le_bytes([pair[0], pair[1]]);
            }
            off += n * bytes_per;
        }
        out.push(pic);
    }
    Ok(out)
}

/// Encode the base picture with vvenc and decode it back with the VVC
/// decoder, using temporary files. `extra` may carry extra ffmpeg encoder
/// options (e.g. "-qp 30").
pub fn encode_decode_vvc(cfg: &LcevcConfig, base: &Picture, base_out: Option<&str>, extra: &str) -> Result<Picture, String> {
    let tmp_dir = std::env::temp_dir();
    let depth = cfg.enhancement_depth;
    let base_yuv = tmp_dir.join(format!("lcevc_enc_base{depth}.yuv"));
    let base_vvc = tmp_dir.join(format!("lcevc_enc_base{depth}.266"));
    let decoded_yuv = tmp_dir.join(format!("lcevc_enc_base{depth}_dec.yuv"));

    let width = base.width as u32;
    let height = base.height as u32;
    if width % 2 != 0 || height % 2 != 0 {
        return Err(format!("VVC base requires even dimensions, got {width}x{height}"));
    }

    // Write the base picture (8- or 10-bit).
    {
        let mut f = std::fs::File::create(&base_yuv).map_err(|e| e.to_string())?;
        crate::base::write_yuv420(base, 10, &mut f).map_err(|e| e.to_string())?;
    }

    // Encode with libvvenc. `extra` carries vvenc options like
    // "preset=fast,qp=27" (defaults: preset medium, qp 30).
    let mut preset = "faster".to_string();
    let mut qp = "30".to_string();
    for opt in extra.split(',') {
        if let Some(v) = opt.strip_prefix("preset=") {
            preset = v.to_string();
        } else if let Some(v) = opt.strip_prefix("qp=") {
            qp = v.to_string();
        }
    }
    let pix_fmt = "yuv420p10le";
    let ncpu = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    let run_ffmpeg = |params: &str, log: &mut String| -> Result<bool, String> {
        let mut cmd = Command::new("ffmpeg");
        cmd.args([
                "-hide_banner", "-loglevel", "error", "-y",
                "-f", "rawvideo", "-pix_fmt", pix_fmt,
                "-s", &format!("{width}x{height}"),
                "-i", base_yuv.to_str().unwrap(),
                "-c:v", "libvvenc", "-preset", &preset, "-qp", &qp,
                "-vvenc-params", params,
            ]);
        if let Some(c) = cfg.colour.as_ref() {
            cmd.args([
                "-color_primaries", &c.primaries_name,
                "-color_trc", &c.transfer_name,
                "-colorspace", &c.matrix_name,
            ]);
        }
        cmd.arg(base_vvc.to_str().unwrap());
        let output = cmd
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .output()
            .map_err(|e| format!("failed to run ffmpeg (libvvenc): {e}"))?;
        if !output.status.success() {
            *log = String::from_utf8_lossy(&output.stderr).into_owned();
            Ok(false)
        } else {
            Ok(true)
        }
    };
    let mut ffmpeg_log = String::new();
    let mtp_params = format!(
        "internalbitdepth=10:inputbitdepth=10:threads={}:mtprofile=3",
        ncpu,
    );
    let base_params = format!("internalbitdepth=10:inputbitdepth=10:threads={}", ncpu);
    let ok = run_ffmpeg(&mtp_params, &mut ffmpeg_log)?
        || (run_ffmpeg(&base_params, &mut ffmpeg_log)?);
    if !ok {
        return Err(format!("ffmpeg libvvenc encode failed:\n{ffmpeg_log}"));
    }
    if let Some(path) = base_out {
        // Append this frame's bitstream to the output file (each call
        // encodes a single frame).
        use std::io::Write;
        let mut out = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .map_err(|e| format!("failed to open base output: {e}"))?;
        let data = std::fs::read(&base_vvc).map_err(|e| e.to_string())?;
        out.write_all(&data).map_err(|e| format!("failed to write base output: {e}"))?;
    }

    // Decode with the native VVC decoder.
    let status = Command::new("ffmpeg")
        .args([
            "-hide_banner", "-loglevel", "error", "-y",
            "-i", base_vvc.to_str().unwrap(),
            "-f", "rawvideo", "-pix_fmt", pix_fmt,
            decoded_yuv.to_str().unwrap(),
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map_err(|e| format!("failed to run ffmpeg (vvc decode): {e}"))?;
    if !status.success() {
        return Err("ffmpeg vvc decode failed".into());
    }

    // Read the decoded base picture.
    let data = std::fs::read(&decoded_yuv).map_err(|e| e.to_string())?;
    let bytes_per = if depth == 10 { 2usize } else { 1usize };
    let plane_size = width as usize * height as usize;
    let expected = (plane_size + 2 * ((width as usize / 2) * (height as usize / 2))) * bytes_per;
    if data.len() < expected {
        return Err(format!("decoded base too small: {} < {}", data.len(), expected));
    }
    let mut decoded = Picture::new(width as usize, height as usize, cfg.chroma);
    let mut off = 0;
    for plane in &mut decoded.planes {
        let n = plane.data.len();
        match depth {
            8 => {
                for (dst, &b) in plane.data.iter_mut().zip(&data[off..off + n]) {
                    *dst = b as u16;
                }
            }
            10 => {
                for (dst, chunk) in plane
                    .data
                    .iter_mut()
                    .zip(data[off..off + n * 2].chunks_exact(2))
                {
                    *dst = u16::from_le_bytes([chunk[0], chunk[1]]);
                }
            }
            other => return Err(format!("unsupported depth {other}")),
        }
        off += n * bytes_per;
    }

    let _ = &mut std::fs::File::create(&base_yuv);
    Ok(decoded)
}
