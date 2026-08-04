//! Base codec interface.
//!
//! The LCEVC enhancement layer requires the *decoded* base pictures to
//! compute the enhancement residuals. Two implementations are provided:
//!
//! * `RawBase` — the base is "lossless" (the downscaled source is used
//!   directly). Useful for testing the enhancement layer without a base
//!   codec.
//! * `VvcBase` — encodes the base picture with a VVC encoder (vvenc via
//!   ffmpeg's `libvvenc`) and decodes it with a VVC decoder (ffmpeg's
//!   native `vvc` decoder, which is the FFmpeg implementation of the VVC /
//!   H.266 spec; the resulting bitstream is decodable by vvdec).
//!
//! The encoder state is process-local; the `process` module is available in
//! `base::vvc` for subprocess management.

use crate::config::LcevcConfig;
use crate::frame::Picture;

pub mod vvc;
pub mod vvenc_lib;
pub use vvc::VvcStreamer;

/// Global base codec selection (set by the CLI).
pub enum BaseMode {
    /// Lossless base (decoded base == base target).
    Raw,
    /// VVC encode + decode via ffmpeg (vvenc + native vvc decoder).
    Vvc,
}

static BASE_STATE: std::sync::Mutex<BaseState> =
    std::sync::Mutex::new(BaseState { mode: BaseMode::Raw, extra: String::new(), base_out: String::new() });

struct BaseState {
    mode: BaseMode,
    extra: String,
    base_out: String,
}

/// Configure the base codec used by `encode_decode_base`.
/// `extra` may carry extra ffmpeg encoder options; `base_out` is the path to
/// save the VVC base bitstream to (empty to discard).
pub fn set_base_codec(mode: BaseMode, extra: &str, base_out: &str) {
    let mut state = BASE_STATE.lock().unwrap();
    state.mode = mode;
    state.extra = extra.to_string();
    state.base_out = base_out.to_string();
}

/// Encode and decode the base picture, returning the decoded base.
pub fn encode_decode_base(cfg: &LcevcConfig, base_targets: &[crate::frame::Plane]) -> Result<Picture, String> {
    let nplanes = base_targets.len();
    let (w, h) = (base_targets[0].width, base_targets[0].height);
    let mut base = Picture::new(w, h, cfg.chroma);
    for p in 0..nplanes {
        base.planes[p] = base_targets[p].clone();
    }

    let state = BASE_STATE.lock().unwrap();
    match state.mode {
        BaseMode::Raw => Ok(base),
        BaseMode::Vvc => {
            let out = state.base_out.clone();
            let extra = state.extra.clone();
            vvc::encode_decode_vvc(cfg, &base, if out.is_empty() { None } else { Some(&out) }, &extra)
        }
    }
}

/// Encode a sequence (GOP) of base pictures with vvenc in a single
/// invocation so that inter prediction is used between frames, and decode
/// the result. This is far more efficient than per-frame IDR encoding.
pub fn encode_decode_base_gop(
    cfg: &LcevcConfig,
    frames: &[crate::frame::Picture],
    base_out: Option<&str>,
) -> Result<Vec<crate::frame::Picture>, String> {
    vvc::encode_decode_vvc_gop(cfg, frames, base_out)
}

/// Write one frame of a YUV420p picture (luma then chroma planes) to a
/// file; 10-bit pictures are written as little-endian 16-bit samples.
pub fn write_yuv420(frame: &Picture, depth: u8, out: &mut dyn std::io::Write) -> std::io::Result<()> {
    if depth == 8 {
        for plane in &frame.planes {
            let mut bytes = Vec::with_capacity(plane.data.len());
            for &v in &plane.data {
                bytes.push(v as u8);
            }
            out.write_all(&bytes)?;
        }
    } else if depth == 10 {
        for plane in &frame.planes {
            let mut bytes = Vec::with_capacity(plane.data.len() * 2);
            for &v in &plane.data {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            out.write_all(&bytes)?;
        }
    } else {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, "unsupported depth"));
    }
    Ok(())
}
