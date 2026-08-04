//! verify_full — decode a full LCEVC segment with the built-in mirror decoder
//! and compare frame-by-frame against the enhanced video arriving on stdin
//! (typically piped from `ffmpeg -i out.mp4 -f rawvideo -pix_fmt yuv420p10le -`).
//!
//! Usage:
//!   verify_full <enh.lcevc> <base.yuv> <W> <H> <baseW> <baseH> <fps>
//!               <sw_l1> <sw_l2> <temporal 0|1> <bit_depth 8|10>
//!
//! base.yuv is a raw yuv420p(10le) dump of the decoded VVC base, one frame
//! per encoded frame, as produced by
//! `ffmpeg -i base.266 -f rawvideo -pix_fmt yuv420p10le base.yuv`.

use lcevc_enc::config::{ChromaFormat, LcevcConfig, ScalingMode, TransformType, UpsampleType};
use lcevc_enc::decoder::{decode_frame, parse_nal};
use lcevc_enc::frame::{Picture, PlaneS16};
use lcevc_enc::mp4::split_lcevc_nals;
use std::io::Read;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 12 {
        eprintln!("usage: verify_full <enh.lcevc> <base.yuv> <W> <H> <baseW> <baseH> <fps> <sw_l1> <sw_l2> <temporal> <bit_depth>");
        std::process::exit(2);
    }
    let (enh_path, base_path) = (&args[1], &args[2]);
    let w: usize = args[3].parse().unwrap();
    let h: usize = args[4].parse().unwrap();
    let bw: usize = args[5].parse().unwrap();
    let bh: usize = args[6].parse().unwrap();
    let _fps: u32 = args[7].parse().unwrap();
    let _sw_l1: u32 = args[8].parse().unwrap();
    let _sw_l2: u32 = args[9].parse().unwrap();
    let temporal = args[10] == "1";
    let depth: u8 = args[11].parse().unwrap();

    let cfg = LcevcConfig {
        width: w as u16,
        height: h as u16,
        chroma: ChromaFormat::C420,
        base_depth: depth,
        enhancement_depth: depth,
        transform: TransformType::Dds,
        upsampler: UpsampleType::Linear,
        scaling_l1: ScalingMode::Scale2D,
        scaling_l2: ScalingMode::Scale2D,
        temporal_enabled: temporal,
        ..Default::default()
    };
    cfg.validate().unwrap();

    let enh = std::fs::read(enh_path).unwrap();
    let nals = split_lcevc_nals(&enh);
    let mut base_file = std::fs::File::open(base_path).unwrap();
    let base_frame_size = bw * bh * 2 + 2 * ((bw / 2) * (bh / 2) * 2);
    let mut base_buf = vec![0u8; base_frame_size];

    let mut temporal_buffers: Vec<PlaneS16> = (0..cfg.num_planes())
        .map(|p| {
            let (pw, ph) = cfg.plane_dimensions(0, p);
            PlaneS16::new(pw as usize, ph as usize)
        })
        .collect();

    let mut stdin = std::io::stdin().lock();
    let out_frame_size = w * h * 2 + 2 * ((w / 2) * (h / 2) * 2);
    let mut out_buf = vec![0u8; out_frame_size];
    let mut filled_all = 0u64;
    let mut max_diff_all = 0u64;
    let mut sse_all = 0u64;
    let mut n_all = 0u64;
    let mut compared = 0u32;

    for (idx, (s, l)) in nals.iter().enumerate() {
        // Read the corresponding base frame.
        let mut got = 0usize;
        while got < base_frame_size {
            match base_file.read(&mut base_buf[got..]) {
                Ok(0) => break,
                Ok(k) => got += k,
                Err(e) => panic!("base read: {e}"),
            }
        }
        if got < base_frame_size {
            eprintln!("base file ended early at frame {idx}");
            break;
        }
        let base_pic = lcevc_enc::yuv::read_yuv420_frame(&base_buf, bw, bh, depth).unwrap();

        // Read the corresponding decoded frame from stdin (the pipe).
        let mut got = 0usize;
        while got < out_frame_size {
            match stdin.read(&mut out_buf[got..]) {
                Ok(0) => {
                    eprintln!("stdin ended early at frame {idx} (got {got}/{out_frame_size})");
                    std::process::exit(1);
                }
                Ok(k) => got += k,
                Err(e) => panic!("stdin: {e}"),
            }
        }
        let decoded = lcevc_enc::yuv::read_yuv420_frame(&out_buf, w, h, depth).unwrap();

        // Mirror-decode this picture.
        let nal = &enh[*s..*s + *l];
        let (_idr, blocks) = parse_nal(nal).unwrap();
        let recon: Picture = match decode_frame(&cfg, &blocks, &base_pic, &mut temporal_buffers) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("frame {idx}: mirror decode error: {e}");
                std::process::exit(1);
            }
        };

        let mut frame_max = 0u64;
        let mut frame_sse = 0u64;
        for p in 0..recon.planes.len() {
            for (a, b) in recon.planes[p].data.iter().zip(decoded.planes[p].data.iter()) {
                let d = (*a as i64 - *b as i64).unsigned_abs();
                if d > frame_max {
                    frame_max = d;
                }
                frame_sse += d * d;
                n_all += 1;
            }
        }
        sse_all += frame_sse;
        if frame_max > max_diff_all {
            max_diff_all = frame_max;
        }
        filled_all += out_frame_size as u64;
        compared += 1;
        if frame_max != 0 {
            eprintln!("frame {idx}: max diff {frame_max}");
        }
    }

    let psnr = if sse_all == 0 {
        f64::INFINITY
    } else {
        10.0 * ((1023.0f64 * 1023.0 * n_all as f64) / sse_all as f64).log10()
    };
    eprintln!(
        "RESULT: {compared} frames compared, {filled_all} bytes, PSNR {psnr:.2} dB, max diff {max_diff_all}"
    );
    if max_diff_all == 0 {
        println!("ALL BIT-EXACT");
    } else {
        std::process::exit(1);
    }
}
