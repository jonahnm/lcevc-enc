//! mirror_dump — decode a full LCEVC segment with the built-in mirror decoder
//! and dump the reconstructed 4K frames to a raw yuv420p10le file.
//!
//! Usage:
//!   mirror_dump <enh.lcevc> <base.yuv> <W> <H> <baseW> <baseH> <bit_depth> <out.yuv>

use lcevc_enc::config::{ChromaFormat, LcevcConfig, ScalingMode, TransformType, UpsampleType};
use lcevc_enc::decoder::{decode_frame, parse_nal};
use lcevc_enc::frame::PlaneS16;
use lcevc_enc::mp4::split_lcevc_nals;
use std::io::{Read, Write};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 9 {
        eprintln!("usage: mirror_dump <enh.lcevc> <base.yuv> <W> <H> <baseW> <baseH> <bit_depth> <out.yuv>");
        std::process::exit(2);
    }
    let (enh_path, base_path) = (&args[1], &args[2]);
    let w: usize = args[3].parse().unwrap();
    let h: usize = args[4].parse().unwrap();
    let bw: usize = args[5].parse().unwrap();
    let bh: usize = args[6].parse().unwrap();
    let depth: u8 = args[7].parse().unwrap();
    let out_path = &args[8];

    let cfg = LcevcConfig {
        width: w as u16,
        height: h as u16,
        chroma: ChromaFormat::C420,
        base_depth: depth,
        enhancement_depth: depth,
        transform: TransformType::Dds,
        upsampler: UpsampleType::Linear,
        scaling_l1: ScalingMode::Scale0D,
        scaling_l2: ScalingMode::Scale2D,
        temporal_enabled: false,
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

    let mut out_file = std::fs::File::create(out_path).unwrap();
    let mut n = 0u32;

    for (idx, (s, l)) in nals.iter().enumerate() {
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

        let nal = &enh[*s..*s + *l];
        let (_idr, blocks) = match parse_nal(nal) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("frame {idx}: parse error: {e}");
                continue;
            }
        };
        let recon = match decode_frame(&cfg, &blocks, &base_pic, &mut temporal_buffers) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("frame {idx}: mirror decode error: {e}");
                std::process::exit(1);
            }
        };

        let mut frame = Vec::with_capacity(w * h * 3 * 2);
        for p in 0..recon.planes.len() {
            let (pw, ph) = cfg.plane_dimensions(0, p);
            let plane = &recon.planes[p];
            for y in 0..ph as usize {
                for x in 0..pw as usize {
                    let v: i32 = plane.data[y * plane.width + x].into();
                    let cv = if depth == 8 { v.clamp(0, 255) << 2 } else { v.clamp(0, 1023) };
                    frame.extend_from_slice(&(cv as u16).to_le_bytes());
                }
            }
        }
        out_file.write_all(&frame).unwrap();
        n += 1;
        eprintln!("frame {idx} written (total {n})");
    }
    eprintln!("done: {n} frames -> {out_path}");
}
