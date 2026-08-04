//! Builds the vendored vvenc (VVC encoder) library and links it statically
//! into lcevc-enc. If the vendored source is absent or no C++ compiler can
//! be found, the build continues without it and the encoder falls back to
//! loading libvvenc at runtime (dlopen / LoadLibrary).

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

const VVENC_VERSION: &str = "1.14.0";

fn main() {
    println!("cargo:rerun-if-changed=vendor/vvenc");
    println!("cargo:rerun-if-env-changed=VVENC_SOURCE");

    let root = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let vendor = env::var_os("VVENC_SOURCE")
        .map(PathBuf::from)
        .unwrap_or_else(|| root.join("vendor").join("vvenc"));
    let src_lib = vendor.join("source").join("Lib");
    if !src_lib.join("vvenc").join("vvenc.cpp").exists() {
        eprintln!("build.rs: vendored vvenc source not found at {}", src_lib.display());
        eprintln!("build.rs: lcevc-enc will load libvvenc at runtime instead");
        return;
    }

    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    let target = env::var("TARGET").unwrap();

    // version.h is generated from version.h.in by the vvenc build.
    gen_version_h(&out);

    let inc = include_dirs(&vendor, &out);
    let arch = if target.contains("x86_64") || target.contains("i686") {
        Arch::X86
    } else if target.contains("aarch64") || target.contains("arm") {
        Arch::Arm
    } else {
        Arch::None
    };

    let files = collect_sources(&src_lib, arch);
    if files.is_empty() {
        eprintln!("build.rs: no vvenc sources collected");
        return;
    }

    // Locate the C++ compiler.
    let compiler = find_compiler(&target);
    let Some(cc) = compiler else {
        eprintln!("build.rs: no C++ compiler found; using runtime libvvenc instead");
        return;
    };

    let objdir = out.join("vvenc_obj");
    std::fs::create_dir_all(&objdir).ok();

    let mut objs = Vec::new();
    let mut failed = false;
    for f in &files {
        let obj = objdir.join(format!("{}.o", f.file_stem().unwrap().to_string_lossy()));
        let mut cmd = cc.command(&f, &obj, &inc, arch);
        let status = cmd.status();
        match status {
            Ok(s) if s.success() => objs.push(obj),
            _ => {
                eprintln!("build.rs: failed to compile {}", f.display());
                failed = true;
                break;
            }
        }
    }
    if failed {
        eprintln!("build.rs: vvenc static build failed; using runtime libvvenc instead");
        return;
    }

    // Archive into libvvenc_static.
    let lib = out.join(format!("libvvenc_static{}", cc.archive_suffix()));
    if !cc.archive(&lib, &objs) {
        eprintln!("build.rs: archiving vvenc failed; using runtime libvvenc instead");
        return;
    }

    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=vvenc_static");
    cc.link_runtime();
    println!("cargo:rustc-cfg=have_vvenc");
    eprintln!("build.rs: linked vendored vvenc {VVENC_VERSION} statically");
}

#[derive(Clone, Copy)]
enum Arch {
    X86,
    Arm,
    None,
}

fn include_dirs(vendor: &Path, out: &Path) -> Vec<PathBuf> {
    let lib = vendor.join("source").join("Lib");
    let mut dirs = vec![
        lib.join("CommonLib"),
        lib.join("EncoderLib"),
        lib.join("Utilities"),
        lib.join("vvenc"),
        lib.clone(),
        vendor.join("include"),
        out.join("vvenc_inc"),
    ];
    // The architecture-specific SIMD sources include headers from their
    // parent and sibling directories.
    for d in ["x86", "x86/sse41", "x86/sse42", "x86/avx", "x86/avx2"] {
        dirs.push(lib.join("CommonLib").join(d));
    }
    for d in ["arm", "arm/neon", "arm/sve", "arm/sve2"] {
        dirs.push(lib.join("CommonLib").join(d));
    }
    dirs
}

fn gen_version_h(out: &Path) {
    let dir = out.join("vvenc_inc").join("vvenc");
    std::fs::create_dir_all(&dir).ok();
    let v = VVENC_VERSION;
    let content = format!(
        "#if !defined( VVENC_VERSION )\n\
         #define VVENC_VERSION \"{v}\"\n\
         #define VVENC_VERSION_MAJOR 1\n\
         #define VVENC_VERSION_MINOR 14\n\
         #define VVENC_VERSION_PATCH 0\n\
         #ifdef _WIN32\n\
         #define VVENC_VS_VERSION      1,14,0\n\
         #define VVENC_VS_VERSION_STR \"1.14.0\"\n\
         #endif\n\
         #endif\n"
    );
    let _ = std::fs::write(dir.join("version.h"), content);
    // vvenc.h is also generated from a template (one substitution).
    let _ = std::fs::write(
        dir.join("vvenc.h"),
        VVENC_H_TEMPLATE.replace("@VVENC_USE_UNSTABLE_API@", "0"),
    );
}

/// Path to the generated vvenc headers (kept for the include list).
#[allow(dead_code)]
fn gen_include_dir(out: &Path) -> PathBuf {
    out.join("vvenc_inc")
}

const VVENC_H_TEMPLATE: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/vendor/vvenc/include/vvenc/vvenc.h.in"
));

fn collect_sources(src_lib: &Path, arch: Arch) -> Vec<PathBuf> {
    let mut files = Vec::new();
    for dir in ["CommonLib", "DecoderLib", "EncoderLib", "Utilities", "vvenc"] {
        let mut names: Vec<_> = std::fs::read_dir(src_lib.join(dir))
            .map(|rd| {
                rd.filter_map(|e| e.ok())
                    .map(|e| e.path())
                    .filter(|p| p.extension().map(|x| x == "cpp").unwrap_or(false))
                    .collect()
            })
            .unwrap_or_default();
        names.sort();
        files.extend(names);
    }
    // Architecture-specific SIMD sources (with their compile defines).
    match arch {
        Arch::X86 => {
            for sub in ["sse41", "sse42", "avx", "avx2"] {
                let mut names: Vec<_> = std::fs::read_dir(src_lib.join("CommonLib").join("x86").join(sub))
                    .map(|rd| {
                        rd.filter_map(|e| e.ok())
                            .map(|e| e.path())
                            .filter(|p| p.extension().map(|x| x == "cpp").unwrap_or(false))
                            .collect()
                    })
                    .unwrap_or_default();
                names.sort();
                files.extend(names);
            }
            let mut base: Vec<_> = std::fs::read_dir(src_lib.join("CommonLib").join("x86"))
                .map(|rd| {
                    rd.filter_map(|e| e.ok())
                        .map(|e| e.path())
                        .filter(|p| p.extension().map(|x| x == "cpp").unwrap_or(false))
                        .collect()
                })
                .unwrap_or_default();
            base.sort();
            files.extend(base);
        }
        Arch::Arm => {
            let mut names: Vec<_> = std::fs::read_dir(src_lib.join("CommonLib").join("arm"))
                .map(|rd| {
                    rd.filter_map(|e| e.ok())
                        .map(|e| e.path())
                        .filter(|p| p.extension().map(|x| x == "cpp").unwrap_or(false))
                        .collect()
                })
                .unwrap_or_default();
            names.sort();
            files.extend(names);
            let neon: Vec<_> = std::fs::read_dir(src_lib.join("CommonLib").join("arm").join("neon"))
                .map(|rd| {
                    rd.filter_map(|e| e.ok())
                        .map(|e| e.path())
                        .filter(|p| p.extension().map(|x| x == "cpp").unwrap_or(false))
                        .collect()
                })
                .unwrap_or_default();
            files.extend(neon);
        }
        Arch::None => {}
    }
    files
}

struct Compiler {
    kind: CompilerKind,
    cxx: PathBuf,
    // MSVC: extra env for the compile (INCLUDE/LIB)
    env: Vec<(String, String)>,
}

enum CompilerKind {
    Msvc,
    Gnu,
}

impl Compiler {
    fn command(&self, src: &Path, obj: &Path, inc: &[PathBuf], arch: Arch) -> Command {
        let mut cmd = Command::new(&self.cxx);
        match self.kind {
            CompilerKind::Msvc => {
                cmd.arg("/nologo").arg("/c").arg("/O2").arg("/MD");
                cmd.arg("/EHsc").arg("/std:c++17").arg("/DWIN32");
                cmd.arg("/D_WIN32").arg("/D_WINDOWS");
                cmd.arg(format!("/Fo{}", obj.display()));
                for i in inc {
                    cmd.arg(format!("/I{}", i.display()));
                }
                match arch {
                    Arch::X86 => {
                        cmd.arg("/DUSE_SSE41").arg("/DUSE_SSE42").arg("/DUSE_AVX").arg("/DUSE_AVX2");
                    }
                    Arch::Arm => {
                        cmd.arg("/DUSE_NEON");
                    }
                    Arch::None => {}
                }
                cmd.arg(src);
            }
            CompilerKind::Gnu => {
                cmd.arg("-c").arg("-O2").arg("-fPIC");
                cmd.arg("-std=c++17").arg("-w");
                for i in inc {
                    cmd.arg(format!("-I{}", i.display()));
                }
                match arch {
                    Arch::X86 => {
                        cmd.arg("-DUSE_SSE41").arg("-DUSE_SSE42").arg("-DUSE_AVX").arg("-DUSE_AVX2");
                    }
                    Arch::Arm => {
                        cmd.arg("-DUSE_NEON");
                    }
                    Arch::None => {}
                }
                cmd.arg("-o").arg(obj).arg(src);
            }
        }
        for (k, v) in &self.env {
            cmd.env(k, v);
        }
        cmd
    }

    fn archive_suffix(&self) -> &'static str {
        match self.kind {
            CompilerKind::Msvc => ".lib",
            CompilerKind::Gnu => ".a",
        }
    }

    fn archive(&self, lib: &Path, objs: &[PathBuf]) -> bool {
        match self.kind {
            CompilerKind::Gnu => {
                let mut cmd = Command::new("ar");
                cmd.arg("crus").arg(lib);
                cmd.args(objs);
                for (k, v) in &self.env {
                    cmd.env(k, v);
                }
                cmd.status().map(|s| s.success()).unwrap_or(false)
            }
            CompilerKind::Msvc => {
                // lib.exe lives next to cl.exe.
                let libexe = self.cxx.with_file_name("lib.exe");
                let mut cmd = Command::new(libexe);
                cmd.arg("/nologo").arg("/OUT:").arg(lib);
                cmd.args(objs);
                for (k, v) in &self.env {
                    cmd.env(k, v);
                }
                cmd.status().map(|s| s.success()).unwrap_or(false)
            }
        }
    }

    fn link_runtime(&self) {
        match self.kind {
            CompilerKind::Msvc => {
                // /MD objects pull the dynamic CRT, which rustc links by
                // default; nothing extra needed.
            }
            CompilerKind::Gnu => {
                println!("cargo:rustc-link-lib=stdc++");
            }
        }
    }
}

fn find_compiler(target: &str) -> Option<Compiler> {
    if target.contains("windows") {
        // 1. cl.exe already on PATH (a VS developer prompt).
        if let Ok(_) = Command::new("cl").arg("/?") .status() {
            return Some(Compiler {
                kind: CompilerKind::Msvc,
                cxx: PathBuf::from("cl"),
                env: vec![],
            });
        }
        // 2. Locate the BuildTools install via vswhere.
        if let Some((cl, env)) = msvc_vswhere() {
            return Some(Compiler {
                kind: CompilerKind::Msvc,
                cxx: cl,
                env,
            });
        }
        None
    } else {
        for cand in ["c++", "g++", "clang++"] {
            if let Ok(_) = Command::new(cand).arg("--version").status() {
                return Some(Compiler {
                    kind: CompilerKind::Gnu,
                    cxx: PathBuf::from(cand),
                    env: vec![],
                });
            }
        }
        None
    }
}

/// Locate the MSVC C++ tools and Windows SDK via vswhere, returning
/// cl.exe and the INCLUDE/LIB environment it needs.
fn msvc_vswhere() -> Option<(PathBuf, Vec<(String, String)>)> {
    let vswhere = r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe";
    if !Path::new(vswhere).exists() {
        return None;
    }
    let out = Command::new(vswhere)
        .args([
            "-latest",
            "-products",
            "*",
            "-requires",
            "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
            "-find",
            r"VC\Tools\MSVC\**\bin\Hostx64\x64\cl.exe",
        ])
        .output()
        .ok()?;
    let cl_path = String::from_utf8_lossy(&out.stdout)
        .lines()
        .next()?
        .trim()
        .to_string();
    if cl_path.is_empty() {
        return None;
    }
    let cl = PathBuf::from(&cl_path);
    // VC tools root: ...\VC\Tools\MSVC\<ver>\
    let vc_root = cl.parent()?.parent()?.parent()?.parent()?.to_path_buf();
    let sdk_root = r"C:\Program Files (x86)\Windows Kits\10";
    // Newest SDK version present.
    let sdk_ver = std::fs::read_dir(format!(r"{sdk_root}\Include"))
        .ok()?
        .filter_map(|e| e.ok())
        .filter_map(|e| e.file_name().to_str().map(String::from))
        .filter(|n| n.starts_with("10."))
        .max()?;
    let mut include = vec![vc_root.join("include")];
    include.push(PathBuf::from(format!(r"{sdk_root}\Include\{sdk_ver}\ucrt")));
    include.push(PathBuf::from(format!(r"{sdk_root}\Include\{sdk_ver}\um")));
    include.push(PathBuf::from(format!(r"{sdk_root}\Include\{sdk_ver}\shared")));
    let mut lib = vec![vc_root.join("lib").join("x64")];
    lib.push(PathBuf::from(format!(r"{sdk_root}\Lib\{sdk_ver}\ucrt\x64")));
    lib.push(PathBuf::from(format!(r"{sdk_root}\Lib\{sdk_ver}\um\x64")));
    let join = |paths: &[PathBuf]| -> String {
        paths
            .iter()
            .map(|p| p.display().to_string())
            .collect::<Vec<_>>()
            .join(";")
    };
    let env = vec![
        ("INCLUDE".to_string(), join(&include)),
        ("LIB".to_string(), join(&lib)),
    ];
    Some((cl, env))
}
