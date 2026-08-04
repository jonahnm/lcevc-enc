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
    println!("cargo:rustc-check-cfg=cfg(have_vvenc)");
    // Status is written to a findable location (target/vvenc-build.txt)
    // and mirrored into the build script's OUT_DIR; `cargo:warning`
    // lines surface it in every build output.
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let status_file = manifest.join("target").join("vvenc-build.txt");
    let _ = std::fs::create_dir_all(status_file.parent().unwrap());
    let mut status = String::new();

    let root = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let vendor = env::var_os("VVENC_SOURCE")
        .map(PathBuf::from)
        .unwrap_or_else(|| root.join("vendor").join("vvenc"));
    let src_lib = vendor.join("source").join("Lib");
    if !src_lib.join("vvenc").join("vvenc.cpp").exists() {
        let msg = format!("vendored vvenc source not found at {}", src_lib.display());
        eprintln!("build.rs: {msg}");
        eprintln!("build.rs: lcevc-enc will load libvvenc at runtime instead");
        finish_status(&status_file, &format!("FALLBACK: {msg}"));
        return;
    }

    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    status.push_str(&format!("vendored vvenc: {}\n", VVENC_VERSION));
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
        let msg = "no vvenc sources collected".to_string();
        eprintln!("build.rs: {msg}");
        finish_status(&status_file, &format!("FALLBACK: {msg}"));
        return;
    }
    status.push_str(&format!("sources: {} files\n", files.len()));

    // Probe every candidate compiler with the first source file; the
    // first one that actually compiles it wins (a broken cl.exe on PATH,
    // for example, must not veto a working vswhere-found toolchain).
    let candidates = find_compilers(&target);
    let mut cc: Option<Compiler> = None;
    let mut probe_error: Option<String> = None;
    for cand in &candidates {
        let obj = out.join("vvenc_probe.o");
        let out = cand.command(&files[0], &obj, &inc, arch).output();
        match out {
            Ok(o) if o.status.success() => {
                cc = Some(cand.clone());
                let _ = std::fs::remove_file(&obj);
                break;
            }
            Ok(o) => {
                probe_error = Some(format!(
                    "{} (exit {})",
                    cand.cxx.display(),
                    o.status.code().unwrap_or(-1)
                ));
            }
            Err(e) => {
                probe_error = Some(format!("{}: {e}", cand.cxx.display()));
            }
        }
    }
    let Some(cc) = cc else {
        let msg = format!(
            "no working C++ compiler (probed {}; last failure: {})",
            candidates.len(),
            probe_error.unwrap_or_else(|| "unknown".into())
        );
        eprintln!("build.rs: {msg}");
        finish_status(&status_file, &format!("FALLBACK: {msg}"));
        return;
    };
    status.push_str(&format!(
        "compiler: {} ({})\n",
        cc.cxx.display(),
        match cc.kind {
            CompilerKind::Msvc => "msvc",
            CompilerKind::Gnu => "gnu",
        }
    ));

    let objdir = out.join("vvenc_obj");
    std::fs::create_dir_all(&objdir).ok();

    let mut objs = Vec::new();
    let mut failed = false;
    for f in &files {
        let obj = objdir.join(format!("{}.o", f.file_stem().unwrap().to_string_lossy()));
        let out = cc.command(&f, &obj, &inc, arch).output();
        match out {
            Ok(o) if o.status.success() => objs.push(obj),
            Ok(o) => {
                let err = String::from_utf8_lossy(&o.stderr);
                let stdout = String::from_utf8_lossy(&o.stdout);
                println!(
                    "cargo:warning=vvenc: compile failed for {} (exit {})",
                    f.display(),
                    o.status.code().unwrap_or(-1)
                );
                let mut shown = 0;
                for l in stdout.lines().chain(err.lines()) {
                    if !l.trim().is_empty() && shown < 8 {
                        println!("cargo:warning=vvenc:     {l}");
                        shown += 1;
                    }
                }
                if shown == 0 {
                    println!("cargo:warning=vvenc:     (no compiler output)");
                }
                failed = true;
                break;
            }
            Err(e) => {
                println!("cargo:warning=vvenc: could not run compiler for {}: {e}", f.display());
                failed = true;
                break;
            }
        }
    }
    if !failed && objs.len() < 60 {
        eprintln!("build.rs: vvenc produced only {} objects; treating as failure", objs.len());
        failed = true;
    }
    if !failed {
        status.push_str(&format!("objects: {}\n", objs.len()));
    }
    if failed {
        let msg = format!("vvenc static build failed (see the compile errors above)");
        eprintln!("build.rs: {msg}; using runtime libvvenc instead");
        finish_status(&status_file, &format!("FALLBACK: {msg}"));
        return;
    }

    // Archive into libvvenc_static.
    let lib = out.join(format!("libvvenc_static{}", cc.archive_suffix()));
    if !cc.archive(&lib, &objs) {
        let msg = "archiving vvenc failed".to_string();
        eprintln!("build.rs: {msg}; using runtime libvvenc instead");
        finish_status(&status_file, &format!("FALLBACK: {msg}"));
        return;
    }

    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=vvenc_static");
    cc.link_runtime();
    println!("cargo:rustc-cfg=have_vvenc");
    eprintln!("build.rs: linked vendored vvenc {VVENC_VERSION} statically");
    finish_status(&status_file, "OK: vendored vvenc linked statically");
}

fn finish_status(path: &Path, line: &str) {
    let _ = std::fs::write(path, format!("{line}\n"));
    // cargo:warning lines are always shown in the build output.
    println!("cargo:warning=vvenc: {line}");
    eprintln!("build.rs: {line} (see {})", path.display());
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

#[derive(Clone)]
struct Compiler {
    kind: CompilerKind,
    cxx: PathBuf,
    // MSVC: extra env for the compile (INCLUDE/LIB)
    env: Vec<(String, String)>,
}

#[derive(Clone)]
enum CompilerKind {
    Msvc,
    Gnu,
}

impl Compiler {
    /// Architecture defines and flags: the vvenc build applies the SIMD
    /// define (and its ISA flag) only to the matching per-directory
    /// sources, so mirror that per file.
    fn simd_for(&self, src: &Path, arch: Arch) -> (Vec<String>, Vec<String>) {
        let s = src.to_string_lossy();
        let (mut defs, mut flags): (Vec<String>, Vec<String>) = (Vec::new(), Vec::new());
        match arch {
            Arch::X86 => {
                if s.contains("/sse41/") {
                    defs.push("USE_SSE41".into());
                    if matches!(self.kind, CompilerKind::Gnu) {
                        flags.push("-msse4.1".into());
                    }
                } else if s.contains("/sse42/") {
                    defs.push("USE_SSE42".into());
                    if matches!(self.kind, CompilerKind::Gnu) {
                        flags.push("-msse4.2".into());
                    }
                } else if s.contains("/avx/") {
                    defs.push("USE_AVX".into());
                    if matches!(self.kind, CompilerKind::Msvc) {
                        flags.push("/arch:AVX".into());
                    } else {
                        flags.push("-mavx".into());
                    }
                } else if s.contains("/avx2/") {
                    defs.push("USE_AVX2".into());
                    if matches!(self.kind, CompilerKind::Msvc) {
                        flags.push("/arch:AVX2".into());
                    } else {
                        flags.push("-mavx2".into());
                    }
                }
            }
            Arch::Arm => {
                if s.contains("/neon/") {
                    defs.push("USE_NEON".into());
                }
            }
            Arch::None => {}
        }
        (defs, flags)
    }

    fn command(&self, src: &Path, obj: &Path, inc: &[PathBuf], arch: Arch) -> Command {
        let mut cmd = Command::new(&self.cxx);
        let (defs, flags) = self.simd_for(src, arch);
        match self.kind {
            CompilerKind::Msvc => {
                cmd.arg("/nologo").arg("/c").arg("/O2").arg("/MD");
                cmd.arg("/EHsc").arg("/std:c++17").arg("/DWIN32");
                cmd.arg("/D_WIN32").arg("/D_WINDOWS");
                cmd.arg(format!("/Fo{}", obj.display()));
                for i in inc {
                    cmd.arg(format!("/I{}", i.display()));
                }
                for d in &defs {
                    cmd.arg(format!("/D{d}"));
                }
                for f in &flags {
                    cmd.arg(f);
                }
                cmd.arg(src);
            }
            CompilerKind::Gnu => {
                cmd.arg("-c").arg("-O2").arg("-fPIC");
                cmd.arg("-std=c++17").arg("-w");
                for i in inc {
                    cmd.arg(format!("-I{}", i.display()));
                }
                for d in &defs {
                    cmd.arg(format!("-D{d}"));
                }
                for f in &flags {
                    cmd.arg(f);
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

fn find_compilers(target: &str) -> Vec<Compiler> {
    let mut out = Vec::new();
    if target.contains("windows") {
        // 1. cl.exe already on PATH (a VS developer prompt).
        if let Ok(_) = Command::new("cl").arg("/?").status() {
            out.push(Compiler {
                kind: CompilerKind::Msvc,
                cxx: PathBuf::from("cl"),
                env: vec![],
            });
        }
        // 2. Locate the BuildTools install via vswhere.
        if let Some((cl, env)) = msvc_vswhere() {
            out.push(Compiler {
                kind: CompilerKind::Msvc,
                cxx: cl,
                env,
            });
        }
        // 3. Probe the well-known install roots directly (no vswhere).
        for root in [
            r"C:\Program Files (x86)\Microsoft Visual Studio",
            r"C:\Program Files\Microsoft Visual Studio",
        ] {
            if let Some((cl, env)) = scan_vs_root(root) {
                out.push(Compiler {
                    kind: CompilerKind::Msvc,
                    cxx: cl,
                    env,
                });
            }
        }
    } else {
        for cand in ["c++", "g++", "clang++"] {
            if let Ok(_) = Command::new(cand).arg("--version").status() {
                out.push(Compiler {
                    kind: CompilerKind::Gnu,
                    cxx: PathBuf::from(cand),
                    env: vec![],
                });
            }
        }
    }
    out
}

/// Find cl.exe + the SDK environment by walking the Visual Studio roots,
/// independent of vswhere.
fn scan_vs_root(root: &str) -> Option<(PathBuf, Vec<(String, String)>)> {
    let root = Path::new(root);
    if !root.is_dir() {
        return None;
    }
    // <root>/<edition>/VC/Tools/MSVC/<ver>/bin/Host<x64>/x64/cl.exe
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Ok(rd) = std::fs::read_dir(root) {
        for edition in rd.filter_map(|e| e.ok()) {
            let vc = edition.path().join("VC").join("Tools").join("MSVC");
            if let Ok(tools) = std::fs::read_dir(&vc) {
                for ver in tools.filter_map(|e| e.ok()) {
                    let bin = ver.path().join("bin");
                    if let Ok(hosts) = std::fs::read_dir(&bin) {
                        for host in hosts.filter_map(|e| e.ok()) {
                            let cl = host.path().join("x64").join("cl.exe");
                            if cl.exists() {
                                candidates.push(cl);
                            }
                        }
                    }
                }
            }
        }
    }
    // Newest MSVC version first.
    candidates.sort_by(|a, b| b.cmp(a));
    let cl = candidates.into_iter().next()?;
    let vc_root = cl
        .parent()?
        .parent()?
        .parent()?
        .parent()?
        .to_path_buf();
    let sdk = find_sdk()?;
    let (include, lib) = sdk_paths(&vc_root, &sdk)?;
    let join = |paths: &[PathBuf]| -> String {
        paths
            .iter()
            .map(|p| p.display().to_string())
            .collect::<Vec<_>>()
            .join(";")
    };
    Some((
        cl,
        vec![
            ("INCLUDE".to_string(), join(&include)),
            ("LIB".to_string(), join(&lib)),
        ],
    ))
}

/// Returns the Windows SDK root and the newest version directory name.
fn find_sdk() -> Option<(PathBuf, String)> {
    for base in [
        r"C:\Program Files (x86)\Windows Kits\10",
        r"C:\Program Files\Windows Kits\10",
    ] {
        let base = Path::new(base);
        if let Ok(rd) = std::fs::read_dir(base.join("Include")) {
            let mut vers: Vec<String> = rd
                .filter_map(|e| e.ok())
                .filter_map(|e| e.file_name().to_str().map(String::from))
                .filter(|n| n.starts_with("10."))
                .collect();
            vers.sort();
            if let Some(v) = vers.pop() {
                return Some((base.to_path_buf(), v));
            }
        }
    }
    None
}

fn sdk_paths(vc_root: &Path, sdk: &(PathBuf, String)) -> Option<(Vec<PathBuf>, Vec<PathBuf>)> {
    let (root, ver) = sdk;
    let inc = vec![
        vc_root.join("include"),
        root.join("Include").join(ver).join("ucrt"),
        root.join("Include").join(ver).join("um"),
        root.join("Include").join(ver).join("shared"),
    ];
    let lib = vec![
        vc_root.join("lib").join("x64"),
        root.join("Lib").join(ver).join("ucrt").join("x64"),
        root.join("Lib").join(ver).join("um").join("x64"),
    ];
    // Keep only directories that exist.
    let inc: Vec<PathBuf> = inc.into_iter().filter(|p| p.is_dir()).collect();
    let lib: Vec<PathBuf> = lib.into_iter().filter(|p| p.is_dir()).collect();
    if inc.is_empty() || lib.is_empty() {
        return None;
    }
    Some((inc, lib))
}

/// Locate the MSVC C++ tools and Windows SDK via vswhere, returning
/// cl.exe and the INCLUDE/LIB environment it needs.
fn msvc_vswhere() -> Option<(PathBuf, Vec<(String, String)>)> {
    let vswhere = r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe";
    if !Path::new(vswhere).exists() {
        eprintln!("build.rs: vswhere not found at {vswhere}");
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
            // HostX64 vs Hostx64 casing varies; use a wildcard.
            r"VC\Tools\MSVC\**\bin\Host*\x64\cl.exe",
        ])
        .output();
    let out = match out {
        Ok(o) => o,
        Err(e) => {
            eprintln!("build.rs: vswhere failed to run: {e}");
            return None;
        }
    };
    let cl_path = String::from_utf8_lossy(&out.stdout)
        .lines()
        .next()
        .unwrap_or("")
        .trim()
        .to_string();
    if cl_path.is_empty() {
        eprintln!(
            "build.rs: vswhere found no cl.exe (stderr: {})",
            String::from_utf8_lossy(&out.stderr).trim()
        );
        return None;
    }
    let cl = PathBuf::from(&cl_path);
    // VC tools root: ...\VC\Tools\MSVC\<ver>\
    let vc_root = cl.parent()?.parent()?.parent()?.parent()?.to_path_buf();
    let sdk = find_sdk()?;
    let (include, lib) = sdk_paths(&vc_root, &sdk)?;
    let join = |paths: &[PathBuf]| -> String {
        paths
            .iter()
            .map(|p| p.display().to_string())
            .collect::<Vec<_>>()
            .join(";")
    };
    Some((
        cl,
        vec![
            ("INCLUDE".to_string(), join(&include)),
            ("LIB".to_string(), join(&lib)),
        ],
    ))
}
