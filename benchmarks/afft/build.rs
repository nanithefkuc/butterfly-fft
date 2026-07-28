use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const LEOPARD_REPOSITORY: &str = "https://github.com/catid/leopard.git";
const LEOPARD_REVISION: &str = "6e5725ebdf9da4370b0bcc4f70fa8eb66f4e6198";
const NANORS_REPOSITORY: &str = "https://github.com/sleepybishop/nanors.git";
const NANORS_REVISION: &str = "593cba13a3db85bcaa9570c61c3aa50cfdd64ccd";

fn command_output(command: &mut Command, description: &str) -> Output {
    let output = command
        .output()
        .unwrap_or_else(|error| panic!("failed to run {description}: {error}"));
    if !output.status.success() {
        panic!(
            "{description} failed with {}\nstdout:\n{}\nstderr:\n{}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
    output
}

fn checkout(root: &Path, name: &str, repository: &str, revision: &str) -> PathBuf {
    let destination = root.join(name);
    if !destination.join(".git").is_dir() {
        if destination.exists() {
            fs::remove_dir_all(&destination).unwrap_or_else(|error| {
                panic!("failed to reset {}: {error}", destination.display())
            });
        }
        command_output(
            Command::new("git")
                .args(["clone", "--filter=blob:none", "--no-checkout", repository])
                .arg(&destination),
            &format!("clone {name}"),
        );
    }

    let current = command_output(
        Command::new("git")
            .arg("-C")
            .arg(&destination)
            .args(["rev-parse", "HEAD"]),
        &format!("read {name} revision"),
    );
    let current = String::from_utf8_lossy(&current.stdout);
    if current.trim() != revision {
        command_output(
            Command::new("git")
                .arg("-C")
                .arg(&destination)
                .args(["fetch", "--depth", "1", "origin", revision]),
            &format!("fetch pinned {name} revision"),
        );
    }
    command_output(
        Command::new("git")
            .arg("-C")
            .arg(&destination)
            .args(["checkout", "--detach", "--force", revision]),
        &format!("checkout pinned {name} revision"),
    );

    let pinned = command_output(
        Command::new("git")
            .arg("-C")
            .arg(&destination)
            .args(["rev-parse", "HEAD"]),
        &format!("verify {name} revision"),
    );
    assert_eq!(
        String::from_utf8_lossy(&pinned.stdout).trim(),
        revision,
        "{name} checkout did not resolve to the pinned revision"
    );
    destination
}

fn compile_leopard(manifest: &Path, source: &Path) {
    let mut build = cc::Build::new();
    build
        .cpp(true)
        .std("c++11")
        .opt_level(3)
        .warnings(false)
        .include(source)
        .file(manifest.join("native/leopard_adapter.cpp"));

    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").expect("target architecture");
    if matches!(target_arch.as_str(), "x86" | "x86_64") {
        build.flag_if_supported("-mssse3");
        build.flag_if_supported("-mavx2");
    }
    build.compile("cafft_leopard_adapter");
}

fn compile_nanors(manifest: &Path, source: &Path) {
    let obl = source.join("deps/obl");
    cc::Build::new()
        .std("c11")
        .opt_level(3)
        .warnings(false)
        .include(source)
        .include(&obl)
        .file(obl.join("oblas_common.c"))
        .file(obl.join("oblas16.c"))
        .file(obl.join("oblas16_afft.c"))
        .file(manifest.join("native/nanors_adapter.c"))
        .compile("cafft_nanors_adapter");
}

fn main() {
    let manifest = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("manifest directory"));
    let upstream = manifest.join(".upstream");
    fs::create_dir_all(&upstream).expect("create upstream checkout directory");

    println!("cargo:rerun-if-changed=native/leopard_adapter.cpp");
    println!("cargo:rerun-if-changed=native/nanors_adapter.c");
    println!("cargo:rerun-if-changed=build.rs");

    let leopard = checkout(&upstream, "leopard", LEOPARD_REPOSITORY, LEOPARD_REVISION);
    let nanors = checkout(&upstream, "nanors", NANORS_REPOSITORY, NANORS_REVISION);

    compile_leopard(&manifest, &leopard);
    compile_nanors(&manifest, &nanors);
}
