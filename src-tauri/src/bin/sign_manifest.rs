//! Offline manifest signing tool.
//!
//! Produces the release trust artifacts (`manifests/trusted_keys.json` and
//! `manifests/signed_manifest.json`) from a production Ed25519 key that never
//! ships in the app binary. The list of files to attest is read from the live
//! model/engine catalog (`manifest::manifest_entries`), so the signed manifest
//! can never drift from the hashes the downloader actually enforces.
//!
//! Usage:
//!   cargo run --bin sign_manifest -- keygen [--out <path>] [--key-id <id>] [--force]
//!   cargo run --bin sign_manifest -- sign   [--key <path>] [--version <n>]
//!
//! The private key file is secret: back it up offline and never commit it.

use std::{
    collections::HashMap,
    error::Error,
    path::{Path, PathBuf},
};

use base64::engine::general_purpose::STANDARD as B64;
use base64::Engine as _;
use ed25519_dalek::{Signer, SigningKey};
use serde::{Deserialize, Serialize};

use local_ai_installer_lib::manifest::{
    manifest_entries, ManifestPayload, SignedDocument, TrustedKey,
};

const DEFAULT_KEY_ID: &str = "localai-root-2026";

/// On-disk private key. `private_seed_hex` is the 32-byte Ed25519 seed; keep it
/// secret. `public_key_hex` is stored alongside only for convenience/checking.
#[derive(Serialize, Deserialize)]
struct KeyFile {
    key_id: String,
    algo: String,
    private_seed_hex: String,
    public_key_hex: String,
}

fn default_key_path() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".localai-signing")
        .join("localai-signing-key.json")
}

fn manifests_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("manifests")
}

/// Minimal flag parser: pulls `--name value` out of the arg list, leaving the
/// subcommand handling to the caller. Boolean flags are handled separately.
fn flag_value(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|arg| arg == name)
        .and_then(|index| args.get(index + 1))
        .cloned()
}

fn has_flag(args: &[String], name: &str) -> bool {
    args.iter().any(|arg| arg == name)
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("keygen") => keygen(&args[1..]),
        Some("sign") => sign(&args[1..]),
        _ => {
            eprintln!(
                "用法:\n  \
                 sign_manifest keygen [--out <私钥文件>] [--key-id <id>] [--force]\n  \
                 sign_manifest sign   [--key <私钥文件>] [--version <n>]"
            );
            std::process::exit(2);
        }
    }
}

fn keygen(args: &[String]) -> Result<(), Box<dyn Error>> {
    let out = flag_value(args, "--out")
        .map(PathBuf::from)
        .unwrap_or_else(default_key_path);
    let key_id = flag_value(args, "--key-id").unwrap_or_else(|| DEFAULT_KEY_ID.to_string());

    if out.exists() && !has_flag(args, "--force") {
        return Err(format!(
            "私钥文件已存在: {}\n若确认要覆盖（会作废旧签名），请加 --force。",
            out.display()
        )
        .into());
    }

    let mut seed = [0u8; 32];
    getrandom::getrandom(&mut seed)?;
    let signing = SigningKey::from_bytes(&seed);
    let public_key_hex = hex::encode(signing.verifying_key().to_bytes());

    let key_file = KeyFile {
        key_id: key_id.clone(),
        algo: "ed25519".to_string(),
        private_seed_hex: hex::encode(seed),
        public_key_hex: public_key_hex.clone(),
    };

    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&out, serde_json::to_string_pretty(&key_file)?)?;

    println!("已生成生产签名密钥。");
    println!("  key_id     : {key_id}");
    println!("  公钥(hex)  : {public_key_hex}");
    println!("  私钥文件   : {}", out.display());
    println!();
    println!("⚠ 请立刻备份该私钥文件到安全的离线位置，且【绝不要】提交到 git 或分发。");
    println!("  丢失私钥将无法再对新清单签名；私钥泄露则签名保护失效。");
    println!("下一步: cargo run --bin sign_manifest -- sign --key \"{}\"", out.display());
    Ok(())
}

fn sign(args: &[String]) -> Result<(), Box<dyn Error>> {
    let key_path = flag_value(args, "--key")
        .map(PathBuf::from)
        .unwrap_or_else(default_key_path);
    let version: u32 = flag_value(args, "--version")
        .map(|value| value.parse())
        .transpose()?
        .unwrap_or(1);

    let raw = std::fs::read_to_string(&key_path).map_err(|error| {
        format!(
            "读不到私钥文件 {}: {error}\n先运行 keygen 生成，或用 --key 指定路径。",
            key_path.display()
        )
    })?;
    let key_file: KeyFile = serde_json::from_str(&raw)?;

    let seed_bytes = hex::decode(key_file.private_seed_hex.trim())?;
    let seed: [u8; 32] = seed_bytes
        .as_slice()
        .try_into()
        .map_err(|_| "私钥种子长度不对，应为 32 字节 hex。")?;
    let signing = SigningKey::from_bytes(&seed);
    let public_key_hex = hex::encode(signing.verifying_key().to_bytes());

    let entries = manifest_entries();
    let entry_count = entries.len();
    let payload = ManifestPayload {
        version,
        created: today(),
        entries,
    };
    // Sign the exact JSON bytes the verifier will decode from payload_b64.
    let payload_bytes = serde_json::to_vec(&payload)?;
    let signature = signing.sign(&payload_bytes);

    let doc = SignedDocument {
        key_id: key_file.key_id.clone(),
        algo: "ed25519".to_string(),
        signature: B64.encode(signature.to_bytes()),
        payload_b64: B64.encode(&payload_bytes),
    };
    let trusted = vec![TrustedKey {
        key_id: key_file.key_id.clone(),
        public_key_hex: public_key_hex.clone(),
    }];

    let dir = manifests_dir();
    std::fs::create_dir_all(&dir)?;
    std::fs::write(
        dir.join("signed_manifest.json"),
        serde_json::to_string_pretty(&doc)?,
    )?;
    std::fs::write(
        dir.join("trusted_keys.json"),
        serde_json::to_string_pretty(&trusted)?,
    )?;

    // Self-check: the freshly written artifacts must verify with this key set,
    // so a bad key/payload is caught here rather than at runtime on a user's box.
    let mut keys = HashMap::new();
    keys.insert(
        key_file.key_id.clone(),
        signing.verifying_key(),
    );
    verify_roundtrip(&doc, &keys)?;

    println!("已重新签名清单（{entry_count} 个文件）。");
    println!("  key_id : {}", key_file.key_id);
    println!("  公钥   : {public_key_hex}");
    println!("  version: {version}");
    println!("  写入   : {}", dir.join("signed_manifest.json").display());
    println!("         : {}", dir.join("trusted_keys.json").display());
    println!("自检通过：签名可用当前公钥验证。");
    Ok(())
}

/// Confirm the written document verifies against the key we just used. Mirrors
/// the runtime's strict verification so signing failures surface immediately.
fn verify_roundtrip(
    doc: &SignedDocument,
    keys: &HashMap<String, ed25519_dalek::VerifyingKey>,
) -> Result<(), Box<dyn Error>> {
    let key = keys
        .get(&doc.key_id)
        .ok_or("自检失败：签发者不在信任集合里。")?;
    let payload = B64.decode(doc.payload_b64.trim())?;
    let sig_bytes = B64.decode(doc.signature.trim())?;
    let sig_bytes: [u8; 64] = sig_bytes
        .as_slice()
        .try_into()
        .map_err(|_| "自检失败：签名长度不对。")?;
    let signature = ed25519_dalek::Signature::from_bytes(&sig_bytes);
    key.verify_strict(&payload, &signature)
        .map_err(|_| "自检失败：签名无法用公钥验证。")?;
    Ok(())
}

fn today() -> String {
    chrono::Utc::now().format("%Y-%m-%d").to_string()
}
