//! Ed25519 signed manifest verification.
//!
//! After a model file or engine package passes its self-declared SHA256 check,
//! it is additionally verified against a bundled, signed manifest. The manifest
//! is the *authority*: it attests the expected SHA256 for each file and is signed
//! with an Ed25519 key whose public half is embedded in the binary. This gives us
//! authenticity (the file is the one the publisher blessed), not just integrity.
//!
//! Trust chain:
//!   - `trusted_keys.json`   — root public key(s), embedded in the binary.
//!   - `signed_manifest.json`— the signed file manifest (key_id + signature + payload).
//!   - `key_rotations.json`  — signed authorizations letting a currently-trusted
//!                             key vouch for a new public key (public-key rotation).
//!
//! SECURITY NOTE: the manifest shipped in this repo is signed with a *development*
//! key generated in the ignored test below. That private key is not secret, so it
//! provides no real authenticity yet. Before release, regenerate the manifest with
//! an offline-held production key and replace `trusted_keys.json`. Rotation of a
//! compromised key is distributed via an app update carrying a new `trusted_keys.json`.

use std::{collections::HashMap, path::Path};

use base64::engine::general_purpose::STANDARD as B64;
use base64::Engine as _;
use ed25519_dalek::{Signature, VerifyingKey};
use serde::{Deserialize, Serialize};

use crate::error::{AppError, AppResult};

const TRUSTED_KEYS_JSON: &str = include_str!("../manifests/trusted_keys.json");
const SIGNED_MANIFEST_JSON: &str = include_str!("../manifests/signed_manifest.json");
const KEY_ROTATIONS_JSON: &str = include_str!("../manifests/key_rotations.json");

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrustedKey {
    pub key_id: String,
    pub public_key_hex: String,
}

/// A payload plus the signature over its exact bytes. `payload_b64` is base64 of
/// the JSON bytes that were signed; we verify and parse those exact bytes rather
/// than re-serializing, so JSON whitespace/ordering can never break verification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignedDocument {
    pub key_id: String,
    pub algo: String,
    pub signature: String,
    pub payload_b64: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ManifestEntry {
    pub file_id: String,
    pub sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManifestPayload {
    pub version: u32,
    pub created: String,
    pub entries: Vec<ManifestEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RotationPayload {
    pub new_key_id: String,
    pub public_key_hex: String,
    pub note: String,
}

fn generic_err() -> AppError {
    AppError::Message("模型安全校验暂时不可用，请稍后重试或重新安装。".to_string())
}

fn parse_verifying_key(public_key_hex: &str) -> AppResult<VerifyingKey> {
    let bytes = hex::decode(public_key_hex.trim()).map_err(|_| generic_err())?;
    let bytes: [u8; 32] = bytes.as_slice().try_into().map_err(|_| generic_err())?;
    VerifyingKey::from_bytes(&bytes).map_err(|_| generic_err())
}

fn parse_signature(signature_b64: &str) -> AppResult<Signature> {
    let bytes = B64.decode(signature_b64.trim()).map_err(|_| generic_err())?;
    let bytes: [u8; 64] = bytes.as_slice().try_into().map_err(|_| generic_err())?;
    Ok(Signature::from_bytes(&bytes))
}

/// Verify a signed document against a set of trusted keys and return the raw
/// payload bytes on success.
fn verify_document(
    doc: &SignedDocument,
    keys: &HashMap<String, VerifyingKey>,
) -> AppResult<Vec<u8>> {
    if !doc.algo.eq_ignore_ascii_case("ed25519") {
        return Err(AppError::Message(
            "模型安全校验用到的签名算法不受支持。".to_string(),
        ));
    }
    let key = keys.get(&doc.key_id).ok_or_else(|| {
        AppError::Message("模型签名清单的签发者不在信任列表里，请通过官方渠道更新。".to_string())
    })?;
    let payload = B64.decode(doc.payload_b64.trim()).map_err(|_| generic_err())?;
    let signature = parse_signature(&doc.signature)?;
    key.verify_strict(&payload, &signature).map_err(|_| {
        AppError::Message("模型签名清单校验不通过，可能已被篡改，请重新安装。".to_string())
    })?;
    Ok(payload)
}

fn builtin_trusted_keys() -> AppResult<HashMap<String, VerifyingKey>> {
    let entries: Vec<TrustedKey> =
        serde_json::from_str(TRUSTED_KEYS_JSON).map_err(|_| generic_err())?;
    let mut keys = HashMap::new();
    for entry in entries {
        if entry.key_id.trim().is_empty() || entry.key_id == "placeholder" {
            continue;
        }
        keys.insert(entry.key_id, parse_verifying_key(&entry.public_key_hex)?);
    }
    Ok(keys)
}

/// Extend a base key set with any rotation documents each currently-trusted key
/// has signed. A rotation only takes effect if verified against the keys trusted
/// *before* it, so a compromised-but-untrusted key cannot introduce new keys.
fn apply_rotations(
    mut keys: HashMap<String, VerifyingKey>,
    rotations: &[SignedDocument],
) -> HashMap<String, VerifyingKey> {
    for doc in rotations {
        let Ok(payload) = verify_document(doc, &keys) else {
            continue;
        };
        let Ok(rotation) = serde_json::from_slice::<RotationPayload>(&payload) else {
            continue;
        };
        if let Ok(key) = parse_verifying_key(&rotation.public_key_hex) {
            keys.insert(rotation.new_key_id, key);
        }
    }
    keys
}

fn effective_trusted_keys() -> AppResult<HashMap<String, VerifyingKey>> {
    let base = builtin_trusted_keys()?;
    let rotations: Vec<SignedDocument> =
        serde_json::from_str(KEY_ROTATIONS_JSON).unwrap_or_default();
    Ok(apply_rotations(base, &rotations))
}

/// Whether a real signed manifest and trusted key have been provisioned. When
/// false, the bundled files are still placeholders and verification cannot run.
pub fn is_configured() -> bool {
    let manifest_ok = serde_json::from_str::<SignedDocument>(SIGNED_MANIFEST_JSON)
        .map(|doc| doc.key_id != "placeholder" && !doc.payload_b64.trim().is_empty())
        .unwrap_or(false);
    let keys_ok = builtin_trusted_keys().map(|k| !k.is_empty()).unwrap_or(false);
    manifest_ok && keys_ok
}

fn load_manifest() -> AppResult<ManifestPayload> {
    if !is_configured() {
        return Err(AppError::Message(
            "模型签名清单尚未配置，请先补充官方签名信息。".to_string(),
        ));
    }
    let doc: SignedDocument =
        serde_json::from_str(SIGNED_MANIFEST_JSON).map_err(|_| generic_err())?;
    let keys = effective_trusted_keys()?;
    let payload = verify_document(&doc, &keys)?;
    serde_json::from_slice(&payload).map_err(|_| generic_err())
}

/// Verify a downloaded file against the signed manifest: the manifest signature
/// must verify against a trusted key, and the file's SHA256 must match the entry
/// the manifest attests for `file_id`.
pub async fn verify_file(file_id: &str, path: &Path) -> AppResult<()> {
    let manifest = load_manifest()?;
    let entry = manifest
        .entries
        .iter()
        .find(|entry| entry.file_id == file_id)
        .ok_or_else(|| {
            AppError::Message("官方签名清单里没有这个文件的校验信息，请通过官方渠道更新。".to_string())
        })?;

    let actual = crate::hash::file_sha256_hex(path).await?;
    if !crate::hash::matches(&entry.sha256, &actual) {
        return Err(AppError::Message(
            "文件与官方签名清单不一致，可能不完整或被篡改，请重新下载。".to_string(),
        ));
    }
    Ok(())
}

/// The full set of files the signed manifest must attest, assembled from the
/// live model catalog (`models.rs`) and engine packages (`engine.rs`). Both the
/// offline signer and the tests read this, so the signed manifest is generated
/// from the exact hashes the runtime enforces — no hand-copied duplicate list.
pub fn manifest_entries() -> Vec<ManifestEntry> {
    let mut entries: Vec<ManifestEntry> = crate::models::builtin_models()
        .into_iter()
        .map(|model| ManifestEntry {
            file_id: format!("model:{}", model.id),
            sha256: model.sha256,
        })
        .collect();
    for (file_id, sha256) in crate::engine::engine_manifest_entries() {
        entries.push(ManifestEntry { file_id, sha256 });
    }
    entries
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::{Signer, SigningKey};

    /// Development signing seed. NOT a secret — see the module-level security note.
    /// Production must sign offline with a key that never ships in the binary.
    const DEV_SEED: [u8; 32] = [
        0x4c, 0x6f, 0x63, 0x61, 0x6c, 0x41, 0x49, 0x2d, 0x72, 0x6f, 0x6f, 0x74, 0x2d, 0x32, 0x30,
        0x32, 0x36, 0x2d, 0x65, 0x64, 0x32, 0x35, 0x35, 0x31, 0x39, 0x2d, 0x64, 0x65, 0x76, 0x21,
        0x21, 0x21,
    ];
    const KEY_ID: &str = "localai-root-2026";

    fn build_signed_doc(signing: &SigningKey, key_id: &str, payload: &[u8]) -> SignedDocument {
        let signature = signing.sign(payload);
        SignedDocument {
            key_id: key_id.to_string(),
            algo: "ed25519".to_string(),
            signature: B64.encode(signature.to_bytes()),
            payload_b64: B64.encode(payload),
        }
    }

    /// Regenerates the bundled `trusted_keys.json` and `signed_manifest.json` from
    /// DEV_SEED and the known hashes. Run on demand:
    ///   cargo test --lib manifest::tests::generate_signed_manifest_artifacts -- --ignored --nocapture
    #[test]
    #[ignore]
    fn generate_signed_manifest_artifacts() {
        let signing = SigningKey::from_bytes(&DEV_SEED);
        let verifying = signing.verifying_key();
        let payload = ManifestPayload {
            version: 1,
            created: "2026-07-02".to_string(),
            entries: manifest_entries(),
        };
        let payload_bytes = serde_json::to_vec(&payload).unwrap();
        let doc = build_signed_doc(&signing, KEY_ID, &payload_bytes);

        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("manifests");
        std::fs::write(
            dir.join("trusted_keys.json"),
            serde_json::to_string_pretty(&vec![TrustedKey {
                key_id: KEY_ID.to_string(),
                public_key_hex: hex::encode(verifying.to_bytes()),
            }])
            .unwrap(),
        )
        .unwrap();
        std::fs::write(
            dir.join("signed_manifest.json"),
            serde_json::to_string_pretty(&doc).unwrap(),
        )
        .unwrap();
        println!("regenerated manifests for key {}", KEY_ID);
    }

    #[test]
    fn bundled_manifest_is_configured_and_loads() {
        assert!(is_configured(), "run the generator test to populate manifests");
        let manifest = load_manifest().expect("manifest should verify");
        assert!(manifest
            .entries
            .iter()
            .any(|e| e.file_id == "model:qwen3_14b_chat")
            || manifest
                .entries
                .iter()
                .any(|e| e.file_id == "model:ornith_1_0_9b"));
        assert_eq!(manifest.entries.len(), 9);
    }

    #[test]
    fn tampered_payload_is_rejected() {
        let signing = SigningKey::from_bytes(&DEV_SEED);
        let mut keys = HashMap::new();
        keys.insert(KEY_ID.to_string(), signing.verifying_key());

        let payload = serde_json::to_vec(&ManifestPayload {
            version: 1,
            created: "2026-07-02".to_string(),
            entries: manifest_entries(),
        })
        .unwrap();
        let mut doc = build_signed_doc(&signing, KEY_ID, &payload);

        // Untampered verifies.
        assert!(verify_document(&doc, &keys).is_ok());

        // Swap in a different payload without re-signing -> rejected.
        let forged = serde_json::to_vec(&ManifestPayload {
            version: 1,
            created: "2026-07-02".to_string(),
            entries: vec![ManifestEntry {
                file_id: "model:evil".to_string(),
                sha256: "00".repeat(32),
            }],
        })
        .unwrap();
        doc.payload_b64 = B64.encode(&forged);
        assert!(verify_document(&doc, &keys).is_err());
    }

    #[test]
    fn untrusted_signer_is_rejected() {
        let root = SigningKey::from_bytes(&DEV_SEED);
        let attacker = SigningKey::from_bytes(&[9u8; 32]);
        let mut keys = HashMap::new();
        keys.insert(KEY_ID.to_string(), root.verifying_key());

        let payload = serde_json::to_vec(&ManifestPayload {
            version: 1,
            created: "2026-07-02".to_string(),
            entries: manifest_entries(),
        })
        .unwrap();
        // Signed by attacker but claiming the trusted key_id.
        let doc = build_signed_doc(&attacker, KEY_ID, &payload);
        assert!(verify_document(&doc, &keys).is_err());
    }

    #[test]
    fn key_rotation_authorizes_a_new_key() {
        let root = SigningKey::from_bytes(&DEV_SEED);
        let mut base = HashMap::new();
        base.insert(KEY_ID.to_string(), root.verifying_key());

        let new_key = SigningKey::from_bytes(&[42u8; 32]);
        let rotation_payload = serde_json::to_vec(&RotationPayload {
            new_key_id: "localai-root-2027".to_string(),
            public_key_hex: hex::encode(new_key.verifying_key().to_bytes()),
            note: "annual rotation".to_string(),
        })
        .unwrap();

        // Root vouches for the new key -> accepted.
        let rotation_doc = build_signed_doc(&root, KEY_ID, &rotation_payload);
        let extended = apply_rotations(base.clone(), std::slice::from_ref(&rotation_doc));
        assert!(extended.contains_key("localai-root-2027"));

        // A rotation signed by an unknown key is ignored.
        let rogue = SigningKey::from_bytes(&[7u8; 32]);
        let rogue_doc = build_signed_doc(&rogue, "who-dis", &rotation_payload);
        let unchanged = apply_rotations(base, std::slice::from_ref(&rogue_doc));
        assert!(!unchanged.contains_key("localai-root-2027"));
    }

    #[tokio::test]
    async fn verify_file_matches_and_detects_mismatch() {
        use std::io::Write;
        let dir = std::env::temp_dir().join(format!("localai-manifest-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();

        // "hello" -> known sha256; register it under a temp file_id via a locally
        // signed manifest is overkill, so instead assert against the bundled entry
        // by writing a file whose hash equals a manifest entry is impractical.
        // Here we exercise the mismatch path against a real bundled file_id.
        let path = dir.join("blob.bin");
        {
            let mut file = std::fs::File::create(&path).unwrap();
            file.write_all(b"not the real model").unwrap();
        }
        let result = verify_file("model:qwen3_14b_chat", &path).await;
        assert!(result.is_err(), "wrong bytes must fail manifest check");

        // Unknown file_id is rejected too.
        let unknown = verify_file("model:does-not-exist", &path).await;
        assert!(unknown.is_err());

        let _ = std::fs::remove_dir_all(&dir);
    }
}
