use std::{
    fs::File,
    io::{self, Read},
    path::Path,
};

use sha2::{Digest, Sha256};

use crate::error::{AppError, AppResult};

const CHUNK_SIZE: usize = 1024 * 1024;

/// Whether an expected hash from the model/engine manifest is usable.
/// Missing or `placeholder` values are treated as "not configured".
pub fn is_configured(expected: &str) -> bool {
    !expected.trim().is_empty() && expected != "placeholder"
}

pub fn matches(expected: &str, actual: &str) -> bool {
    actual.eq_ignore_ascii_case(expected)
}

/// Hash a file in fixed-size chunks so multi-GB models never load fully into
/// memory. Runs the blocking file read on a dedicated thread.
pub async fn file_sha256_hex(path: &Path) -> AppResult<String> {
    let path = path.to_path_buf();
    tokio::task::spawn_blocking(move || sha256_hex_blocking(&path))
        .await
        .map_err(|_| AppError::Message("文件校验暂时没能完成，请稍后重试。".to_string()))?
        .map_err(AppError::from)
}

fn sha256_hex_blocking(path: &Path) -> io::Result<String> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; CHUNK_SIZE];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hex::encode(hasher.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn configuration_and_matching_rules() {
        assert!(!is_configured(""));
        assert!(!is_configured("   "));
        assert!(!is_configured("placeholder"));
        assert!(is_configured("abc123"));

        assert!(matches("ABCD", "abcd"));
        assert!(!matches("abcd", "abce"));
    }

    #[tokio::test]
    async fn hashes_file_in_chunks() {
        // sha256("hello") is a well-known constant.
        let dir = std::env::temp_dir().join(format!("localai-hash-test-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("blob.bin");
        {
            let mut file = File::create(&path).unwrap();
            file.write_all(b"hello").unwrap();
        }

        let actual = file_sha256_hex(&path).await.unwrap();
        assert_eq!(
            actual,
            "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
        );

        // Larger-than-one-chunk input still hashes correctly.
        let big_path = dir.join("big.bin");
        let payload = vec![7u8; CHUNK_SIZE * 2 + 123];
        {
            let mut file = File::create(&big_path).unwrap();
            file.write_all(&payload).unwrap();
        }
        let big_hash = file_sha256_hex(&big_path).await.unwrap();
        let expected = hex::encode(Sha256::digest(&payload));
        assert_eq!(big_hash, expected);

        let _ = std::fs::remove_dir_all(&dir);
    }
}
