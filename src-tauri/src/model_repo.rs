use std::path::Path;

use rusqlite::{params, Connection};

use crate::{
    error::AppResult,
    gguf::GgufMetadata,
    models::ModelMetadata,
};

/// A locally imported model surfaced back to the installed-models list. Only the
/// fields the UI needs are read; the full row is stored for completeness.
#[derive(Debug, Clone)]
pub struct ImportedModelRow {
    pub id: String,
    pub name: String,
    pub local_path: String,
    pub file_size_mb: u64,
}

/// Mirror the bundled catalog into `model_metadata`. Catalog fields are refreshed
/// on every startup, but a row's `local_path` is preserved so runtime install
/// state recorded elsewhere is never clobbered by re-seeding.
pub fn seed_catalog(db_path: &Path, models: &[ModelMetadata]) -> AppResult<()> {
    let connection = Connection::open(db_path)?;
    for model in models {
        upsert(&connection, model, false, false)?;
    }
    Ok(())
}

/// Persist a locally imported `.gguf` so it survives restarts and keeps showing
/// in the installed-models list even after switching to another model.
pub fn upsert_imported_model(
    db_path: &Path,
    id: &str,
    name: &str,
    local_path: &Path,
    file_size_mb: u64,
    vram_required_gb: f64,
    gguf: Option<&GgufMetadata>,
) -> AppResult<()> {
    let model = imported_metadata(id, name, local_path, file_size_mb, vram_required_gb, gguf);
    let connection = Connection::open(db_path)?;
    upsert(&connection, &model, true, true)
}

pub fn list_imported_models(db_path: &Path) -> AppResult<Vec<ImportedModelRow>> {
    let connection = Connection::open(db_path)?;
    let mut statement = connection.prepare(
        "SELECT id, name, local_path, file_size_mb
         FROM model_metadata
         WHERE is_imported = 1 AND local_path IS NOT NULL
         ORDER BY name",
    )?;
    let rows = statement.query_map([], |row| {
        Ok(ImportedModelRow {
            id: row.get(0)?,
            name: row.get(1)?,
            local_path: row.get(2)?,
            file_size_mb: row.get::<_, i64>(3)?.max(0) as u64,
        })
    })?;

    let mut models = Vec::new();
    for row in rows {
        models.push(row?);
    }
    Ok(models)
}

/// Untrack an imported model. Missing rows return Ok — uninstall is idempotent
/// so a stale UI or a second click can't turn into an error the user has to
/// puzzle over. Only rows flagged `is_imported = 1` are touched, so a stray
/// catalog id can never wipe a bundled entry.
pub fn delete_imported_model(db_path: &Path, id: &str) -> AppResult<()> {
    let connection = Connection::open(db_path)?;
    connection.execute(
        "DELETE FROM model_metadata WHERE id = ?1 AND is_imported = 1",
        params![id],
    )?;
    Ok(())
}

/// Insert a model row, or update it on id conflict. `is_imported` flags locally
/// imported models. `overwrite_local_path` controls whether a conflicting row's
/// `local_path` is replaced (true for imports) or kept (false for catalog seeds).
fn upsert(
    connection: &Connection,
    model: &ModelMetadata,
    is_imported: bool,
    overwrite_local_path: bool,
) -> AppResult<()> {
    const CATALOG_SET: &str = "name = excluded.name, author = excluded.author, \
        base_model = excluded.base_model, quant_level = excluded.quant_level, \
        quant_type = excluded.quant_type, quant_bpp = excluded.quant_bpp, \
        parameter_count = excluded.parameter_count, parameters_raw = excluded.parameters_raw, \
        active_parameters = excluded.active_parameters, is_moe = excluded.is_moe, \
        vram_required_gb = excluded.vram_required_gb, file_size_mb = excluded.file_size_mb, \
        context_length = excluded.context_length, recommended_scene = excluded.recommended_scene, \
        chinese_score = excluded.chinese_score, speed_rating = excluded.speed_rating, \
        architecture = excluded.architecture, model_format = excluded.model_format, \
        release_date = excluded.release_date, download_url = excluded.download_url, \
        mirror_urls = excluded.mirror_urls, sha256 = excluded.sha256, \
        supported_gpu = excluded.supported_gpu, license = excluded.license, \
        is_imported = excluded.is_imported";
    let set_clause = if overwrite_local_path {
        format!("{}, local_path = excluded.local_path", CATALOG_SET)
    } else {
        CATALOG_SET.to_string()
    };
    let sql = format!(
        "INSERT INTO model_metadata (
            id, name, author, base_model, quant_level, quant_type, quant_bpp,
            parameter_count, parameters_raw, active_parameters, is_moe, vram_required_gb,
            file_size_mb, context_length, recommended_scene, chinese_score, speed_rating,
            architecture, model_format, release_date, download_url, mirror_urls, sha256,
            supported_gpu, license, is_imported, local_path, supersedes_model_id
         ) VALUES (
            ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17,
            ?18, ?19, ?20, ?21, ?22, ?23, ?24, ?25, ?26, ?27, ?28
         )
         ON CONFLICT(id) DO UPDATE SET {}",
        set_clause
    );

    connection.execute(
        &sql,
        params![
            model.id,
            model.name,
            model.author,
            model.base_model,
            model.quant_level,
            model.quant_type,
            model.quant_bpp,
            model.parameter_count,
            model.parameters_raw as i64,
            model.active_parameters.map(|value| value as i64),
            is_moe_flag(model.is_moe),
            model.vram_required_gb,
            model.file_size_mb as i64,
            model.context_length as i64,
            serde_json::to_string(&model.recommended_scene)?,
            model.chinese_score as i64,
            model.speed_rating as i64,
            model.architecture,
            model.model_format,
            model.release_date,
            model.download_url,
            serde_json::to_string(&model.mirror_urls)?,
            model.sha256,
            serde_json::to_string(&model.supported_gpu)?,
            model.license,
            is_imported as i64,
            model.local_path,
            Option::<String>::None,
        ],
    )?;
    Ok(())
}

fn is_moe_flag(is_moe: bool) -> i64 {
    is_moe as i64
}

/// Build a catalog row for an imported `.gguf`.
///
/// When GGUF metadata is available the real architecture values are used;
/// otherwise every field falls back to a neutral default. The fallbacks matter:
/// before GGUF parsing existed an imported Spark-X2.5-4B reported a 4K context
/// even though the model natively supports 1M, and every import carried the
/// hardcoded 0.4 bytes-per-param that Q4_K_M alone would already beat.
fn imported_metadata(
    id: &str,
    name: &str,
    local_path: &Path,
    file_size_mb: u64,
    vram_required_gb: f64,
    gguf: Option<&GgufMetadata>,
) -> ModelMetadata {
    let 参数量 = gguf.map(|数据| 数据.parameter_count).unwrap_or(0);
    // 实测每参数字节数，替代此前写死的 0.4 —— 真实的 Q4_K_M 约 0.61。
    let 实测bpp = gguf
        .and_then(|数据| 数据.bytes_per_param(file_size_mb.saturating_mul(1_000_000)))
        .unwrap_or(0.0);
    let 上下文长度 = gguf
        .and_then(|数据| 数据.context_length)
        .map(u64::from)
        .unwrap_or(4096);
    let 架构 = gguf
        .map(|数据| 数据.architecture.clone())
        .filter(|值| !值.is_empty())
        .unwrap_or_else(|| "unknown".to_string());
    let 量化档 = gguf
        .and_then(|数据| 数据.file_type_label())
        .unwrap_or("unknown");

    ModelMetadata {
        id: id.to_string(),
        name: name.to_string(),
        author: "本地导入".to_string(),
        base_model: gguf
            .and_then(|数据| 数据.name.clone())
            .unwrap_or_else(|| name.to_string()),
        display_compression: 压缩档说明(量化档),
        quant_level: 量化档.to_string(),
        quant_type: format!("GGUF/{量化档}"),
        quant_bpp: 实测bpp,
        parameter_count: gguf
            .and_then(|数据| 数据.parameter_label())
            .unwrap_or_default(),
        parameters_raw: 参数量,
        active_parameters: gguf.and_then(|数据| 数据.active_parameters()),
        is_moe: gguf.map(|数据| 数据.is_moe()).unwrap_or(false),
        vram_required_gb,
        file_size_mb,
        context_length: 上下文长度,
        recommended_scene: Vec::new(),
        chinese_score: 0,
        speed_rating: 0,
        architecture: 架构,
        model_format: "gguf".to_string(),
        release_date: String::new(),
        download_url: String::new(),
        mirror_urls: Vec::new(),
        sha256: String::new(),
        supported_gpu: Vec::new(),
        license: String::new(),
        is_installed: true,
        local_path: Some(local_path.to_string_lossy().to_string()),
    }
}

/// 量化档位到界面中文说明的映射。
fn 压缩档说明(量化档: &str) -> String {
    match 量化档 {
        "F32" | "F16" | "BF16" => "未压缩（原始精度）".to_string(),
        "Q8_0" => "高质量压缩版".to_string(),
        "Q6_K" | "Q5_K_M" | "Q5_K_S" => "较高质量压缩版".to_string(),
        "Q4_K_M" | "Q4_K_S" | "Q4_0" | "Q4_1" => "普通压缩版".to_string(),
        "unknown" => "本地模型文件".to_string(),
        _ => "压缩版".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{database::init_database, models::builtin_models};

    fn temp_db() -> (std::path::PathBuf, std::path::PathBuf) {
        let dir = std::env::temp_dir().join(format!("localai-modelrepo-{}", uuid::Uuid::new_v4()));
        let db = init_database(&dir).expect("init database");
        (dir, db)
    }

    fn catalog_count(db: &Path) -> i64 {
        let conn = Connection::open(db).unwrap();
        conn.query_row("SELECT COUNT(*) FROM model_metadata", [], |row| row.get(0))
            .unwrap()
    }

    fn local_path_of(db: &Path, id: &str) -> Option<String> {
        let conn = Connection::open(db).unwrap();
        conn.query_row(
            "SELECT local_path FROM model_metadata WHERE id = ?1",
            [id],
            |row| row.get::<_, Option<String>>(0),
        )
        .unwrap()
    }

    #[test]
    fn seeding_is_idempotent_and_excludes_imported() {
        let (dir, db) = temp_db();
        let catalog = builtin_models();

        seed_catalog(&db, &catalog).unwrap();
        seed_catalog(&db, &catalog).unwrap(); // re-seed must not duplicate rows

        assert_eq!(catalog_count(&db), catalog.len() as i64);
        // Seeded catalog rows are not imported.
        assert!(list_imported_models(&db).unwrap().is_empty());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn imported_models_round_trip_and_upsert() {
        let (dir, db) = temp_db();
        let path = dir.join("my-model.gguf");

        upsert_imported_model(&db, "imported-my-model", "我的模型", &path, 4200, 5.0, None).unwrap();
        let rows = list_imported_models(&db).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].id, "imported-my-model");
        assert_eq!(rows[0].name, "我的模型");
        assert_eq!(rows[0].file_size_mb, 4200);

        // Re-importing the same id updates in place instead of duplicating.
        upsert_imported_model(
            &db,
            "imported-my-model",
            "我的模型-改名",
            &path,
            4300,
            5.0,
            None,
        )
        .unwrap();
        let rows = list_imported_models(&db).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].name, "我的模型-改名");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn imported_model_uses_real_gguf_parameters_when_available() {
        // 回归测试：导入时若解析出 GGUF 元数据，架构 / 参数量 / 原生上下文 /
        // 实测 bpp 都应来自真实值，而不是此前写死的 unknown / 0 / 4096 / 0.4。
        let (dir, db) = temp_db();
        let path = dir.join("ornith.gguf");

        let 元数据 = GgufMetadata {
            architecture: "qwen35".to_string(),
            name: Some("Ornith-1.5-9B".to_string()),
            parameter_count: 9_197_000_000,
            file_type: Some(15),
            block_count: Some(33),
            context_length: Some(262_144),
            head_count_kv: Some(4),
            key_length: Some(256),
            value_length: Some(256),
            full_attention_interval: Some(4),
            ..Default::default()
        };

        upsert_imported_model(
            &db,
            "imported-ornith",
            "Ornith",
            &path,
            5383,
            7.0,
            Some(&元数据),
        )
        .unwrap();

        let connection = Connection::open(&db).unwrap();
        let (架构, 参数量, 上下文, 每参数字节): (String, i64, i64, f64) = connection
            .query_row(
                "SELECT architecture, parameters_raw, context_length, quant_bpp \
                 FROM model_metadata WHERE id = ?1",
                params!["imported-ornith"],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
            )
            .unwrap();

        assert_eq!(架构, "qwen35");
        assert_eq!(参数量, 9_197_000_000);
        assert_eq!(上下文, 262_144, "原生的 256K 上下文不应被写成 4096");
        // 5383 MB ÷ 9.197B 参数 ≈ 0.585 字节/参数
        assert!((每参数字节 - 0.585).abs() < 0.01, "实际为 {每参数字节}");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn delete_imported_model_is_idempotent_and_scoped() {
        let (dir, db) = temp_db();

        // Seed catalog + import one file, then confirm both rows exist.
        let catalog = builtin_models();
        seed_catalog(&db, &catalog).unwrap();
        let path = dir.join("mine.gguf");
        upsert_imported_model(&db, "imported-mine", "我的模型", &path, 1200, 3.0, None).unwrap();
        assert_eq!(list_imported_models(&db).unwrap().len(), 1);

        // Removing the imported id drops just that row.
        delete_imported_model(&db, "imported-mine").unwrap();
        assert!(list_imported_models(&db).unwrap().is_empty());
        // Catalog rows are untouched.
        assert_eq!(catalog_count(&db), catalog.len() as i64);

        // Second delete is a no-op, not an error.
        delete_imported_model(&db, "imported-mine").unwrap();

        // A catalog id must never be wiped even if passed here by mistake — the
        // `is_imported = 1` guard is the whole point of the API.
        let seeded_id = catalog[0].id.clone();
        delete_imported_model(&db, &seeded_id).unwrap();
        assert_eq!(catalog_count(&db), catalog.len() as i64);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn reseeding_preserves_runtime_local_path() {
        let (dir, db) = temp_db();
        let catalog = builtin_models();
        seed_catalog(&db, &catalog).unwrap();

        // Simulate a runtime install recording a local path on a catalog row.
        let id = &catalog[0].id;
        let conn = Connection::open(&db).unwrap();
        conn.execute(
            "UPDATE model_metadata SET local_path = ?2 WHERE id = ?1",
            params![id, "C:/models/x/model.gguf"],
        )
        .unwrap();
        drop(conn);

        // Re-seeding refreshes catalog fields but must keep the local path.
        seed_catalog(&db, &catalog).unwrap();
        assert_eq!(
            local_path_of(&db, id).as_deref(),
            Some("C:/models/x/model.gguf")
        );

        let _ = std::fs::remove_dir_all(&dir);
    }
}
