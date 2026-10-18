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

/// 按 id 取回一条模型的**完整元数据**（内置与本地导入都算），查不到返回 `None`。
///
/// 🔴 **为什么需要它（P2-11，2026-10-16 实机发现）**：启动时要按模型算出
/// `-ngl` 与上下文（`service.rs::start_service_core`），而那里的数据源
/// `catalog::effective_models` 只由「内置清单 + 在线清单缓存」拼成，
/// **从不含 `is_imported = 1` 的导入模型**。导入模型的 id 形如
/// `local-<文件主名>`，在那里必然查不到 ⇒ 模型元数据恒为 `None` ⇒ 启动参数
/// 恒落到 `DEFAULT_CUDA_PARAMS`（`24` 层 / `4096` 上下文）：P100 的 16 GB 上
/// 只用掉 6424 MiB，Ornith-1.5-9B 的 256K 原生上下文被钉死在 4K。
/// 本函数就是那条链路的**回落数据源**。
///
/// 与 `list_imported_models` 的两点区别都是**有意的**，不要「顺手统一」：
/// 1. **不过滤 `is_imported`** —— 内置模型同样要能取到，否则回落没有意义；
/// 2. **返回全部字段** —— 启动参数要用到 `file_size_mb`、`context_length`、
///    `active_parameters`、`parameters_raw`、`quant_bpp`、`local_path`。
///
/// 两个派生字段的取法：
/// * `display_compression` 表里不存，现由 `quant_level` 算，保证与写入侧同一口径；
/// * `is_installed` 按 `local_path` 是否存在给值（与 `recommend.rs` 按磁盘存在性
///   判定的口径一致；启动参数不读这个字段）。
pub fn find_model(db_path: &Path, id: &str) -> AppResult<Option<ModelMetadata>> {
    let connection = Connection::open(db_path)?;
    let mut statement = connection.prepare(
        "SELECT id, name, author, base_model, quant_level, quant_type, quant_bpp,
                parameter_count, parameters_raw, active_parameters, is_moe,
                vram_required_gb, file_size_mb, context_length, recommended_scene,
                chinese_score, speed_rating, architecture, model_format, release_date,
                download_url, mirror_urls, sha256, supported_gpu, license, local_path
         FROM model_metadata WHERE id = ?1",
    )?;
    // 先精确、再试历史写法（见 `候选标识`）。老库里的导入行是**旧版导入流程**
    // 写的 `imported-…`，而调用方拿到的是扫描侧的 `local-…` —— 不试候选，
    // 那些行**永远查不到**，等于代码改完、老机器上问题照旧（2026-10-16 实测）。
    for 候选 in 候选标识(id) {
        // 用 `next()` 而不是 `query_row`：查不到是**正常结果**（未知 id），
        // 不是错误 —— 把它变成 Err 会让「模型不在库里」和「库读坏了」同形。
        let mut rows = statement.query(params![候选])?;
        if let Some(row) = rows.next()? {
            // id 归成**调用方给的那个**：命中历史写法时不能让上层突然收到另一个
            // 名字，否则日志/状态里换名会被误读成「换了模型」。
            return Ok(Some(读取一行模型元数据(id, row)?));
        }
    }
    Ok(None)
}

/// 同一个导入模型可能存在的**两种历史 id 写法**，按优先级给出候选。
///
/// 🔴 **为什么需要这层兼容**（P2-11 续，2026-10-16 实机抓出）：「导入模型 id」
/// 此前有**两份生成规则** —— 文件扫描侧（UI 真正拿到的那个）写 `local-<主名>`，
/// 导入写库侧写 `imported-<主名>`。于是「按 UI 给的 id 回库查元数据」在两台实机
/// 上都必然落空，显存反解与 KV 精度在导入模型上恒不生效。
///
/// 新写入已统一到 `models::导入模型标识`（`local-…`），但**老库里已经躺着
/// `imported-…` 的行**（P100 上的 `imported-Ornith-1.5-9B-Q8_0` 就是）——
/// 没有这一层，老机器上「修完代码问题照旧」。等老库自然淘汰后可整块删掉。
///
/// 只在**已知的导入命名空间**里互换前缀，不给任意 id 编别名 ——
/// 否则一个巧合的名字就能误命中别人的行。
pub fn 候选标识(id: &str) -> Vec<String> {
    let mut 候选 = vec![id.to_string()];
    let 别名 = id
        .strip_prefix("local-")
        .map(|主名| ("imported-", 主名))
        .or_else(|| id.strip_prefix("imported-").map(|主名| ("local-", 主名)));
    if let Some((前缀, 主名)) = 别名 {
        // 主名为空（如 id 恰好是 `local-`）不编别名，避免变成前缀泛匹配。
        if !主名.is_empty() {
            候选.push(format!("{前缀}{主名}"));
        }
    }
    候选
}

/// 把 `model_metadata` 的一行还原成 `ModelMetadata`。
///
/// 列按**名字**取而不是序号 —— 序号在 SELECT 与这里两处各写一遍，改一处忘了
/// 另一处不报错、只给错值；名字由 SQLite 自己核对，对不上当场失败。
fn 读取一行模型元数据(id: &str, row: &rusqlite::Row<'_>) -> rusqlite::Result<ModelMetadata> {
    let 量化档: String = row.get("quant_level")?;
    let 推荐场景: String = row.get("recommended_scene")?;
    let 镜像地址: String = row.get("mirror_urls")?;
    let 支持显卡: String = row.get("supported_gpu")?;
    let 本地路径: Option<String> = row.get("local_path")?;
    // SQLite 没有无符号整数：负值只可能来自手工改坏的库，按「无」处理比
    // 回绕成一个天文数字安全（后者会被当真实参数量算进 KV）。
    let 活跃参数量: Option<i64> = row.get("active_parameters")?;

    Ok(ModelMetadata {
        id: id.to_string(),
        name: row.get("name")?,
        author: row.get("author")?,
        base_model: row.get("base_model")?,
        display_compression: 压缩档说明(&量化档),
        quant_level: 量化档,
        quant_type: row.get("quant_type")?,
        quant_bpp: row.get("quant_bpp")?,
        parameter_count: row.get("parameter_count")?,
        parameters_raw: row.get::<_, i64>("parameters_raw")?.max(0) as u64,
        active_parameters: 活跃参数量.and_then(|值| u64::try_from(值).ok()),
        is_moe: row.get::<_, i64>("is_moe")? != 0,
        vram_required_gb: row.get("vram_required_gb")?,
        file_size_mb: row.get::<_, i64>("file_size_mb")?.max(0) as u64,
        context_length: row.get::<_, i64>("context_length")?.max(0) as u64,
        recommended_scene: 解析json列(&推荐场景, id, "recommended_scene"),
        chinese_score: row.get::<_, i64>("chinese_score")?.clamp(0, 100) as u8,
        speed_rating: row.get::<_, i64>("speed_rating")?.clamp(0, 100) as u8,
        architecture: row.get("architecture")?,
        model_format: row.get("model_format")?,
        release_date: row.get("release_date")?,
        download_url: row.get("download_url")?,
        mirror_urls: 解析json列(&镜像地址, id, "mirror_urls"),
        sha256: row.get("sha256")?,
        supported_gpu: 解析json列(&支持显卡, id, "supported_gpu"),
        license: row.get("license")?,
        is_installed: 本地路径.is_some(),
        local_path: 本地路径,
    })
}

/// 反序列化一个 JSON 数组列。
///
/// 🔴 解析失败**不让整条查询失败**，但也**不许静默**：这三列只供界面展示，
/// 缺了不影响启动参数，为一条坏 JSON 把「模型能启动」变成「启动报错」是错的
/// 权衡；可一旦连日志都不留，就再也查不出库里脏了（本项目的铁律：
/// 降级必须与「正常」可区分）。
fn 解析json列<T: serde::de::DeserializeOwned + Default>(
    原始: &str,
    id: &str,
    列名: &str,
) -> T {
    serde_json::from_str(原始).unwrap_or_else(|err| {
        tracing::warn!(
            model_id = %id,
            column = %列名,
            error = %err,
            "模型元数据里的 JSON 列无法解析，已按空值处理（不影响启动参数）"
        );
        T::default()
    })
}

/// Untrack an imported model. Missing rows return Ok — uninstall is idempotent
/// so a stale UI or a second click can't turn into an error the user has to
/// puzzle over. Only rows flagged `is_imported = 1` are touched, so a stray
/// catalog id can never wipe a bundled entry.
///
/// 候选 id 都删一遍：卸载传进来的是 **UI 手里的 id**（`local-…`），而老库里
/// 那个行是旧版流程写的 `imported-…` —— 只按精确匹配删的话，用户点了卸载、
/// 文件删了、行却留在库里；本函数是幂等的，多删一个不存在的 id 不会有副作用。
pub fn delete_imported_model(db_path: &Path, id: &str) -> AppResult<()> {
    let connection = Connection::open(db_path)?;
    for 候选 in 候选标识(id) {
        connection.execute(
            "DELETE FROM model_metadata WHERE id = ?1 AND is_imported = 1",
            params![候选],
        )?;
    }
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

/// 从磁盘上的 `.gguf` **现场**构造模型元数据（P2-11 续二，2026-10-16 实机抓出）。
///
/// 🔴 **为什么需要第三级**：`查找启动模型` 的前两级（清单 / 数据库）都建立在
/// 同一个**假设**上 —— 「这台机器上曾经有人把这个模型写进过库」。而真实用法
/// 是：把 `.gguf` 丢进 `models/<目录>/`，再由界面扫描发现它。
/// `installed_model_info`（models.rs）会为它算出一个 id（`local-<主名>`），
/// UI 也拿得到，**但没有任何一处把它写进库** —— `switch_installed_model`
/// 只设置路径与当前模型指针，不落库。
///
/// 于是「按 UI 给的 id 回库查元数据」必然落空，启动参数退回兜底默认值。
/// P100 的日志就是这一形态：模型能跑、id 有、库里查不到，日志里连一条
/// 失败提示都没有（`find_model` 查不到是**正常结果**，不是错误）。
///
/// 这一级把**文件本身**当作最后的事实来源：只要 `.gguf` 在磁盘上，就能拿到
/// 真实文件大小与 GGUF 头里的层结构 / 原生上下文 —— 而这三样正是启动参数
/// 计算的全部输入（见 service.rs 的 `cuda_launch_params`）。
///
/// 构造逻辑**复用 `imported_metadata`**（与导入写库时逐字段同一份），不另起
/// 一套：否则同一份元数据会有两处构造，改一处、另一处不报错、只给错值。
pub fn 从文件读取模型元数据(id: &str, path: &Path) -> Option<ModelMetadata> {
    // 大小用与 `installed_model_info` 相同的向上取整口径（十进制 MB），
    // 保证「磁盘兜底」与「扫描侧」算出的 file_size_mb 一字不差。
    let 文件大小mb = (std::fs::metadata(path).ok()?.len() + 999_999) / 1_000_000;
    let 主名 = path.file_stem().and_then(|名| 名.to_str())?.to_string();
    // 解析失败不阻断：异常/占位文件仍能得到「大小对、上下文取兜底」的元数据，
    // 比直接返回 None 好 —— 后者会让这条链路又退回 24 层 / 4096。
    let gguf = crate::gguf::parse_file(path).ok();
    // `vram_required_gb` 只供界面「这台机器能不能跑」的门槛用，启动参数不读它；
    // 这里给 0.0 而不是现算，是为了不把 `models::estimate_vram_gb` 反向引进来。
    Some(imported_metadata(
        id,
        &主名,
        path,
        文件大小mb,
        0.0,
        gguf.as_ref(),
    ))
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
    fn find_model_reads_both_imported_and_catalog_rows() {
        // P2-11：启动时的模型元数据回落源。三条判据 —— 导入行取得到、
        // 内置行取得到且**字段一致**、未知 id 返回 None 而不是报错。
        let (dir, db) = temp_db();
        let catalog = builtin_models();
        seed_catalog(&db, &catalog).unwrap();
        let path = dir.join("mine.gguf");
        // id 由**唯一规则**派生、不手抄字面量 —— P2-11 续的教训正是「测试自己造了
        // 一个真实代码从不写的 id，于是全绿而实机照旧」。
        let 导入id = crate::models::导入模型标识("mine");
        upsert_imported_model(&db, &导入id, "我的模型", &path, 4342, 6.0, None).unwrap();

        let 导入 = find_model(&db, &导入id)
            .unwrap()
            .expect("导入模型应能取到");
        assert_eq!(导入.name, "我的模型");
        assert_eq!(导入.file_size_mb, 4342);
        assert_eq!(导入.author, "本地导入");
        assert_eq!(导入.local_path, Some(path.to_string_lossy().to_string()));
        assert!(导入.is_installed, "有 local_path 就算已安装");
        // 无 GGUF 元数据 ⇒ quant_level 是 unknown，压缩说明按同一函数派生
        assert_eq!(导入.display_compression, "本地模型文件");

        let 内置 = find_model(&db, &catalog[0].id)
            .unwrap()
            .expect("内置模型也应能取到 —— 否则回落对内置模型没有意义");
        assert_eq!(内置.name, catalog[0].name);
        assert_eq!(内置.file_size_mb, catalog[0].file_size_mb);
        assert_eq!(内置.parameters_raw, catalog[0].parameters_raw);
        assert_eq!(内置.active_parameters, catalog[0].active_parameters);
        assert_eq!(内置.context_length, catalog[0].context_length);
        assert_eq!(内置.quant_bpp, catalog[0].quant_bpp);
        assert_eq!(内置.is_moe, catalog[0].is_moe);
        assert_eq!(内置.local_path, catalog[0].local_path);
        // 三个 JSON 列必须原样还原（写进去的是 serde_json，读回来是同一个类型）
        assert_eq!(内置.recommended_scene, catalog[0].recommended_scene);
        assert_eq!(内置.mirror_urls, catalog[0].mirror_urls);
        assert_eq!(内置.supported_gpu, catalog[0].supported_gpu);

        // 未知 id 是**正常结果**，不是错误。
        assert!(find_model(&db, "根本不存在的模型").unwrap().is_none());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn 候选标识_只在两种导入写法之间互换() {
        assert_eq!(
            候选标识("local-x"),
            vec!["local-x".to_string(), "imported-x".to_string()]
        );
        assert_eq!(
            候选标识("imported-x"),
            vec!["imported-x".to_string(), "local-x".to_string()]
        );
        // 内置模型的 id 不编别名 —— 否则一个巧合的名字就能误命中别人的行。
        assert_eq!(候选标识("qwen3_1_7b_light"), vec!["qwen3_1_7b_light".to_string()]);
        // 主名为空时不编别名，避免退化成「前缀泛匹配」。
        assert_eq!(候选标识("local-"), vec!["local-".to_string()]);
    }

    #[test]
    fn find_model_对老库里的历史写法导入行仍能命中() {
        // 复刻 P100 的真实库：行是**旧版导入流程**写的 `imported-…`，
        // 而调用方（UI / 文件扫描侧）手里是 `local-…`。不加兼容层的话，这条
        // 必然返回 None ⇒ 启动参数退回 24 层 / 4096（修完代码、老机器照旧）。
        let (dir, db) = temp_db();
        let path = dir.join("Ornith-1.5-9B-Q8_0.gguf");
        upsert_imported_model(
            &db,
            "imported-Ornith-1.5-9B-Q8_0",
            "Ornith-1.5-9B-Q8_0",
            &path,
            9785,
            11.0,
            None,
        )
        .unwrap();

        let 命中 = find_model(&db, "local-Ornith-1.5-9B-Q8_0")
            .unwrap()
            .expect("老行必须能按新写法查到");
        assert_eq!(命中.file_size_mb, 9785);
        assert_eq!(
            命中.id, "local-Ornith-1.5-9B-Q8_0",
            "命中历史写法后 id 归成调用方给的那个"
        );

        // 反向也成立：库是新写法、查询给旧写法。
        let 另一个 = dir.join("other.gguf");
        upsert_imported_model(&db, "local-other", "other", &另一个, 1234, 3.0, None).unwrap();
        assert_eq!(
            find_model(&db, "imported-other").unwrap().unwrap().file_size_mb,
            1234
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn delete_imported_model_对历史写法行也能删掉() {
        // 卸载传进来的是 **UI 手里的 id**（`local-…`），老库里那行是 `imported-…`；
        // 只按精确匹配删 ⇒ 用户点了卸载、文件删了、行却还留在库里。
        let (dir, db) = temp_db();
        let path = dir.join("Ornith-1.5-9B-Q8_0.gguf");
        upsert_imported_model(&db, "imported-Ornith-1.5-9B-Q8_0", "Ornith", &path, 9785, 11.0, None)
            .unwrap();
        assert_eq!(list_imported_models(&db).unwrap().len(), 1);

        delete_imported_model(&db, "local-Ornith-1.5-9B-Q8_0").unwrap();
        assert!(
            list_imported_models(&db).unwrap().is_empty(),
            "老写法那一行没被删掉"
        );

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
