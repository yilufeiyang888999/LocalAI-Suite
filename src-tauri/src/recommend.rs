use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::{
    app_state::AppState,
    hardware::{estimate_tokens_per_second, GpuVendor, HardwareInfo},
    models::{is_model_installed, model_install_path, ModelMetadata, Scene},
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecommendedModel {
    pub model: ModelMetadata,
    pub rank: RecommendationRank,
    pub score: f64,
    pub reason: String,
    pub speed_label: String,
    pub quality_label: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RecommendationRank {
    Primary,
    Alternative,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecommendRequest {
    pub scene: Scene,
    pub hardware: HardwareInfo,
    pub manual_vram_gb: Option<f64>,
}

#[tauri::command]
pub fn get_recommend_models(
    request: RecommendRequest,
    state: tauri::State<'_, AppState>,
) -> Vec<RecommendedModel> {
    let (install_dir, data_dir) = {
        let inner = state.inner.lock().expect("state lock poisoned");
        (inner.settings.install_dir.clone(), inner.data_dir.clone())
    };
    let catalog = crate::catalog::effective_models(&data_dir);
    let scene = request.scene.clone();
    let results = recommend_models(request, &install_dir, &catalog);
    let ids: Vec<&str> = results.iter().map(|item| item.model.id.as_str()).collect();
    tracing::info!(scene = ?scene, count = results.len(), models = ?ids, "computed model recommendations");
    results
}

pub fn recommend_models(
    request: RecommendRequest,
    install_dir: &Path,
    catalog: &[ModelMetadata],
) -> Vec<RecommendedModel> {
    let usable_vram = request.manual_vram_gb.unwrap_or_else(|| {
        if request.hardware.has_gpu {
            request.hardware.gpu_vram_gb
        } else {
            3.0
        }
    });

    let cpu_only = !request.hardware.has_gpu && request.manual_vram_gb.is_none();
    let max_required = if cpu_only { 2.8 } else { usable_vram };
    let gpu_bandwidth = primary_gpu_bandwidth(&request.hardware);

    let mut scored: Vec<_> = catalog
        .iter()
        .cloned()
        .filter(|model| model.vram_required_gb <= max_required)
        .map(|model| {
            let scene_match = if model.recommended_scene.contains(&request.scene) {
                35.0
            } else {
                5.0
            };
            let fit = fit_score(usable_vram, model.vram_required_gb);
            let quality = quality_score(&model);
            let speed = speed_value(&model, gpu_bandwidth);
            let score = scene_match + fit * 0.25 + quality * 0.25 + speed * 0.15;
            (model, score)
        })
        .collect();

    if scored.is_empty() {
        scored = catalog
            .iter()
            .cloned()
            .filter(|model| model.vram_required_gb <= 2.5)
            .map(|model| (model, 50.0))
            .collect();
    }

    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored
        .into_iter()
        .take(3)
        .enumerate()
        .map(|(index, (model, score))| {
            let rank = if index == 0 {
                RecommendationRank::Primary
            } else {
                RecommendationRank::Alternative
            };
            let reason = reason_for(&model, &request.scene, usable_vram);
            let speed_label = speed_label_for(&model, gpu_bandwidth);
            let quality_label = quality_label(model.chinese_score);
            let mut model = model;
            if is_model_installed(install_dir, &model) {
                model.is_installed = true;
                model.local_path =
                    Some(model_install_path(install_dir, &model).to_string_lossy().to_string());
            } else {
                model.is_installed = false;
                model.local_path = None;
            }
            RecommendedModel {
                reason,
                speed_label,
                quality_label,
                model,
                rank,
                score: (score * 10.0).round() / 10.0,
            }
        })
        .collect()
}

fn fit_score(usable_vram: f64, required: f64) -> f64 {
    if usable_vram <= 0.0 {
        return 45.0;
    }
    let ratio = required / usable_vram;
    if ratio <= 0.5 {
        75.0
    } else if ratio <= 0.8 {
        100.0
    } else if ratio <= 0.9 {
        70.0
    } else {
        50.0
    }
}

fn quality_score(model: &ModelMetadata) -> f64 {
    let size_bonus = if model.parameters_raw >= 14_000_000_000 {
        20.0
    } else if model.parameters_raw >= 7_000_000_000 {
        14.0
    } else {
        8.0
    };
    f64::from(model.chinese_score) * 0.7 + size_bonus
}

fn reason_for(model: &ModelMetadata, scene: &Scene, usable_vram: f64) -> String {
    let scene_text = scene.label();
    if usable_vram <= 3.0 {
        return format!(
            "当前按 CPU 模式推荐，优先选择资源占用低的模型。这个模型预计需要约 {:.1}GB 显卡内存等效资源，适合先完成简单问答体验。",
            model.vram_required_gb
        );
    }
    if model.recommended_scene.contains(scene) {
        format!(
            "推荐此模型是因为它适合{}，预计需要约 {:.1}GB 显卡内存，你的电脑可以尝试运行。",
            scene_text, model.vram_required_gb
        )
    } else {
        format!(
            "这个模型资源占用较低，预计需要约 {:.1}GB 显卡内存，适合先完成本地AI体验。",
            model.vram_required_gb
        )
    }
}

/// Bandwidth (GB/s) of the GPU the service would actually use. MVP auto-uses the
/// fastest NVIDIA card, so pick the max bandwidth among NVIDIA GPUs; `None` when
/// there's no NVIDIA card or the model isn't in the bandwidth table.
fn primary_gpu_bandwidth(hardware: &HardwareInfo) -> Option<u32> {
    hardware
        .gpu_list
        .iter()
        .filter(|gpu| matches!(gpu.vendor, GpuVendor::NVIDIA))
        .filter_map(|gpu| gpu.bandwidth_gbs)
        .max()
}

/// Resident weight size (GB) read per decoded token. The GGUF file size is the
/// best proxy for dense models; MoE models only read the active expert fraction.
fn model_weight_gb(model: &ModelMetadata) -> f64 {
    let file_gb = model.file_size_mb as f64 / 1024.0;
    match model.active_parameters {
        Some(active) if model.is_moe && model.parameters_raw > 0 => {
            file_gb * (active as f64 / model.parameters_raw as f64)
        }
        _ => file_gb,
    }
}

/// Estimated decode speed (tokens/sec) when GPU bandwidth is known, else `None`.
fn estimated_tps(model: &ModelMetadata, bandwidth: Option<u32>) -> Option<f64> {
    let bandwidth = bandwidth?;
    let tps = estimate_tokens_per_second(bandwidth, model_weight_gb(model));
    (tps > 0.0).then_some(tps)
}

/// Speed dimension in the same 0..50 range as the legacy `speed_rating * 10`, so
/// weighting stays balanced. Uses the bandwidth-based tok/s estimate when
/// available (40 t/s ≈ full marks), otherwise falls back to the static rating.
fn speed_value(model: &ModelMetadata, bandwidth: Option<u32>) -> f64 {
    match estimated_tps(model, bandwidth) {
        Some(tps) => (tps / 40.0 * 50.0).clamp(0.0, 50.0),
        None => f64::from(model.speed_rating) * 10.0,
    }
}

fn speed_label_for(model: &ModelMetadata, bandwidth: Option<u32>) -> String {
    match estimated_tps(model, bandwidth) {
        Some(tps) if tps >= 30.0 => "快".to_string(),
        Some(tps) if tps >= 15.0 => "中".to_string(),
        Some(_) => "慢".to_string(),
        None => speed_label(model.speed_rating),
    }
}

fn speed_label(speed: u8) -> String {
    match speed {
        4..=5 => "快".to_string(),
        3 => "中".to_string(),
        _ => "慢".to_string(),
    }
}

fn quality_label(score: u8) -> String {
    match score {
        90..=100 => "好".to_string(),
        75..=89 => "中".to_string(),
        _ => "一般".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::builtin_models;

    fn light_model() -> ModelMetadata {
        // The 1.5B model: small file, high static speed_rating.
        builtin_models()
            .into_iter()
            .min_by_key(|m| m.file_size_mb)
            .expect("at least one builtin model")
    }

    #[test]
    fn speed_value_falls_back_to_rating_without_bandwidth() {
        let model = light_model();
        assert_eq!(
            speed_value(&model, None),
            f64::from(model.speed_rating) * 10.0
        );
        assert_eq!(speed_label_for(&model, None), speed_label(model.speed_rating));
    }

    #[test]
    fn speed_value_uses_bandwidth_estimate_when_available() {
        let model = light_model();
        // A ~1.1GB model on a 448 GB/s card decodes far above 40 t/s, so the
        // bandwidth path should saturate the 0..50 speed dimension.
        let fast = speed_value(&model, Some(448));
        assert_eq!(fast, 50.0);
        assert_eq!(speed_label_for(&model, Some(448)), "快");

        // A very slow card drags the same model below full marks.
        let slow = speed_value(&model, Some(50));
        assert!(slow < 50.0, "expected throttled speed, got {slow}");
    }

    #[test]
    fn moe_weight_uses_active_fraction() {
        let mut model = light_model();
        model.is_moe = true;
        model.parameters_raw = 10_000_000_000;
        model.active_parameters = Some(2_000_000_000);
        // Only ~20% of the weights are read per token, so the effective size is
        // a fifth of the file size.
        let full = model.file_size_mb as f64 / 1024.0;
        assert!((model_weight_gb(&model) - full * 0.2).abs() < 1e-9);
    }
}

