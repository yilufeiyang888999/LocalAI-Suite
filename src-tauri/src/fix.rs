use serde::{Deserialize, Serialize};
use tauri::AppHandle;

use crate::{
    app_state::AppState,
    error::AppResult,
    models::{is_model_installed, model_install_path},
    service::{start_service_core, ServiceInfo},
    settings::save_settings,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LowVramFixResult {
    /// Human-readable, non-technical outcome message.
    pub message: String,
    /// True when the running model was actually switched to a lighter one.
    pub switched: bool,
    /// A lighter model the user should install when none is installed yet.
    pub suggested_model_id: Option<String>,
    pub service: Option<ServiceInfo>,
}

/// §3.7.2 A级: when VRAM is too tight, switch to a lighter already-installed model
/// automatically. If nothing lighter is installed, suggest one to download.
#[tauri::command]
pub async fn apply_low_vram_fix(
    app: AppHandle,
    state: tauri::State<'_, AppState>,
) -> AppResult<LowVramFixResult> {
    let (install_dir, current_path, data_dir) = {
        let inner = state.inner.lock().expect("state lock poisoned");
        (
            inner.settings.install_dir.clone(),
            inner.settings.local_model_path.clone(),
            inner.data_dir.clone(),
        )
    };

    let hardware = tokio::task::spawn_blocking(crate::hardware::detect_hardware)
        .await
        .unwrap_or_else(|_| crate::hardware::fallback_hardware());
    let available_vram = hardware.gpu_vram_gb;

    let catalog = crate::catalog::effective_models(&data_dir);
    let installed_ids: Vec<String> = catalog
        .iter()
        .filter(|model| is_model_installed(&install_dir, model))
        .map(|model| model.id.clone())
        .collect();
    let current_id = current_path.as_ref().and_then(|path| {
        catalog
            .iter()
            .find(|model| model_install_path(&install_dir, model) == *path)
            .map(|model| model.id.clone())
    });

    match choose_smaller_model(&catalog, available_vram, current_id.as_deref(), &installed_ids) {
        SmallerModel::Switch(model_id) => {
            let model = catalog
                .iter()
                .find(|model| model.id == model_id)
                .expect("chosen model exists in catalog");
            let path = model_install_path(&install_dir, model);
            {
                let mut inner = state.inner.lock().expect("state lock poisoned");
                inner.settings.local_model_path = Some(path);
                save_settings(&inner.data_dir, &inner.settings)?;
            }
            let service = start_service_core(&model_id, &app, &state).await.ok();
            tracing::info!(to = %model_id, available_vram, "low-vram fix: switched to lighter model");
            Ok(LowVramFixResult {
                message: format!("已切换到更省显存的「{}」并重新启动。", model.name),
                switched: true,
                suggested_model_id: None,
                service,
            })
        }
        SmallerModel::SuggestInstall(model_id) => {
            let name = catalog
                .iter()
                .find(|model| model.id == model_id)
                .map(|model| model.name.clone())
                .unwrap_or_else(|| model_id.clone());
            tracing::info!(suggested = %model_id, "low-vram fix: suggesting lighter model install");
            Ok(LowVramFixResult {
                message: format!("显存偏紧，建议安装更省显存的「{}」。", name),
                switched: false,
                suggested_model_id: Some(model_id),
                service: None,
            })
        }
        SmallerModel::None => {
            tracing::info!("low-vram fix: no lighter model available to switch");
            Ok(LowVramFixResult {
                message: "暂时没有更省显存的模型可切换，可改用 CPU 模式运行。".to_string(),
                switched: false,
                suggested_model_id: None,
                service: None,
            })
        }
    }
}
enum SmallerModel {
    /// A lighter model is installed — switch to it.
    Switch(String),
    /// The lightest catalog model isn't installed — suggest downloading it.
    SuggestInstall(String),
    None,
}

/// Pick a lighter model than the current one. Prefers the heaviest *installed*
/// model that still fits within `available_vram` (best quality that runs); if
/// none is installed, suggests the lightest catalog model instead.
fn choose_smaller_model(
    catalog: &[crate::models::ModelMetadata],
    available_vram: f64,
    current_id: Option<&str>,
    installed_ids: &[String],
) -> SmallerModel {
    let current_need = current_id
        .and_then(|id| catalog.iter().find(|model| model.id == id))
        .map(|model| model.vram_required_gb)
        .unwrap_or(f64::MAX);

    // Installed models lighter than the current one (and lighter is the point).
    let mut lighter_installed: Vec<_> = catalog
        .iter()
        .filter(|model| {
            installed_ids.iter().any(|id| id == &model.id)
                && Some(model.id.as_str()) != current_id
                && model.vram_required_gb < current_need
        })
        .collect();
    // Prefer ones that fit available VRAM; among those, the heaviest (best).
    lighter_installed.sort_by(|a, b| {
        b.vram_required_gb
            .partial_cmp(&a.vram_required_gb)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    if let Some(fitting) = lighter_installed
        .iter()
        .find(|model| available_vram <= 0.0 || model.vram_required_gb <= available_vram)
    {
        return SmallerModel::Switch(fitting.id.clone());
    }
    // Nothing fits but something lighter is installed — still the best option.
    if let Some(lightest) = lighter_installed.last() {
        return SmallerModel::Switch(lightest.id.clone());
    }

    // Nothing lighter is installed: suggest the lightest catalog model.
    let lightest = catalog
        .iter()
        .filter(|model| Some(model.id.as_str()) != current_id)
        .min_by(|a, b| {
            a.vram_required_gb
                .partial_cmp(&b.vram_required_gb)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    match lightest {
        Some(model) => SmallerModel::SuggestInstall(model.id.clone()),
        None => SmallerModel::None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::builtin_models;

    #[test]
    fn switches_to_heaviest_installed_that_fits() {
        let catalog = builtin_models();
        // Sort ids by vram to reason about the fixture.
        let mut by_vram: Vec<_> = catalog.iter().collect();
        by_vram.sort_by(|a, b| a.vram_required_gb.partial_cmp(&b.vram_required_gb).unwrap());
        let lightest = by_vram[0].id.clone();
        let middle = by_vram[1].id.clone();
        let heaviest = by_vram.last().unwrap().id.clone();

        // Current is the heaviest; lightest + middle are installed; tiny VRAM.
        let installed = vec![lightest.clone(), middle.clone()];
        let choice = choose_smaller_model(&catalog, 2.0, Some(&heaviest), &installed);
        match choice {
            // With 2GB VRAM only the lightest (1.8GB) fits.
            SmallerModel::Switch(id) => assert_eq!(id, lightest),
            _ => panic!("expected a switch to the lightest installed model"),
        }
    }

    #[test]
    fn suggests_install_when_nothing_lighter_installed() {
        let catalog = builtin_models();
        let mut by_vram: Vec<_> = catalog.iter().collect();
        by_vram.sort_by(|a, b| a.vram_required_gb.partial_cmp(&b.vram_required_gb).unwrap());
        let heaviest = by_vram.last().unwrap().id.clone();

        // Current is heaviest, nothing installed → suggest the lightest catalog model.
        let choice = choose_smaller_model(&catalog, 2.0, Some(&heaviest), &[]);
        match choice {
            SmallerModel::SuggestInstall(id) => assert_eq!(id, by_vram[0].id),
            _ => panic!("expected an install suggestion"),
        }
    }
}
