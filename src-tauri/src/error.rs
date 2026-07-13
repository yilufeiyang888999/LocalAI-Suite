use serde::Serialize;
use thiserror::Error;

pub type AppResult<T> = Result<T, AppError>;

#[derive(Debug, Error)]
pub enum AppError {
    #[error("{0}")]
    Message(String),
    #[error("设置暂时没能保存，请稍后再试")]
    Io(#[from] std::io::Error),
    #[error("数据暂时没能解析，请稍后再试")]
    Json(#[from] serde_json::Error),
    #[error("本地数据暂时不可用，请稍后再试")]
    Sqlite(#[from] rusqlite::Error),
}

impl Serialize for AppError {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(&self.to_string())
    }
}
