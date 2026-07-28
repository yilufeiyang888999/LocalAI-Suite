use chrono::Utc;
use rusqlite::{Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::{
    app_state::AppState,
    error::{AppError, AppResult},
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatSession {
    pub id: String,
    pub name: String,
    pub model_id: String,
    pub role_preset: RolePreset,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub id: String,
    pub session_id: String,
    pub role: MessageRole,
    pub content: String,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RolePreset {
    DailyAssistant,
    Writing,
    Coding,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MessageRole {
    User,
    Assistant,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SaveChatMessageRequest {
    pub session_id: String,
    pub role: MessageRole,
    pub content: String,
}

#[tauri::command]
pub fn create_chat_session(
    model_id: String,
    role_preset: RolePreset,
    state: tauri::State<'_, AppState>,
) -> AppResult<ChatSession> {
    let trimmed_model = model_id.trim();
    if trimmed_model.is_empty() {
        return Err(AppError::Message("请先启动一个可用的模型。".to_string()));
    }

    let now = Utc::now().to_rfc3339();
    let session = ChatSession {
        id: Uuid::new_v4().to_string(),
        name: "新的对话".to_string(),
        model_id: trimmed_model.to_string(),
        role_preset,
        created_at: now.clone(),
        updated_at: now,
    };
    let db_path = {
        let inner = state.inner.lock().expect("state lock poisoned");
        inner.db_path.clone()
    };
    let connection = Connection::open(db_path)?;
    insert_session(&connection, &session)?;
    tracing::info!(session_id = %session.id, model_id = %session.model_id, "chat session created");
    Ok(session)
}

fn insert_session(connection: &Connection, session: &ChatSession) -> AppResult<()> {
    connection.execute(
        "INSERT INTO chat_sessions (id, name, model_id, role_preset, created_at, updated_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
        rusqlite::params![
            session.id,
            session.name,
            session.model_id,
            role_preset_to_db(&session.role_preset),
            session.created_at,
            session.updated_at,
        ],
    )?;
    Ok(())
}

#[tauri::command]
pub fn get_chat_sessions(state: tauri::State<'_, AppState>) -> AppResult<Vec<ChatSession>> {
    let db_path = {
        let inner = state.inner.lock().expect("state lock poisoned");
        inner.db_path.clone()
    };
    let connection = Connection::open(db_path)?;
    query_sessions(&connection)
}

fn query_sessions(connection: &Connection) -> AppResult<Vec<ChatSession>> {
    let mut statement = connection.prepare(
        "SELECT id, name, model_id, role_preset, created_at, updated_at
         FROM chat_sessions
         ORDER BY updated_at DESC",
    )?;
    let rows = statement.query_map([], |row| {
        let role_preset: String = row.get(3)?;
        Ok(ChatSession {
            id: row.get(0)?,
            name: row.get(1)?,
            model_id: row.get(2)?,
            role_preset: role_preset_from_db(&role_preset),
            created_at: row.get(4)?,
            updated_at: row.get(5)?,
        })
    })?;

    let mut sessions = Vec::new();
    for row in rows {
        sessions.push(row?);
    }
    Ok(sessions)
}

#[tauri::command]
pub fn get_chat_messages(
    session_id: String,
    state: tauri::State<'_, AppState>,
) -> AppResult<Vec<ChatMessage>> {
    if session_id.trim().is_empty() {
        return Err(AppError::Message("暂时没找到这个对话。".to_string()));
    }
    let db_path = {
        let inner = state.inner.lock().expect("state lock poisoned");
        inner.db_path.clone()
    };
    let connection = Connection::open(db_path)?;
    ensure_session_exists(&connection, &session_id)?;
    query_messages(&connection, &session_id)
}

fn query_messages(connection: &Connection, session_id: &str) -> AppResult<Vec<ChatMessage>> {
    let mut statement = connection.prepare(
        "SELECT id, session_id, role, content, created_at
         FROM chat_messages
         WHERE session_id = ?1
         ORDER BY created_at ASC",
    )?;
    let rows = statement.query_map([session_id], |row| {
        let role: String = row.get(2)?;
        Ok(ChatMessage {
            id: row.get(0)?,
            session_id: row.get(1)?,
            role: message_role_from_db(&role),
            content: row.get(3)?,
            created_at: row.get(4)?,
        })
    })?;

    let mut messages = Vec::new();
    for row in rows {
        messages.push(row?);
    }
    Ok(messages)
}

#[tauri::command]
pub fn save_chat_message(
    request: SaveChatMessageRequest,
    state: tauri::State<'_, AppState>,
) -> AppResult<ChatMessage> {
    if request.session_id.trim().is_empty() {
        return Err(AppError::Message("暂时没找到这个对话。".to_string()));
    }
    if request.content.trim().is_empty() {
        return Err(AppError::Message("消息内容不能为空。".to_string()));
    }

    let db_path = {
        let inner = state.inner.lock().expect("state lock poisoned");
        inner.db_path.clone()
    };
    let connection = Connection::open(db_path)?;
    ensure_session_exists(&connection, &request.session_id)?;
    let now = Utc::now().to_rfc3339();
    let message = ChatMessage {
        id: Uuid::new_v4().to_string(),
        session_id: request.session_id,
        role: request.role,
        content: request.content,
        created_at: now,
    };
    insert_message(&connection, &message)?;
    tracing::debug!(session_id = %message.session_id, role = ?message.role, content_len = message.content.len(), "chat message saved");
    Ok(message)
}

fn insert_message(connection: &Connection, message: &ChatMessage) -> AppResult<()> {
    connection.execute(
        "INSERT INTO chat_messages (id, session_id, role, content, created_at)
         VALUES (?1, ?2, ?3, ?4, ?5)",
        rusqlite::params![
            message.id,
            message.session_id,
            message_role_to_db(&message.role),
            message.content,
            message.created_at,
        ],
    )?;
    connection.execute(
        "UPDATE chat_sessions SET updated_at = ?1 WHERE id = ?2",
        rusqlite::params![message.created_at, message.session_id],
    )?;
    Ok(())
}

#[tauri::command]
pub fn clear_chat_context(session_id: String, state: tauri::State<'_, AppState>) -> AppResult<()> {
    if session_id.trim().is_empty() {
        return Err(AppError::Message("暂时没找到这个对话。".to_string()));
    }
    let db_path = {
        let inner = state.inner.lock().expect("state lock poisoned");
        inner.db_path.clone()
    };
    let connection = Connection::open(db_path)?;
    ensure_session_exists(&connection, &session_id)?;
    delete_messages(&connection, &session_id)?;
    tracing::info!(%session_id, "chat context cleared");
    Ok(())
}

#[tauri::command]
pub fn rename_chat_session(
    session_id: String,
    name: String,
    state: tauri::State<'_, AppState>,
) -> AppResult<()> {
    let trimmed = name.trim();
    if session_id.trim().is_empty() {
        return Err(AppError::Message("暂时没找到这个对话。".to_string()));
    }
    if trimmed.is_empty() {
        return Err(AppError::Message("对话名称不能为空。".to_string()));
    }
    let db_path = {
        let inner = state.inner.lock().expect("state lock poisoned");
        inner.db_path.clone()
    };
    let connection = Connection::open(db_path)?;
    ensure_session_exists(&connection, &session_id)?;
    connection.execute(
        "UPDATE chat_sessions SET name = ?1, updated_at = ?2 WHERE id = ?3",
        rusqlite::params![trimmed, Utc::now().to_rfc3339(), session_id],
    )?;
    tracing::info!(%session_id, "chat session renamed");
    Ok(())
}

#[tauri::command]
pub fn delete_chat_session(
    session_id: String,
    state: tauri::State<'_, AppState>,
) -> AppResult<()> {
    if session_id.trim().is_empty() {
        return Err(AppError::Message("暂时没找到这个对话。".to_string()));
    }
    let db_path = {
        let inner = state.inner.lock().expect("state lock poisoned");
        inner.db_path.clone()
    };
    let connection = Connection::open(db_path)?;
    ensure_session_exists(&connection, &session_id)?;
    delete_messages(&connection, &session_id)?;
    connection.execute("DELETE FROM chat_sessions WHERE id = ?1", [&session_id])?;
    tracing::info!(%session_id, "chat session deleted");
    Ok(())
}

#[tauri::command]
pub fn update_session_role(
    session_id: String,
    role_preset: RolePreset,
    state: tauri::State<'_, AppState>,
) -> AppResult<()> {
    if session_id.trim().is_empty() {
        return Err(AppError::Message("暂时没找到这个对话。".to_string()));
    }
    let db_path = {
        let inner = state.inner.lock().expect("state lock poisoned");
        inner.db_path.clone()
    };
    let connection = Connection::open(db_path)?;
    ensure_session_exists(&connection, &session_id)?;
    connection.execute(
        "UPDATE chat_sessions SET role_preset = ?1, updated_at = ?2 WHERE id = ?3",
        rusqlite::params![role_preset_to_db(&role_preset), Utc::now().to_rfc3339(), session_id],
    )?;
    tracing::info!(%session_id, role = ?role_preset, "chat session role updated");
    Ok(())
}

fn delete_messages(connection: &Connection, session_id: &str) -> AppResult<()> {
    connection.execute("DELETE FROM chat_messages WHERE session_id = ?1", [session_id])?;
    Ok(())
}

/// Delete every chat session bound to a given model, along with all their
/// messages. Used by the uninstall flow when the user opts in to also wiping
/// chat history for the model being removed. Returns the number of sessions
/// that were removed (0 is a normal outcome — no sessions ever existed).
pub fn delete_chat_sessions_for_model(
    db_path: &std::path::Path,
    model_id: &str,
) -> AppResult<u32> {
    let connection = Connection::open(db_path)?;
    // Wipe messages first so we don't leave orphans if the second statement
    // fails partway. Runs as one SQL statement with a subquery, no explicit tx.
    connection.execute(
        "DELETE FROM chat_messages
         WHERE session_id IN (SELECT id FROM chat_sessions WHERE model_id = ?1)",
        [model_id],
    )?;
    let deleted = connection.execute(
        "DELETE FROM chat_sessions WHERE model_id = ?1",
        [model_id],
    )?;
    Ok(deleted as u32)
}

fn ensure_session_exists(connection: &Connection, session_id: &str) -> AppResult<()> {
    let exists = connection
        .query_row(
            "SELECT 1 FROM chat_sessions WHERE id = ?1 LIMIT 1",
            [session_id],
            |_| Ok(()),
        )
        .optional()?
        .is_some();
    if !exists {
        return Err(AppError::Message("暂时没找到这个对话。".to_string()));
    }
    Ok(())
}

fn role_preset_to_db(role: &RolePreset) -> &'static str {
    match role {
        RolePreset::DailyAssistant => "daily_assistant",
        RolePreset::Writing => "writing",
        RolePreset::Coding => "coding",
    }
}

fn role_preset_from_db(role: &str) -> RolePreset {
    match role {
        "writing" => RolePreset::Writing,
        "coding" => RolePreset::Coding,
        _ => RolePreset::DailyAssistant,
    }
}

fn message_role_to_db(role: &MessageRole) -> &'static str {
    match role {
        MessageRole::User => "user",
        MessageRole::Assistant => "assistant",
    }
}

fn message_role_from_db(role: &str) -> MessageRole {
    match role {
        "assistant" => MessageRole::Assistant,
        _ => MessageRole::User,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::database::init_database;

    struct TempDb {
        dir: std::path::PathBuf,
        db: std::path::PathBuf,
    }

    impl TempDb {
        fn new() -> Self {
            let dir = std::env::temp_dir().join(format!("localai-chat-test-{}", Uuid::new_v4()));
            let db = init_database(&dir).expect("init database");
            Self { dir, db }
        }
        fn open(&self) -> Connection {
            Connection::open(&self.db).expect("open db")
        }
    }

    impl Drop for TempDb {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }

    fn sample_session() -> ChatSession {
        let now = Utc::now().to_rfc3339();
        ChatSession {
            id: Uuid::new_v4().to_string(),
            name: "新的对话".to_string(),
            model_id: "qwen2-1_5b".to_string(),
            role_preset: RolePreset::Coding,
            created_at: now.clone(),
            updated_at: now,
        }
    }

    #[test]
    fn chat_records_persist_across_reopen() {
        let temp = TempDb::new();
        let session = sample_session();

        // Write through the same code the Tauri commands use, then drop the connection.
        {
            let conn = temp.open();
            insert_session(&conn, &session).unwrap();
            let turns = [
                (MessageRole::User, "你好"),
                (MessageRole::Assistant, "你好，有什么可以帮忙的？"),
            ];
            for (i, (role, content)) in turns.into_iter().enumerate() {
                let message = ChatMessage {
                    id: Uuid::new_v4().to_string(),
                    session_id: session.id.clone(),
                    role,
                    content: content.to_string(),
                    created_at: format!("2026-07-01T00:00:0{}Z", i),
                };
                insert_message(&conn, &message).unwrap();
            }
        }

        // Reopen the database file to simulate an app restart.
        let conn = temp.open();
        let sessions = query_sessions(&conn).unwrap();
        assert_eq!(sessions.len(), 1);
        assert_eq!(sessions[0].id, session.id);
        assert!(matches!(sessions[0].role_preset, RolePreset::Coding));

        let messages = query_messages(&conn, &session.id).unwrap();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].content, "你好");
        assert!(matches!(messages[0].role, MessageRole::User));
        assert!(matches!(messages[1].role, MessageRole::Assistant));

        // Clearing context wipes messages but keeps the session.
        delete_messages(&conn, &session.id).unwrap();
        assert!(query_messages(&conn, &session.id).unwrap().is_empty());
        assert_eq!(query_sessions(&conn).unwrap().len(), 1);
    }

    #[test]
    fn deleting_a_session_removes_it_and_its_messages() {
        let temp = TempDb::new();
        let session = sample_session();
        let conn = temp.open();
        insert_session(&conn, &session).unwrap();
        insert_message(
            &conn,
            &ChatMessage {
                id: Uuid::new_v4().to_string(),
                session_id: session.id.clone(),
                role: MessageRole::User,
                content: "hi".to_string(),
                created_at: "2026-07-01T00:00:00Z".to_string(),
            },
        )
        .unwrap();

        // Mirror delete_chat_session: wipe messages then the session row.
        delete_messages(&conn, &session.id).unwrap();
        conn.execute("DELETE FROM chat_sessions WHERE id = ?1", [&session.id])
            .unwrap();

        assert!(query_sessions(&conn).unwrap().is_empty());
        assert!(query_messages(&conn, &session.id).unwrap().is_empty());
    }

    #[test]
    fn deleting_all_sessions_for_a_model_leaves_other_models_alone() {
        let temp = TempDb::new();

        // Two sessions for the model being uninstalled, one for an untouched model.
        let mut targeted_a = sample_session();
        targeted_a.model_id = "target-model".to_string();
        let mut targeted_b = sample_session();
        targeted_b.id = Uuid::new_v4().to_string();
        targeted_b.model_id = "target-model".to_string();
        let mut other = sample_session();
        other.id = Uuid::new_v4().to_string();
        other.model_id = "keep-model".to_string();

        {
            let conn = temp.open();
            for session in [&targeted_a, &targeted_b, &other] {
                insert_session(&conn, session).unwrap();
                insert_message(
                    &conn,
                    &ChatMessage {
                        id: Uuid::new_v4().to_string(),
                        session_id: session.id.clone(),
                        role: MessageRole::User,
                        content: "hi".to_string(),
                        created_at: "2026-07-01T00:00:00Z".to_string(),
                    },
                )
                .unwrap();
            }
        }

        let removed = delete_chat_sessions_for_model(&temp.db, "target-model").unwrap();
        assert_eq!(removed, 2);

        let conn = temp.open();
        let remaining = query_sessions(&conn).unwrap();
        assert_eq!(remaining.len(), 1);
        assert_eq!(remaining[0].model_id, "keep-model");
        // Messages for the untouched session must survive.
        assert_eq!(query_messages(&conn, &other.id).unwrap().len(), 1);
        // And messages for the wiped sessions must be gone.
        assert!(query_messages(&conn, &targeted_a.id).unwrap().is_empty());
        assert!(query_messages(&conn, &targeted_b.id).unwrap().is_empty());
    }
}
