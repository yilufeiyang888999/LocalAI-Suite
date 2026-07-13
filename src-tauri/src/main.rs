// Hide the extra console window on Windows release builds (debug keeps it for logs).
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    local_ai_installer_lib::run();
}
