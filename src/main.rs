#![cfg_attr(windows, windows_subsystem = "windows")]

mod app;
mod document;
mod job;
mod media;
mod pipeline;
mod reading;
mod scenes;
mod settings;
#[cfg(test)]
#[path = "../tests/app/mod.rs"]
mod tests;
mod ui;

use app::{Screen, State};
use document::{Document, Paragraph};
use rune::AppSpec;
use settings::Settings;

fn main() {
    #[cfg(windows)]
    rune::set_app_icon_resource(1);
    #[cfg(target_os = "macos")]
    let _ = rune::set_app_icon(include_bytes!("../assets/app-icon.png").to_vec());
    rune::set_app_id("app.vellum.reader");
    rune::App::new(AppSpec {
        scenes: scenes::ALL,
        fonts: rune_theme_opencode::FONTS,
        families: rune_theme_opencode::FAMILIES,
        min_size: Some((560.0, 600.0)),
        setup: Some(|s: &mut State, wake| {
            s.wake = Some(wake);
            if matches!(s.screen, Screen::Models) {
                s.check_cached_models();
            }
        }),
        poll: Some(app::poll),
        on_key: Some(ui::on_key),
        ..AppSpec::new("Vellum", (980.0, 780.0), app::init, ui::view)
    })
    .compositor(ui::fx::Gfx::default())
    .run();
}
