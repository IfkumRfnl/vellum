//! Vellum's look: black, white and greys. The progress dots add orange for work in progress
//! (`fx/dots.wgsl`), and green marks what's done.

use rune::{Color, TextStyle};

pub const PAPER: &str = "#090909";
pub const PANEL: &str = "#171717";
pub const EDGE: &str = "#353535";
pub const HOVER: &str = "#272727";
pub const INK: &str = "#eeeeee";
pub const MUTED: &str = "#a8a8a8";
pub const FAINT: &str = "#6b6b6b";
pub const ACCENT: &str = "#ffffff";
pub const SELECTION: &str = "#555555";
/// Finished.
pub const DONE: &str = "#30d158";

pub fn c(hex: &str) -> Color {
    rune::color::hex(hex).unwrap()
}

pub fn kit() -> &'static rune_kit::Kit {
    rune_theme_opencode::kit(true)
}

/// Numbers that tick: tabular figures so digits don't jostle.
pub fn figures(size: f32, weight: f32) -> TextStyle {
    TextStyle::new(size, weight)
        .features("'tnum' 1")
        .tracking(-size * 0.02)
}
