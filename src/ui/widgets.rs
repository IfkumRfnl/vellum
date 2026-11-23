//! The controls every screen shares.

use super::theme::{ACCENT, EDGE, HOVER, INK, MUTED, PANEL, PAPER, c, kit};
use crate::State;
use rune::{El, ScrollSpec, TextStyle, Tx, col, el, row, text};

pub fn label(value: impl Into<String>, style: TextStyle, color: &str) -> El {
    text(value).font(style).color(c(color))
}

pub fn heading(value: &str) -> El {
    label(
        value,
        TextStyle::new(26.0, 550.0).lh(32.0).tracking(-0.35),
        INK,
    )
    .wrap()
}

#[derive(Clone, Copy, PartialEq)]
pub enum Tone {
    /// White, for the one action a screen is about.
    Primary,
    Quiet,
    /// No fill until hovered.
    Ghost,
}

pub fn button(name: &str, tone: Tone, action: impl Fn(&mut State, &mut Tx) + 'static) -> El {
    let (bg, fg, over) = match tone {
        Tone::Primary => (ACCENT, PAPER, "#dedede"),
        Tone::Quiet => (PANEL, INK, HOVER),
        Tone::Ghost => (PAPER, MUTED, HOVER),
    };
    row()
        .key(name)
        .role("button")
        .label(name)
        .focusable(true)
        .focus_ring(2.0, 2.0, c(ACCENT))
        .center()
        .px(14.0)
        .h(36.0)
        .radius(8.0)
        .fill(c(bg))
        .hover(c(over))
        .child(label(name, kit().ty.ui_medium, fg))
        .on_press_tx(action)
}

/// A button that reads as chosen while `on`.
pub fn choice(name: &str, on: bool, action: impl Fn(&mut State, &mut Tx) + 'static) -> El {
    button(name, if on { Tone::Primary } else { Tone::Quiet }, action)
}

pub fn line() -> El {
    el().w_full().h(1.0).fill(c(EDGE))
}

pub fn toggle(name: &str, on: bool, action: impl Fn(&mut State) + 'static) -> El {
    let track = row()
        .w(30.0)
        .h(17.0)
        .px(3.0)
        .center_y()
        .radius(9.0)
        .fill(c(if on { ACCENT } else { EDGE }));
    let track = if on { track.justify_end() } else { track };
    row()
        .key(name)
        .w_full()
        .h(29.0)
        .justify_between()
        .center_y()
        .role("switch")
        .label(name)
        .checked(on)
        .focusable(true)
        .focus_ring(1.0, 2.0, c(ACCENT))
        .on_press(action)
        .child(label(name, kit().ty.ui, INK))
        .child(
            track.child(
                el().size(11.0, 11.0)
                    .radius(6.0)
                    .fill(c(if on { PAPER } else { MUTED })),
            ),
        )
}

/// A screen's content, centred and scrollable when the window is short.
pub fn centered(key: &str, content: El, width: f32) -> El {
    col()
        .key(key)
        .grow(1.0)
        .min_h(0.0)
        .scroll(ScrollSpec {
            stick_bottom: false,
        })
        .child(
            col()
                .w_full()
                .grow(1.0)
                .center()
                .padding(24.0)
                .child(content.w_full().max_w(width)),
        )
}
