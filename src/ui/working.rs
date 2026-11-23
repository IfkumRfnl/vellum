//! The progress screen: the video's title, one plate of dots that fills as the transcript is
//! made, the percentage and the time left, and the words as they're recognized.

use super::{
    fx,
    theme::{DONE, EDGE, FAINT, INK, MUTED, PANEL, c, figures, kit},
    widgets::{Tone, button, label},
};
use crate::{State, app::Working, document::timestamp};
use rune::{El, TextStyle, Transition, col, effect, row, text};
use rune_kit::RollingText;

pub fn view(s: &State, w: &Working) -> El {
    let t = &w.tracker;
    let finished = t.finished;
    let title = label(
        t.title.as_deref().unwrap_or("Opening the video…"),
        TextStyle::new(20.0, 550.0).lh(26.0).tracking(-0.3),
        if t.title.is_some() { INK } else { MUTED },
    )
    .key("working-title")
    .max_lines(2)
    .h(52.0)
    .transition(Transition::opacity());
    let plate = col()
        .key("plate")
        .role("progressbar")
        .label("Transcript progress")
        .w_full()
        .h(64.0)
        .radius(14.0)
        .fill(c(PANEL))
        .border(1.0, c(EDGE))
        .child(
            effect(fx::PLATE)
                .key("plate-dots")
                .effect_data([t.shown, finished.map_or(-1.0, |at| at as f32), 0.0, 0.0])
                .pin(Some(10.0), Some(6.0), Some(10.0), Some(6.0)),
        );
    let percent = (t.shown * 100.0).floor().min(100.0) as i64;
    let number = RollingText::new("percent", format!("{percent}%"))
        .font(figures(40.0, 600.0))
        .color(c(INK))
        .value(percent as f64)
        .build(kit());
    let (status, color) = if finished.is_some() {
        ("Ready".to_owned(), DONE)
    } else if let Some(left) = t.remaining(s.clock) {
        (format!("about {} left", timestamp(left.max(1.0))), MUTED)
    } else if !t.note.is_empty() {
        (t.note.clone(), MUTED)
    } else {
        ("Working…".to_owned(), MUTED)
    };
    let readout = row()
        .w_full()
        .justify_between()
        .items_end()
        .child(number)
        .child(label(status, figures(14.0, 500.0), color).pb(7.0));
    let mut heard = col()
        .key("heard")
        .w_full()
        .h(96.0)
        .gap(6.0)
        .justify_end()
        .clip_fade(0.0, 36.0, 0.0, 0.0);
    for (n, chunk) in &t.heard {
        heard = heard.child(
            text(chunk)
                .key(format!("heard-{n}"))
                .font(kit().ty.body)
                .color(c(MUTED))
                .wrap()
                .reveal(rune::edge::WORD_FADE.word_fade),
        );
    }
    let content = col()
        .w_full()
        .max_w(560.0)
        .gap(18.0)
        .child(title)
        .child(plate)
        .child(readout)
        .child(heard.mt(10.0));
    let mut screen = col()
        .key("working")
        .w_full()
        .center()
        .padding(24.0)
        .child(content);
    if finished.is_none() {
        screen = screen.child(
            button("Cancel", Tone::Ghost, |s, _| s.cancel())
                .help("Stop and go back (Esc)")
                .pin(None, Some(12.0), Some(20.0), None),
        );
    }
    if let Some(device) = &t.device {
        screen = screen.child(
            row()
                .pin(Some(0.0), None, Some(0.0), Some(14.0))
                .justify_center()
                .child(label(device, kit().ty.text_xsmall, FAINT)),
        );
    }
    screen
}
