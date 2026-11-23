//! The link field.

use super::{
    theme::{EDGE, INK, MUTED, PANEL, SELECTION, c, kit},
    widgets::{Tone, button, heading, label, toggle},
};
use crate::{Screen, State};
use rune::{El, col, row, text_input};

pub fn view(s: &State) -> El {
    let ty = &kit().ty;
    let field = text_input(&s.input, ty.body)
        .key("url")
        .label("YouTube link")
        .placeholder("Paste a YouTube link", c(MUTED))
        .caret(c(INK), c(SELECTION))
        .color(c(INK))
        .single_line()
        .grow(1.0)
        .min_w(0.0)
        .on_edit(|s: &mut State| &mut s.input)
        .on_submit(|s: &mut State, _| s.start());
    let mut content = col().gap(18.0).child(heading("Read a video")).child(
        row()
            .h(61.0)
            .w_full()
            .px(12.0)
            .gap(10.0)
            .center_y()
            .radius(12.0)
            .fill(c(PANEL))
            .border(1.0, c(EDGE))
            .focus_ring_for("url", 1.0, 1.0, c(MUTED))
            .child(field)
            .child(button("Transcribe", Tone::Primary, |s, _| s.start())),
    );
    content = content.child(
        toggle("Label speakers", s.settings.speakers, |s| {
            s.settings.speakers = !s.settings.speakers;
            s.save();
        })
        .help("Detect voices in the audio and label their turns in the transcript."),
    );
    if !s.message.is_empty() {
        content = content.child(
            label(&s.message, ty.ui, INK)
                .wrap()
                .selectable(true)
                .selection_color(c(SELECTION)),
        );
    } else if let Some(model) = &s.model_job {
        let status = match model.fraction {
            Some(f) => format!("Downloading speech model… {:.0}%", f * 100.0),
            None => "Preparing speech model…".into(),
        };
        content = content.child(label(status, ty.ui, MUTED).role("status"));
    }
    if let Some(d) = &s.doc {
        content = content.child(
            row()
                .gap(8.0)
                .center_y()
                .child(button("Resume", Tone::Quiet, |s, _| {
                    s.screen = Screen::Reader
                }))
                .child(label(&d.title, ty.ui, MUTED).shrink()),
        );
    }
    content
}
