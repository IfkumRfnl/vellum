//! Choosing what turns speech into text: a speech model, or the video's own captions.

use super::{
    theme::{ACCENT, EDGE, HOVER, INK, MUTED, PANEL, c, kit},
    widgets::{Tone, button, heading, label, line, toggle},
};
use crate::{State, media::models::SpeechModel};
use rune::{El, col, el, row};

fn model_status(s: &State, model: &SpeechModel) -> (El, String) {
    let kit = kit();
    let (indicator, description) = if s.ready_models.iter().any(|id| id == model.id) {
        (
            el().size(7.0, 7.0).radius(3.5).fill(c("#62be84")),
            "Downloaded".to_owned(),
        )
    } else if let Some(job) = s.model_job.as_ref().filter(|m| m.id == model.id) {
        let indicator = match job.fraction {
            Some(f) => label(
                format!("{:.0}%", f * 100.0),
                super::theme::figures(12.0, 500.0),
                MUTED,
            ),
            None => rune_kit::Loader::new().size(13.0).build(kit),
        };
        (indicator, "Downloading".to_owned())
    } else {
        let size = model.artifact.megabytes();
        (
            label(format!("↓ {size} MB"), kit.ty.text_xsmall, MUTED),
            format!("{size} MB download"),
        )
    };
    (indicator, format!("{} · {description}", model.language))
}

pub(super) fn view(s: &State) -> El {
    let ty = &kit().ty;
    let mut menu = col()
        .role("radiogroup")
        .label("Speech model")
        .w_full()
        .radius(12.0)
        .clip_rounded(12.0)
        .padding(1.0)
        .border(1.0, c(EDGE))
        .fill(c(PANEL));
    for (index, model) in crate::media::models::SPEECH.iter().enumerate() {
        let (id, name, language) = (model.id, model.name, model.language);
        let chosen = s.settings.model == id;
        let (status, help) = model_status(s, model);
        let mut dot = el()
            .size(16.0, 16.0)
            .radius(8.0)
            .center()
            .border(1.0, c(if chosen { ACCENT } else { MUTED }));
        if chosen {
            dot = dot.child(el().size(8.0, 8.0).radius(4.0).fill(c(ACCENT)));
        }
        if index > 0 {
            menu = menu.child(line());
        }
        menu = menu.child(
            row()
                .key(id)
                .corners(match index {
                    0 => [11.0, 11.0, 0.0, 0.0],
                    2 => [0.0, 0.0, 11.0, 11.0],
                    _ => [0.0; 4],
                })
                .role("radio")
                .label(name)
                .help(help)
                .checked(chosen)
                .focusable(true)
                .focus_ring(2.0, 0.0, c(ACCENT))
                .w_full()
                .h(52.0)
                .px(16.0)
                .gap(12.0)
                .center_y()
                .hover(c(HOVER))
                .hover_motion(rune::Anim::INSTANT, rune::Anim::INSTANT)
                .child(dot)
                .child(label(name, ty.ui, INK))
                .child(el().grow(1.0))
                .child(label(language, ty.text_xsmall, MUTED))
                .child(status)
                .on_press_tx(move |s: &mut State, tx| {
                    tx.instant();
                    s.select_model(id);
                }),
        );
    }
    let mut form = col()
        .key("model-picker")
        .gap(22.0)
        .child(
            row()
                .justify_center()
                .child(heading("Choose a speech model")),
        )
        .child(menu);
    if !s.message.is_empty() {
        form = form.child(label(&s.message, ty.ui, MUTED).role("status").wrap());
    }
    form.child(row().justify_center().child(button(
        "Continue",
        Tone::Primary,
        State::finish_models,
    )))
}

pub(super) fn source_picker(s: &State, layer: &mut Vec<El>) -> El {
    let kit = kit();
    let active =
        crate::media::models::speech(&s.settings.model).map_or("YouTube captions", |m| m.name);
    let trigger = button(&format!("{active}  ▾"), Tone::Quiet, |s, tx| {
        tx.instant();
        s.sources_open = !s.sources_open;
        if s.sources_open {
            s.check_cached_models();
        }
    })
    .key("Transcription")
    .label(format!("Transcription source: {active}"));
    let mut choices = col()
        .w(280.0)
        .role("radiogroup")
        .label("Transcription source");
    let caption = std::iter::once((
        "captions",
        "YouTube captions",
        None,
        "Use the video's caption track".to_owned(),
    ));
    let models = crate::media::models::SPEECH.iter().map(|m| {
        let (status, help) = model_status(s, m);
        (m.id, m.name, Some(status), help)
    });
    for (id, name, status, help) in caption.chain(models) {
        let selected = s.settings.model == id;
        choices = choices.child(
            row()
                .key(id)
                .role("radio")
                .label(name)
                .help(help)
                .checked(selected)
                .focusable(true)
                .focus_ring(1.0, 1.0, c(ACCENT))
                .h(36.0)
                .px(8.0)
                .gap(8.0)
                .center_y()
                .radius(6.0)
                .hover(c(HOVER))
                .child(label(if selected { "✓" } else { "" }, kit.ty.ui, INK).w(15.0))
                .child(label(name, kit.ty.ui, INK))
                .child(el().grow(1.0))
                .children(status)
                .on_press(move |s: &mut State| s.select_model(id)),
        );
    }
    let mut panel = col().gap(7.0).child(choices);
    if s.settings.model != "captions" {
        panel = panel.child(line()).child(
            toggle(
                "Use YouTube captions first",
                s.settings.captions_first,
                |s| {
                    s.settings.captions_first = !s.settings.captions_first;
                    s.save();
                },
            )
            .px(8.0),
        );
    }
    rune_kit::Popover::new("sources")
        .label("Transcription")
        .trigger(trigger)
        .placement(rune_kit::Placement::BottomEnd)
        .padding(8.0)
        .child(
            panel
                .modal(true, |s: &mut State| s.sources_open = false)
                .focus_return("Transcription"),
        )
        .open(s.sources_open)
        .on_close(|s: &mut State| s.sources_open = false)
        .build(kit, layer)
}
