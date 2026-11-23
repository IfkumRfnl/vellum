//! The window: a toolbar, then whichever screen is up.

pub mod fx;
mod home;
mod model_picker;
pub mod reader;
pub mod theme;
mod widgets;
mod working;

use crate::{Screen, State};
use rune::{Cx, El, Key, KeyEvent, Transition, Tx, col, row};
use theme::{PAPER, c};
use widgets::{Tone, button, centered};

pub fn view(s: &State, _: &mut Cx<State>) -> El {
    let mut layer = Vec::new();
    let tools = match &s.screen {
        Screen::Models => Some(
            button("Skip", Tone::Quiet, |s, tx| {
                s.settings.model = "captions".into();
                s.finish_models(tx);
            })
            .opacity(0.75)
            .help("Read YouTube captions without a speech model"),
        ),
        Screen::Home => Some(model_picker::source_picker(s, &mut layer)),
        Screen::Reader => Some(reader::toolbar(s)),
        Screen::Working(_) => None,
    };
    let body = match &s.screen {
        Screen::Models => centered("models", model_picker::view(s), 460.0),
        Screen::Home => centered("home", home::view(s), 600.0),
        Screen::Working(working) => working::view(s, working),
        Screen::Reader => reader::view(s, s.doc.as_ref().expect("reader has a transcript")),
    };
    let body = body.grow(1.0).min_h(0.0).transition(Transition::opacity());
    let mut window = col().w_full().h_full().fill(c(PAPER));
    if let Some(tools) = tools {
        window = window.child(
            row()
                .key("toolbar")
                .w_full()
                .px(20.0)
                .py(12.0)
                .justify_end()
                .child(tools),
        );
    }
    if matches!(s.screen, Screen::Reader) {
        window = window.on_wheel(reader::font_wheel);
    }
    rune_kit::root(window.child(body), layer)
}

/// Keys that work wherever focus is.
pub fn on_key(s: &mut State, e: &KeyEvent, tx: &mut Tx) -> bool {
    match (&s.screen, &e.key) {
        (Screen::Working(_), Key::Escape) => {
            s.cancel();
            true
        }
        (Screen::Reader, Key::Escape) if s.reader.appearance => {
            s.reader.appearance = false;
            true
        }
        (Screen::Reader, _) => reader::on_key(s, e, tx),
        _ => false,
    }
}
