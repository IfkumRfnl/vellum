mod live;
mod onboarding;
mod progress;
mod reader;

use crate::{
    Document, Paragraph, Screen, Settings, State,
    app::{load_state, poll},
    job::{Event, Job},
    media,
    settings::ReadingWidth,
    ui,
};
use rune::{Env, Runtime};
use std::{
    sync::Arc,
    thread,
    time::{Duration, Instant},
};

fn tap(rt: &mut Runtime<State>, key: &str) {
    let (x, y) = rt.find(key).unwrap_or_else(|| panic!("missing {key}"));
    rt.pointer_down(x, y);
    rt.pointer_up(x, y);
    rt.frame(0.02);
}
