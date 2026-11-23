//! What the window holds and how it changes: screens, actions and the per-frame poll.

use crate::{
    Document,
    job::{Event, Job},
    media,
    pipeline::{self, Tracker},
    reading::Reading,
    settings::{Settings, Store},
};
use rune::{TextEdit, Tx, Waker};
use std::{cell::RefCell, path::PathBuf, rc::Rc, time::Duration};

/// How long the finished meter celebrates before the transcript opens.
pub const CELEBRATION: f64 = 1.35;
/// How long **Copy transcript** says it copied.
pub const COPIED: f64 = 1.6;

pub enum Screen {
    Models,
    Home,
    Working(Box<Working>),
    Reader,
}

/// A transcript in the making.
pub struct Working {
    pub job: Job,
    pub tracker: Tracker,
    /// The finished transcript, held while the meter celebrates.
    pub result: Option<Document>,
}

/// A speech model downloading in the background.
pub struct ModelJob {
    pub id: String,
    pub job: Job,
    pub fraction: Option<f32>,
}

/// The reader's own state, reset for each transcript.
#[derive(Default)]
pub struct ReaderState {
    pub appearance: bool,
    pub edits: Option<Vec<TextEdit>>,
    pub wheel_remainder: f32,
    /// Shared with the scroll handler, which updates it on every frame of a scroll.
    pub progress: Rc<RefCell<Option<Reading>>>,
    pub hud: crate::ui::reader::Hud,
    pub finished_at: Option<f64>,
    /// When the transcript was last copied, for the button's confirmation.
    pub copied_at: Option<f64>,
}

pub struct State {
    pub screen: Screen,
    pub settings: Settings,
    store: Store,
    pub input: TextEdit,
    pub doc: Option<Document>,
    pub reader: ReaderState,
    pub sources_open: bool,
    pub message: String,
    pub model_job: Option<ModelJob>,
    pub ready_models: Vec<String>,
    pub wake: Option<Waker>,
    /// The clock at the latest poll.
    pub clock: f64,
}

pub fn init() -> State {
    load_state(crate::settings::settings_path())
}

pub fn load_state(path: Option<PathBuf>) -> State {
    let settings = Settings::load(path.as_ref());
    State {
        screen: if settings.configured {
            Screen::Home
        } else {
            Screen::Models
        },
        input: TextEdit::new(""),
        doc: None,
        reader: ReaderState::default(),
        sources_open: false,
        message: String::new(),
        model_job: None,
        ready_models: Vec::new(),
        settings,
        store: Store::new(path),
        wake: None,
        clock: 0.0,
    }
}

impl State {
    pub fn save(&self) {
        self.store.save(&self.settings);
    }

    pub fn select_model(&mut self, id: &str) {
        self.settings.model = id.into();
        self.message.clear();
        self.prepare_model();
        if self.settings.configured {
            self.sources_open = false;
            self.save();
        }
    }

    pub fn check_cached_models(&mut self) {
        if let Ok(root) = media::models::cache_root() {
            self.ready_models = media::models::installed(&root);
        }
    }

    pub fn prepare_model(&mut self) {
        let model = &self.settings.model;
        if matches!(&self.model_job, Some(m) if &m.id == model) {
            return;
        }
        self.model_job = None;
        if model != "captions"
            && !self.ready_models.contains(model)
            && let Some(wake) = self.wake.clone()
        {
            let id = model.clone();
            self.model_job = Some(ModelJob {
                id: model.clone(),
                job: Job::start(wake, move |progress| {
                    let root = media::models::cache_root()?;
                    media::models::speech(&id)?.artifact.get(&root, progress)?;
                    Ok(Event::Cached(media::models::installed(&root)))
                }),
                fraction: None,
            });
        }
    }

    pub fn finish_models(&mut self, tx: &mut Tx) {
        tx.instant();
        self.prepare_model();
        self.settings.configured = true;
        self.save();
        self.screen = Screen::Home;
        self.message.clear();
        tx.focus("url", false);
    }

    pub fn start(&mut self) {
        let url = self.input.text.trim();
        if url.is_empty() {
            self.message = "Paste a YouTube link to begin.".into();
            return;
        }
        if let Err(e) = media::video_url(url) {
            self.message = e.to_string();
            return;
        }
        if let Some(wake) = self.wake.clone() {
            self.message.clear();
            let url = url.to_owned();
            let settings = self.settings.clone();
            let tracker = Tracker::new(pipeline::plan(&settings));
            self.screen = Screen::Working(Box::new(Working {
                job: Job::start(wake, move |progress| {
                    media::read_video(&url, &settings, progress).map(Event::Result)
                }),
                tracker,
                result: None,
            }));
        }
    }

    pub fn cancel(&mut self) {
        self.screen = Screen::Home;
        self.message.clear();
    }

    pub fn open_reader(&mut self, doc: Document) {
        self.keep_reading_speed();
        let words = doc.words();
        self.doc = Some(doc);
        self.reader = ReaderState::default();
        *self.reader.progress.borrow_mut() = Some(Reading::new(words, self.settings.reading_wpm));
        self.message.clear();
        self.screen = Screen::Reader;
    }

    pub fn new_video(&mut self, tx: &mut Tx) {
        self.finish_editing();
        self.keep_reading_speed();
        self.input = TextEdit::new("");
        self.message.clear();
        self.screen = Screen::Home;
        tx.focus("url", false);
    }

    /// Remembers the measured reading speed for the next transcript, once it's trustworthy.
    pub fn keep_reading_speed(&mut self) {
        let measured = self
            .reader
            .progress
            .borrow()
            .as_ref()
            .filter(|r| r.confident())
            .map(|r| r.wpm() as f32);
        if let Some(wpm) = measured
            && (wpm - self.settings.reading_wpm).abs() >= 2.0
        {
            self.settings.reading_wpm = wpm;
            self.save();
        }
    }

    /// How far through the transcript the reader is, 0 to 1.
    #[cfg(test)]
    pub fn reading_fraction(&self) -> f32 {
        self.reader
            .progress
            .borrow()
            .as_ref()
            .map_or(0.0, |r| r.fraction)
    }

    pub fn toggle_editing(&mut self) {
        if self.reader.edits.is_some() {
            self.finish_editing();
        } else if let Some(doc) = &self.doc {
            self.reader.edits = Some(
                doc.paragraphs
                    .iter()
                    .map(|p| TextEdit::new(p.text.clone()))
                    .collect(),
            );
        }
    }

    pub fn finish_editing(&mut self) {
        if let (Some(edits), Some(doc)) = (self.reader.edits.take(), self.doc.as_mut()) {
            for (paragraph, mut edit) in doc.paragraphs.iter_mut().zip(edits) {
                paragraph.text = std::mem::take(&mut edit.text);
            }
        }
    }

    pub fn adjust_font(&mut self, steps: i32) {
        let size = (i32::from(self.settings.size).saturating_add(steps)).clamp(16, 32) as u8;
        if size != self.settings.size {
            self.settings.size = size;
            self.save();
        }
    }

    /// The transcript as plain text, with any corrections in progress.
    pub fn transcript_text(&self) -> Option<String> {
        let doc = self.doc.as_ref()?;
        Some(doc.plain_text(|i| {
            self.reader
                .edits
                .as_ref()
                .map_or(doc.paragraphs[i].text.as_str(), |e| e[i].text.as_str())
        }))
    }
}

pub fn poll(s: &mut State, clock: f64, tx: &mut Tx) -> bool {
    let dt = (clock - s.clock).clamp(0.0, 0.1);
    s.clock = clock;
    let mut changed = false;
    if let Ok(error) = s.store.errors.try_recv() {
        s.message = error;
        changed = true;
    }
    changed |= poll_model(s);
    changed |= match &s.screen {
        Screen::Working(_) => poll_working(s, clock, dt, tx),
        Screen::Reader => poll_reader(s, clock, tx),
        _ => false,
    };
    changed
}

fn poll_model(s: &mut State) -> bool {
    let Some(model) = &mut s.model_job else {
        return false;
    };
    let mut changed = false;
    let mut finished = None;
    for event in model.job.events.try_iter() {
        match event {
            Event::Update(pipeline::Update::Fraction(f)) => {
                model.fraction = Some(f);
                changed = true;
            }
            Event::Update(_) => {}
            Event::Cached(models) => finished = Some(Ok(models)),
            Event::Error(error) => finished = Some(Err(error)),
            Event::Result(_) => unreachable!("model preparation returns no transcript"),
        }
    }
    match finished {
        Some(Ok(models)) => s.ready_models = models,
        Some(Err(error)) if !matches!(s.screen, Screen::Working(_)) => {
            s.message = format!("Could not prepare the model. {error}");
        }
        _ => return changed,
    }
    s.model_job = None;
    true
}

fn poll_working(s: &mut State, clock: f64, dt: f64, tx: &mut Tx) -> bool {
    let Screen::Working(working) = &mut s.screen else {
        return false;
    };
    let mut changed = false;
    let mut failure = None;
    for event in working.job.events.try_iter() {
        changed = true;
        match event {
            Event::Update(update) => working.tracker.apply(update, clock),
            Event::Result(doc) => {
                working.tracker.finish(clock);
                working.result = Some(doc);
            }
            Event::Error(error) => failure = Some(error),
            Event::Cached(_) => unreachable!("transcription returns no model inventory"),
        }
    }
    if let Some(error) = failure {
        s.message = error;
        s.screen = Screen::Home;
        return true;
    }
    changed |= working.tracker.tick(clock, dt);
    let celebrated = working
        .tracker
        .finished
        .is_some_and(|at| clock - at >= CELEBRATION);
    if celebrated && let Some(doc) = working.result.take() {
        s.open_reader(doc);
        return true;
    }
    // The meter moves every frame while the job runs.
    tx.wake_after(Duration::from_millis(16));
    changed
}

fn poll_reader(s: &mut State, clock: f64, tx: &mut Tx) -> bool {
    if let Some(at) = s.reader.copied_at {
        if clock - at >= COPIED {
            s.reader.copied_at = None;
            return true;
        }
        tx.wake_after(Duration::from_secs_f64(COPIED - (clock - at)));
    }
    let finished = s
        .reader
        .progress
        .borrow()
        .as_ref()
        .is_some_and(|r| r.finished);
    if finished && s.reader.finished_at.is_none() {
        s.reader.finished_at = Some(clock);
        s.keep_reading_speed();
        return true;
    }
    // The time-left estimate refines while you read; the handler refreshes it on scroll, and
    // this checks for the end of the transcript between scrolls.
    if s.reader.finished_at.is_none() {
        tx.wake_after(Duration::from_millis(400));
    }
    false
}
