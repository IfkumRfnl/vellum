//! Background jobs: a worker thread, its cancellation, and the updates it sends the window.

use crate::{
    Document,
    pipeline::{Stage, Update},
};
use anyhow::{Context, Result, bail, ensure};
use rune::Waker;
use std::{
    cell::Cell,
    io::{BufRead, BufReader, Read, Seek, SeekFrom},
    process::{Child, Command, Stdio},
    sync::{
        Arc, Mutex,
        mpsc::{self, Receiver, Sender},
    },
    thread,
    time::{Duration, Instant},
};
use transcribe_cpp::CancelToken;

pub enum Event {
    /// Model preparation finished: the speech models now on disk.
    Cached(Vec<String>),
    Update(Update),
    Result(Document),
    Error(String),
}

#[derive(Default)]
struct Cancellation {
    token: CancelToken,
    child: Mutex<Option<Child>>,
    media_dir: Mutex<Option<tempfile::TempDir>>,
}

pub struct Job {
    pub events: Receiver<Event>,
    cancel: Arc<Cancellation>,
}

/// The worker's side of a job: report progress, run tools, notice cancellation.
pub struct Progress {
    cancel: Arc<Cancellation>,
    send: Sender<Event>,
    wake: Waker,
    /// The last fraction sent and when, so a tight loop doesn't flood the window.
    sent: Cell<(f32, Option<Instant>)>,
}

impl Job {
    pub fn start(
        wake: Waker,
        run: impl FnOnce(&Progress) -> Result<Event> + Send + 'static,
    ) -> Self {
        let (send, events) = mpsc::channel();
        let cancel = Arc::new(Cancellation::default());
        let progress = Progress {
            cancel: cancel.clone(),
            send,
            wake,
            sent: Cell::new((-1.0, None)),
        };
        thread::spawn(move || {
            let result = run(&progress).unwrap_or_else(|e| Event::Error(format!("{e:#}")));
            if !progress.token().is_cancelled() && progress.send.send(result).is_ok() {
                (progress.wake)();
            }
        });
        Self { events, cancel }
    }
}

impl Drop for Job {
    fn drop(&mut self) {
        self.cancel.token.cancel();
        if let Some(mut child) = self.cancel.child.lock().unwrap().take() {
            #[cfg(windows)]
            {
                use std::os::windows::process::CommandExt;
                let _ = Command::new("taskkill")
                    .creation_flags(0x08000000)
                    .args(["/T", "/F", "/PID", &child.id().to_string()])
                    .output();
            }
            #[cfg(unix)]
            {
                // SAFETY: Progress::command starts this child in its own process
                // group. Its negative PID signals the downloader and its helpers.
                unsafe { libc::kill(-(child.id() as libc::pid_t), libc::SIGTERM) };
            }
            let _ = child.kill();
            let _ = child.wait();
        }
        self.cancel.media_dir.lock().unwrap().take();
    }
}

impl Progress {
    pub fn media_dir(&self) -> Result<std::path::PathBuf> {
        let mut slot = self.cancel.media_dir.lock().unwrap();
        self.check()?;
        let dir = tempfile::tempdir()?;
        let path = dir.path().to_owned();
        *slot = Some(dir);
        Ok(path)
    }

    pub fn token(&self) -> &CancelToken {
        &self.cancel.token
    }

    pub fn check(&self) -> Result<()> {
        ensure!(!self.token().is_cancelled(), "Cancelled");
        Ok(())
    }

    pub fn update(&self, update: Update) -> Result<()> {
        self.check()?;
        self.send
            .send(Event::Update(update))
            .map_err(|_| anyhow::anyhow!("Cancelled"))?;
        (self.wake)();
        Ok(())
    }

    pub fn stage(&self, stage: Stage) -> Result<()> {
        self.sent.set((-1.0, None));
        self.update(Update::Stage(stage))
    }

    pub fn note(&self, note: impl Into<String>) -> Result<()> {
        self.update(Update::Note(note.into()))
    }

    /// The current stage's fraction, sent at most every 40 ms unless it finishes.
    pub fn fraction(&self, fraction: f32) -> Result<()> {
        self.check()?;
        let fraction = fraction.clamp(0.0, 1.0);
        let (last, at) = self.sent.get();
        let due = at.is_none_or(|t| t.elapsed() >= Duration::from_millis(40));
        if (fraction - last).abs() < 0.001 || !(due || fraction >= 1.0) {
            return Ok(());
        }
        self.sent.set((fraction, Some(Instant::now())));
        self.update(Update::Fraction(fraction))
    }

    fn spawn(&self, cmd: &mut Command, name: &str) -> Result<()> {
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            cmd.process_group(0);
        }
        let mut slot = self.cancel.child.lock().unwrap();
        self.check()?;
        *slot = Some(cmd.spawn().with_context(|| {
            format!("Could not start {name}. Install it or put it beside Vellum.")
        })?);
        Ok(())
    }

    /// Waits for the running tool, calling `tick` every 50 ms while it runs.
    fn wait(&self, mut tick: impl FnMut() -> Result<()>) -> Result<std::process::ExitStatus> {
        loop {
            {
                let mut slot = self.cancel.child.lock().unwrap();
                self.check()?;
                if let Some(status) = slot.as_mut().context("Cancelled")?.try_wait()? {
                    slot.take();
                    return Ok(status);
                }
            }
            tick()?;
            thread::sleep(Duration::from_millis(50));
        }
    }

    fn failure(name: &str, stderr: &mut std::fs::File) -> Result<()> {
        let mut diagnostic = String::new();
        stderr.seek(SeekFrom::Start(0))?;
        stderr.read_to_string(&mut diagnostic)?;
        bail!(
            "{name}: {}",
            diagnostic
                .lines()
                .rev()
                .find(|s| !s.trim().is_empty())
                .unwrap_or("command failed")
        )
    }

    /// Runs a tool to completion and returns its standard output.
    pub fn command(&self, cmd: &mut Command, name: &str) -> Result<Vec<u8>> {
        let mut stdout = tempfile::tempfile()?;
        let mut stderr = tempfile::tempfile()?;
        cmd.stdout(stdout.try_clone()?).stderr(stderr.try_clone()?);
        self.spawn(cmd, name)?;
        if !self.wait(|| Ok(()))?.success() {
            Self::failure(name, &mut stderr)?;
        }
        stdout.seek(SeekFrom::Start(0))?;
        let mut bytes = Vec::new();
        stdout.read_to_end(&mut bytes)?;
        Ok(bytes)
    }

    /// Runs a tool to completion, handing each line it prints to `line` as it prints it.
    pub fn watch(&self, cmd: &mut Command, name: &str, mut line: impl FnMut(&str)) -> Result<()> {
        let mut stderr = tempfile::tempfile()?;
        cmd.stdout(Stdio::piped()).stderr(stderr.try_clone()?);
        self.spawn(cmd, name)?;
        let stdout = self
            .cancel
            .child
            .lock()
            .unwrap()
            .as_mut()
            .and_then(|c| c.stdout.take())
            .context("Cancelled")?;
        let (lines, heard) = mpsc::channel::<String>();
        let reader = thread::spawn(move || {
            for text in BufReader::new(stdout).lines().map_while(Result::ok) {
                if lines.send(text).is_err() {
                    break;
                }
            }
        });
        let status = self.wait(|| {
            heard.try_iter().for_each(|text| line(&text));
            Ok(())
        })?;
        let _ = reader.join();
        heard.try_iter().for_each(|text| line(&text));
        if !status.success() {
            Self::failure(name, &mut stderr)?;
        }
        Ok(())
    }
}
