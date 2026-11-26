//! A small team of threads that splits one window's decoding work: each predictor step's and
//! joint evaluation's products are cut by output rows into one part per member, the calling
//! thread doing part 0. A step is a few tens of microseconds of work, far too little for a
//! general thread pool's wake-up, so the helpers spin while a window is being decoded, and
//! park when it's done (or after a while without work).
//!
//! Splitting by output rows changes no result: every output is still one whole dot product,
//! computed exactly as it would be on one thread.

use std::{
    panic::{AssertUnwindSafe, catch_unwind, resume_unwind},
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicPtr, AtomicU64, AtomicUsize, Ordering::SeqCst},
    },
    thread::JoinHandle,
    time::{Duration, Instant},
};

/// How long a helper spins for the next part before it parks: longer than any gap inside a
/// window's decoding, short enough not to keep cores busy between recordings.
const SPIN: Duration = Duration::from_micros(500);

/// A task as the helpers see it: a pointer to the caller's `&dyn Fn(usize)`, valid while
/// [`Team::run`] waits for them.
type Task<'a> = &'a (dyn Fn(usize) + Sync);

struct Shared {
    /// Bumped once per task; a helper runs the task when it sees a new value.
    epoch: AtomicU64,
    task: AtomicPtr<Task<'static>>,
    /// Helpers that finished the current task.
    done: AtomicUsize,
    /// Helpers parked (or about to park), so the caller knows to wake them.
    parked: AtomicUsize,
    /// No more work for now: helpers park at once instead of spinning.
    resting: AtomicBool,
    /// A helper's part panicked.
    panicked: AtomicBool,
    stop: AtomicBool,
}

pub struct Team {
    shared: Arc<Shared>,
    helpers: Vec<JoinHandle<()>>,
}

impl Team {
    /// A team of `size` (at least one: the caller alone, with no helper threads).
    pub fn new(size: usize) -> Team {
        let shared = Arc::new(Shared {
            epoch: AtomicU64::new(0),
            task: AtomicPtr::new(std::ptr::null_mut()),
            done: AtomicUsize::new(0),
            parked: AtomicUsize::new(0),
            resting: AtomicBool::new(false),
            panicked: AtomicBool::new(false),
            stop: AtomicBool::new(false),
        });
        let helpers = (1..size.max(1))
            .map(|part| {
                let shared = shared.clone();
                std::thread::Builder::new()
                    .name(format!("decoder helper {part}"))
                    .spawn(move || help(&shared, part))
                    .expect("a decoder helper thread")
            })
            .collect();
        Team { shared, helpers }
    }

    /// Members, the caller included.
    pub fn size(&self) -> usize {
        self.helpers.len() + 1
    }

    /// `task(part)` for every part in `0..size()`, part 0 on this thread; returns when all are
    /// done. A part that panics panics here, once every part is done.
    pub fn run(&self, task: &(dyn Fn(usize) + Sync)) {
        if self.helpers.is_empty() {
            task(0);
            return;
        }
        let s = &*self.shared;
        s.resting.store(false, SeqCst);
        // SAFETY (for the helpers' dereference): the pointer is to `task`, which outlives this
        // call, and this call doesn't return until every helper is done with it.
        let task_ref: Task = task;
        let erased: *const Task<'static> = (&raw const task_ref).cast();
        s.task.store(erased.cast_mut(), SeqCst);
        s.done.store(0, SeqCst);
        s.epoch.fetch_add(1, SeqCst);
        if s.parked.load(SeqCst) > 0 {
            for helper in &self.helpers {
                helper.thread().unpark();
            }
        }
        // The helpers hold a reference to `task` until they're done, so this waits for them
        // even when its own part panics.
        let mine = catch_unwind(AssertUnwindSafe(|| task(0)));
        let mut spins = 0u32;
        while s.done.load(SeqCst) < self.helpers.len() {
            // A helper that isn't running (the OS gave its core to someone else) would keep
            // this thread spinning for nothing: past a while, give the core away instead.
            spins += 1;
            if spins < 1 << 14 {
                std::hint::spin_loop();
            } else {
                std::thread::yield_now();
            }
        }
        if let Err(panic) = mine {
            resume_unwind(panic);
        }
        if s.panicked.swap(false, SeqCst) {
            panic!("a decoder helper panicked");
        }
    }

    /// Tells the helpers there's no more work for now, so they park at once and give their
    /// cores back. The next [`run`](Self::run) wakes them.
    pub fn rest(&self) {
        self.shared.resting.store(true, SeqCst);
    }
}

/// A helper's life: wait for each new task, run its part, report.
fn help(s: &Shared, part: usize) {
    let mut seen = 0;
    loop {
        let mut idle = Instant::now();
        let mut spins = 0u32;
        loop {
            if s.stop.load(SeqCst) {
                return;
            }
            let epoch = s.epoch.load(SeqCst);
            if epoch != seen {
                seen = epoch;
                break;
            }
            spins += 1;
            if spins.is_multiple_of(256) && (s.resting.load(SeqCst) || idle.elapsed() > SPIN) {
                // Announced before the last check, so a caller bumping the epoch after it sees
                // the announcement and wakes this thread.
                s.parked.fetch_add(1, SeqCst);
                if s.epoch.load(SeqCst) == seen && !s.stop.load(SeqCst) {
                    std::thread::park();
                }
                s.parked.fetch_sub(1, SeqCst);
                idle = Instant::now();
            } else {
                std::hint::spin_loop();
            }
        }
        // SAFETY: the caller set the task before bumping the epoch, and waits for `done`.
        let task: Task = unsafe { *s.task.load(SeqCst) };
        if catch_unwind(AssertUnwindSafe(|| task(part))).is_err() {
            s.panicked.store(true, SeqCst);
        }
        s.done.fetch_add(1, SeqCst);
    }
}

impl Drop for Team {
    fn drop(&mut self) {
        self.shared.stop.store(true, SeqCst);
        for helper in self.helpers.drain(..) {
            helper.thread().unpark();
            let _ = helper.join();
        }
    }
}

/// Part `part` of `0..n` cut into `parts` near-equal ranges whose bounds are multiples of
/// `align` (except the end).
pub fn split(n: usize, parts: usize, part: usize, align: usize) -> std::ops::Range<usize> {
    let chunks = n.div_ceil(align);
    let at = |p: usize| (chunks * p / parts * align).min(n);
    at(part)..at(part + 1)
}

/// A slice the parts of a task write at disjoint places.
pub struct Disjoint<'a> {
    ptr: *mut f32,
    len: usize,
    _slice: std::marker::PhantomData<&'a mut [f32]>,
}

// SAFETY: parts only ever write disjoint ranges (see `range`).
unsafe impl Send for Disjoint<'_> {}
unsafe impl Sync for Disjoint<'_> {}

impl<'a> Disjoint<'a> {
    pub fn new(slice: &'a mut [f32]) -> Disjoint<'a> {
        Disjoint {
            ptr: slice.as_mut_ptr(),
            len: slice.len(),
            _slice: std::marker::PhantomData,
        }
    }

    /// The values in `range`, to write.
    ///
    /// # Safety
    /// No other part reads or writes `range` while this one holds it.
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn range(&self, range: std::ops::Range<usize>) -> &mut [f32] {
        assert!(range.start <= range.end && range.end <= self.len);
        // SAFETY: in bounds (checked), and exclusive as the caller promises.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.add(range.start), range.len()) }
    }
}
