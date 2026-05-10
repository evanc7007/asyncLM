//! AsyncLM inferlet — paper-faithful minimal port of arXiv:2412.07017.
//!
//! Built as lib2.rs + the §5 features the paper insists on:
//!   • Critical-section flag (§5.1, §5.3): parser emits Enter/Exit events on
//!     state transitions in/out of CML blocks.
//!   • InterruptManager (§5.3): queues completed results; flushes them at the
//!     next decode-step boundary when not in a critical section.
//!   • Mid-stream injection (§5.3): every decode step drains the queue if the
//!     parser is in a clean Normal state — not only on [TRAP][END].
//!   • CheckpointManager (§5.4): forks the context on the first `[` of an
//!     entry delimiter so Swap and Recompute branches have a restore point.
//!   • TrapHandler (§5.4): 3-way Keep / Recompute / Swap decision using the
//!     paper's rule:
//!         if recompute_t > wait_t AND swap_t > wait_t → Keep
//!         else if recompute_t ≤ swap_t                → Recompute
//!         else                                          → Swap
//!     with paper-consistent scaling (recompute O(n²), swap O(n)).
//!
//! Dummy executor preserved verbatim from lib2.rs — calls return canned
//! strings after a wall-clock deadline.  No real broadcast/subscribe.
//!
//! To build this file, set `path = "src/lib3.rs"` in Cargo.toml.

use inferlet::context::Context as PieContext;
use inferlet::sampler::Sample;
use inferlet::stop_condition::{ends_with_any, max_len, StopCondition};
use inferlet::{Args, Result, Sampler, Tokenizer};
use std::collections::{HashMap, HashSet, VecDeque};
use std::future::{poll_fn, Future};
use std::pin::Pin;
use std::task::{Context as TaskCtx, Poll, RawWaker, RawWakerVTable, Waker};
use std::time::{Duration, Instant};

// ============================================================================
// CML token registry  (unchanged from lib2.rs)
// ============================================================================

struct CmlRegistry {
    call_ids: Vec<u32>,
    end_ids: Vec<u32>,
    head_ids: Vec<u32>,
    trap_ids: Vec<u32>,
    intr_ids: Vec<u32>,
    trap_end_ids: Vec<Vec<u32>>,
    suppressed: HashSet<u32>,
}

impl CmlRegistry {
    fn new(tokenizer: &Tokenizer) -> Self {
        let (special_ids, special_bytes) = tokenizer.get_special_tokens();
        let special_map: HashMap<Vec<u8>, u32> =
            special_bytes.into_iter().zip(special_ids.into_iter()).collect();

        let resolve = |s: &str| -> Vec<u32> {
            if let Some(&id) = special_map.get(s.as_bytes()) {
                vec![id]
            } else {
                let ids = tokenizer.tokenize(s);
                assert!(!ids.is_empty(), "CML token '{}' tokenized to empty", s);
                ids
            }
        };

        let call_ids = resolve("[CALL]");
        let end_ids = resolve("[END]");
        let head_ids = resolve("[HEAD]");
        let trap_ids = resolve("[TRAP]");
        let intr_ids = resolve("[INTR]");

        let mut trap_end_ids = vec![trap_ids.iter().chain(end_ids.iter()).cloned().collect()];
        let merge = tokenizer.tokenize("][");
        if merge.len() == 1 {
            let mut merged: Vec<u32> = trap_ids[..trap_ids.len() - 1].to_vec();
            merged.push(merge[0]);
            merged.extend_from_slice(&end_ids[1..]);
            trap_end_ids.push(merged);
        }

        let shared: HashSet<u32> = call_ids.iter()
            .chain(end_ids.iter())
            .chain(head_ids.iter())
            .chain(trap_ids.iter())
            .copied()
            .collect();
        let suppressed: HashSet<u32> = intr_ids.iter().copied().filter(|id| !shared.contains(id)).collect();

        CmlRegistry { call_ids, end_ids, head_ids, trap_ids, intr_ids, trap_end_ids, suppressed }
    }
}

// ============================================================================
// CML parser (FSM) — extended with critical-section events
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq)]
enum State { Normal, InCallId, InCallBody, InTrap }

#[derive(Debug)]
enum Event {
    Passthrough(u32),
    Call { id: String, code: String },
    Trap,
    /// Parser entered a CML block (InCallId / InCallBody / InTrap).
    /// §5.3 critical-section flag → false (interrupts deferred).
    EnterCriticalSection,
    /// Parser left a CML block back to Normal.
    /// §5.3 critical-section flag → true (interrupts may inject again).
    ExitCriticalSection,
    /// First `[` of an entry delimiter just landed.  Coordinator should
    /// snapshot the context here — §5.4 checkpoints precede the block so
    /// Swap/Recompute restore to a clean state.
    EnteringBlockHint,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Delim { Call, End, Head, Trap, TrapEnd, None }

struct Parser {
    call_ids: Vec<u32>,
    end_ids: Vec<u32>,
    head_ids: Vec<u32>,
    trap_ids: Vec<u32>,
    trap_end_ids: Vec<Vec<u32>>,
    call_inner: Vec<u32>,
    end_inner: Vec<u32>,
    head_inner: Vec<u32>,
    trap_inner: Vec<u32>,
    trap_end_inner: Vec<Vec<u32>>,
    state: State,
    id_tokens: Vec<u32>,
    body_tokens: Vec<u32>,
    buf: Vec<u32>,
    /// Whether we have already emitted EnteringBlockHint for the current buf.
    /// Reset on every state change.
    hint_emitted: bool,
}

impl Parser {
    fn new(reg: &CmlRegistry) -> Self {
        Parser {
            call_inner: reg.call_ids[1..].to_vec(),
            end_inner: reg.end_ids[1..].to_vec(),
            head_inner: reg.head_ids[1..].to_vec(),
            trap_inner: reg.trap_ids[1..].to_vec(),
            trap_end_inner: reg.trap_end_ids.iter().map(|s| s[1..].to_vec()).collect(),
            call_ids: reg.call_ids.clone(),
            end_ids: reg.end_ids.clone(),
            head_ids: reg.head_ids.clone(),
            trap_ids: reg.trap_ids.clone(),
            trap_end_ids: reg.trap_end_ids.clone(),
            state: State::Normal,
            id_tokens: Vec::new(),
            body_tokens: Vec::new(),
            buf: Vec::new(),
            hint_emitted: false,
        }
    }

    /// True iff the parser is in `Normal` state with no pending delimiter
    /// prefix — safe seam for mid-stream injection.
    fn is_clean(&self) -> bool {
        self.state == State::Normal && self.buf.is_empty()
    }

    fn ends_with(buf: &[u32], pat: &[u32]) -> bool {
        !pat.is_empty() && buf.len() >= pat.len() && buf[buf.len() - pat.len()..] == *pat
    }

    fn is_prefix(buf: &[u32], pat: &[u32]) -> bool {
        !pat.is_empty() && buf.len() <= pat.len() && pat[..buf.len()] == *buf
    }

    fn match_full(&self, buf: &[u32]) -> (Delim, usize) {
        if Self::ends_with(buf, &self.call_ids) { return (Delim::Call, self.call_ids.len()); }
        if Self::ends_with(buf, &self.end_ids) { return (Delim::End, self.end_ids.len()); }
        for s in &self.trap_end_ids {
            if Self::ends_with(buf, s) { return (Delim::TrapEnd, s.len()); }
        }
        if Self::ends_with(buf, &self.trap_ids) { return (Delim::Trap, self.trap_ids.len()); }
        if Self::ends_with(buf, &self.head_ids) { return (Delim::Head, self.head_ids.len()); }
        (Delim::None, 0)
    }

    fn match_inner(&self, buf: &[u32]) -> (Delim, usize) {
        if Self::ends_with(buf, &self.call_inner) { return (Delim::Call, self.call_inner.len()); }
        if Self::ends_with(buf, &self.end_inner) { return (Delim::End, self.end_inner.len()); }
        if Self::ends_with(buf, &self.trap_inner) { return (Delim::Trap, self.trap_inner.len()); }
        if Self::ends_with(buf, &self.head_inner) { return (Delim::Head, self.head_inner.len()); }
        (Delim::None, 0)
    }

    fn match_normal_inner(&self, buf: &[u32]) -> (Delim, usize) {
        for s in &self.trap_end_inner {
            if Self::ends_with(buf, s) { return (Delim::TrapEnd, s.len()); }
        }
        if Self::ends_with(buf, &self.call_inner) { return (Delim::Call, self.call_inner.len()); }
        if Self::ends_with(buf, &self.trap_inner) { return (Delim::Trap, self.trap_inner.len()); }
        (Delim::None, 0)
    }

    fn is_prefix_of_any(&self, buf: &[u32]) -> bool {
        Self::is_prefix(buf, &self.call_ids)
            || Self::is_prefix(buf, &self.end_ids)
            || Self::is_prefix(buf, &self.trap_ids)
            || Self::is_prefix(buf, &self.head_ids)
            || self.trap_end_ids.iter().any(|s| Self::is_prefix(buf, s))
            || Self::is_prefix(buf, &self.call_inner)
            || Self::is_prefix(buf, &self.end_inner)
            || Self::is_prefix(buf, &self.trap_inner)
            || Self::is_prefix(buf, &self.head_inner)
            || self.trap_end_inner.iter().any(|s| Self::is_prefix(buf, s))
    }

    fn strip_trailing_bracket(tokens: &mut Vec<u32>, tokenizer: &Tokenizer) {
        if let Some(&last) = tokens.last() {
            if tokenizer.detokenize(&[last]).ends_with('[') {
                tokens.pop();
            }
        }
    }

    fn feed(&mut self, token_id: u32, tokenizer: &Tokenizer) -> Vec<Event> {
        let mut events = Vec::new();
        let prev_state = self.state;

        match self.state {
            State::Normal => {
                self.buf.push(token_id);

                // §5.4 checkpoint hint: emit the moment buf becomes a strict
                // prefix of an entry delimiter (only [CALL] and [TRAP] count
                // as entries; [END] and [HEAD] never start a block).
                if !self.hint_emitted
                    && (Self::is_prefix(&self.buf, &self.call_ids)
                        || Self::is_prefix(&self.buf, &self.trap_ids)
                        || self.trap_end_ids.iter().any(|s| Self::is_prefix(&self.buf, s)))
                {
                    events.push(Event::EnteringBlockHint);
                    self.hint_emitted = true;
                }

                let (d, dlen) = {
                    let (d, l) = self.match_full(&self.buf);
                    if d != Delim::None { (d, l) } else { self.match_normal_inner(&self.buf) }
                };
                match d {
                    Delim::Call => {
                        let end = self.buf.len() - dlen;
                        let mut prefix: Vec<u32> = self.buf[..end].to_vec();
                        Self::strip_trailing_bracket(&mut prefix, tokenizer);
                        for t in prefix { events.push(Event::Passthrough(t)); }
                        self.buf.clear();
                        self.id_tokens.clear();
                        self.body_tokens.clear();
                        self.state = State::InCallId;
                    }
                    Delim::Trap => {
                        let end = self.buf.len() - dlen;
                        let mut prefix: Vec<u32> = self.buf[..end].to_vec();
                        Self::strip_trailing_bracket(&mut prefix, tokenizer);
                        for t in prefix { events.push(Event::Passthrough(t)); }
                        self.buf.clear();
                        self.state = State::InTrap;
                    }
                    Delim::TrapEnd => {
                        let end = self.buf.len() - dlen;
                        let mut prefix: Vec<u32> = self.buf[..end].to_vec();
                        Self::strip_trailing_bracket(&mut prefix, tokenizer);
                        for t in prefix { events.push(Event::Passthrough(t)); }
                        self.buf.clear();
                        events.push(Event::Trap);
                        // Stays in Normal; reset hint so next entry re-fires.
                        self.hint_emitted = false;
                    }
                    Delim::None => {
                        while !self.buf.is_empty() && !self.is_prefix_of_any(&self.buf) {
                            let t = self.buf.remove(0);
                            if tokenizer.detokenize(&[t]).ends_with('[') {
                                continue;
                            }
                            events.push(Event::Passthrough(t));
                        }
                        // If buf drained, the entry-delimiter prefix dissolved
                        // — re-arm the hint so a future [CALL]/[TRAP] still
                        // triggers a checkpoint.
                        if self.buf.is_empty() {
                            self.hint_emitted = false;
                        }
                    }
                    _ => {
                        for &t in &self.buf {
                            if tokenizer.detokenize(&[t]).ends_with('[') {
                                continue;
                            }
                            events.push(Event::Passthrough(t));
                        }
                        self.buf.clear();
                        self.hint_emitted = false;
                    }
                }
            }

            State::InCallId => {
                self.buf.push(token_id);
                let (d, dlen) = {
                    let (d, l) = self.match_full(&self.buf);
                    if d != Delim::None { (d, l) } else { self.match_inner(&self.buf) }
                };
                match d {
                    Delim::Head => {
                        let id_end = self.buf.len() - dlen;
                        self.id_tokens.extend_from_slice(&self.buf[..id_end]);
                        self.buf.clear();
                        Self::strip_trailing_bracket(&mut self.id_tokens, tokenizer);
                        self.state = State::InCallBody;
                    }
                    Delim::End => {
                        let id_end = self.buf.len() - dlen;
                        self.id_tokens.extend_from_slice(&self.buf[..id_end]);
                        self.buf.clear();
                        Self::strip_trailing_bracket(&mut self.id_tokens, tokenizer);
                        let id = tokenizer.detokenize(&self.id_tokens).trim().to_string();
                        events.push(Event::Call { id, code: String::new() });
                        self.state = State::Normal;
                    }
                    Delim::None => {
                        if !self.is_prefix_of_any(&self.buf) {
                            self.id_tokens.extend_from_slice(&self.buf);
                            self.buf.clear();
                        }
                    }
                    _ => {}
                }
            }

            State::InCallBody => {
                self.buf.push(token_id);
                let (d, dlen) = {
                    let (d, l) = self.match_full(&self.buf);
                    if d != Delim::None { (d, l) } else { self.match_inner(&self.buf) }
                };
                match d {
                    Delim::End => {
                        let body_end = self.buf.len() - dlen;
                        self.body_tokens.extend_from_slice(&self.buf[..body_end]);
                        self.buf.clear();
                        Self::strip_trailing_bracket(&mut self.body_tokens, tokenizer);
                        let id = tokenizer.detokenize(&self.id_tokens).trim().to_string();
                        let code = tokenizer.detokenize(&self.body_tokens).trim().to_string();
                        events.push(Event::Call { id, code });
                        self.state = State::Normal;
                    }
                    Delim::None => {
                        if !self.is_prefix_of_any(&self.buf) {
                            self.body_tokens.extend_from_slice(&self.buf);
                            self.buf.clear();
                        }
                    }
                    _ => {}
                }
            }

            State::InTrap => {
                self.buf.push(token_id);
                let (d, _) = {
                    let (d, l) = self.match_full(&self.buf);
                    if d != Delim::None { (d, l) } else { self.match_inner(&self.buf) }
                };
                match d {
                    Delim::End => {
                        self.buf.clear();
                        events.push(Event::Trap);
                        self.state = State::Normal;
                    }
                    Delim::None => {
                        if !self.is_prefix_of_any(&self.buf) { self.buf.clear(); }
                    }
                    _ => {}
                }
            }
        }

        // §5.3 critical-section transitions: emit Enter on Normal→non-Normal,
        // Exit on non-Normal→Normal.  Order matters: the events list reflects
        // the within-step sequence, so flush state change *after* the events
        // produced by this step (the Call / Trap event already fired).
        if prev_state == State::Normal && self.state != State::Normal {
            events.push(Event::EnterCriticalSection);
            self.hint_emitted = true; // already in a block, no further hint needed
        } else if prev_state != State::Normal && self.state == State::Normal {
            events.push(Event::ExitCriticalSection);
            self.hint_emitted = false;
        }

        events
    }

    fn flush(&mut self, tokenizer: &Tokenizer) -> Vec<Event> {
        self.buf
            .drain(..)
            .filter(|&t| !tokenizer.detokenize(&[t]).ends_with('['))
            .map(Event::Passthrough)
            .collect()
    }
}

// ============================================================================
// Sampler — INTR suppression  (unchanged from lib2.rs)
// ============================================================================

struct SuppressingSampler {
    suppressed: HashSet<u32>,
    top_p: f32,
}

impl Sample for SuppressingSampler {
    fn sample(&self, ids: &[u32], probs: &[f32]) -> u32 {
        let mut pairs: Vec<(u32, f32)> = ids.iter().zip(probs.iter())
            .map(|(&id, &p)| if self.suppressed.contains(&id) { (id, 0.0) } else { (id, p) })
            .collect();
        pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut cum = 0.0f32;
        let mut cutoff = pairs.len();
        for (i, &(_, p)) in pairs.iter().enumerate() {
            cum += p;
            if cum >= self.top_p { cutoff = i + 1; break; }
        }
        pairs[..cutoff].iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|&(id, _)| id)
            .unwrap_or(pairs[0].0)
    }
}

// ============================================================================
// Dummy executor  (unchanged from lib2.rs — preserved by request)
// ============================================================================

fn noop_waker() -> Waker {
    const VTABLE: RawWakerVTable =
        RawWakerVTable::new(|_| RawWaker::new(std::ptr::null(), &VTABLE), |_| {}, |_| {}, |_| {});
    unsafe { Waker::from_raw(RawWaker::new(std::ptr::null(), &VTABLE)) }
}

async fn execute_call(id: String, code: String, wait_ms: u64) -> String {
    let deadline = Instant::now() + Duration::from_millis(wait_ms);
    poll_fn(|cx| {
        if Instant::now() >= deadline {
            Poll::Ready(())
        } else {
            cx.waker().wake_by_ref();
            Poll::Pending
        }
    })
    .await;

    inferlet::send(&format!("[Async3] call id={} completed (wait {}ms)", id, wait_ms));
    let lc = code.to_lowercase();
    if lc.contains("get_weather") || lc.contains("weather(") {
        if lc.contains("london") { "{\"temp_f\": 60, \"sky\": \"cloudy\"}".to_string() }
        else if lc.contains("paris") { "{\"temp_f\": 59, \"sky\": \"rainy\"}".to_string() }
        else if lc.contains("boston") { "{\"temp_f\": 68, \"sky\": \"clear\"}".to_string() }
        else if lc.contains("tokyo") { "{\"temp_f\": 75, \"sky\": \"humid\"}".to_string() }
        else { "{\"temp_f\": 72, \"sky\": \"sunny\"}".to_string() }
    } else if lc.contains("stock_price") || lc.contains("get_stock") {
        if lc.contains("aapl") { "{\"ticker\": \"AAPL\", \"price_usd\": 189.50}".to_string() }
        else if lc.contains("goog") { "{\"ticker\": \"GOOG\", \"price_usd\": 141.20}".to_string() }
        else if lc.contains("tsla") { "{\"ticker\": \"TSLA\", \"price_usd\": 258.30}".to_string() }
        else { "{\"ticker\": \"UNKNOWN\", \"price_usd\": 100.00}".to_string() }
    } else if lc.contains("convert_currency") || lc.contains("exchange_rate") {
        if lc.contains("eur") { "{\"rate\": 0.92, \"quote\": \"USD->EUR\"}".to_string() }
        else if lc.contains("jpy") { "{\"rate\": 155.40, \"quote\": \"USD->JPY\"}".to_string() }
        else if lc.contains("gbp") { "{\"rate\": 0.79, \"quote\": \"USD->GBP\"}".to_string() }
        else { "{\"rate\": 1.00, \"quote\": \"USD->USD\"}".to_string() }
    } else if lc.contains("search_restaurants") || lc.contains("find_restaurants") {
        if lc.contains("tokyo") { "[\"Sukiyabashi Jiro\", \"Ichiran\", \"Sushi Saito\"]".to_string() }
        else if lc.contains("paris") { "[\"Le Jules Verne\", \"L'Ami Jean\", \"Septime\"]".to_string() }
        else { "[\"Katz's Deli\", \"Joe's Pizza\", \"Lombardi's\"]".to_string() }
    } else if lc.contains("get_reviews") || lc.contains("reviews(") {
        if lc.contains("katz") { "{\"stars\": 4.6, \"summary\": \"Iconic pastrami sandwiches\"}".to_string() }
        else if lc.contains("jiro") { "{\"stars\": 4.9, \"summary\": \"Legendary omakase, tiny counter\"}".to_string() }
        else if lc.contains("joe") { "{\"stars\": 4.4, \"summary\": \"Classic NY slice, cheap and fast\"}".to_string() }
        else { "{\"stars\": 4.0, \"summary\": \"Solid, well-reviewed spot\"}".to_string() }
    } else if lc.contains("get_time") || lc.contains("time_in") || lc.contains("current_time") {
        if lc.contains("tokyo") || lc.contains("jst") { "{\"time\": \"2026-04-21T23:32+09:00\", \"tz\": \"JST\"}".to_string() }
        else if lc.contains("london") || lc.contains("gmt") { "{\"time\": \"2026-04-21T15:32+01:00\", \"tz\": \"BST\"}".to_string() }
        else if lc.contains("new_york") || lc.contains("nyc") || lc.contains("est") { "{\"time\": \"2026-04-21T10:32-04:00\", \"tz\": \"EDT\"}".to_string() }
        else { "{\"time\": \"2026-04-21T14:32+00:00\", \"tz\": \"UTC\"}".to_string() }
    } else {
        format!("result({})", code)
    }
}

struct Pending {
    id: String,
    dispatched_at: Instant,
    /// Total expected wait in ms — used by TrapHandler to estimate `wait_t`.
    wait_ms: u64,
    fut: Pin<Box<dyn Future<Output = String>>>,
}

impl Pending {
    /// Best-effort remaining wait in ms (saturating to 0).
    fn remaining_ms(&self) -> u64 {
        let elapsed = self.dispatched_at.elapsed().as_millis() as u64;
        self.wait_ms.saturating_sub(elapsed)
    }
}

fn poll_once(p: &mut Pending) -> Option<String> {
    let waker = noop_waker();
    let mut cx = TaskCtx::from_waker(&waker);
    match p.fut.as_mut().poll(&mut cx) {
        Poll::Ready(r) => Some(r),
        Poll::Pending => None,
    }
}

// ============================================================================
// §5.3 Interrupt Manager — queue + critical-section gate
// ============================================================================

struct InterruptManager {
    /// Queued frames waiting to be injected.  Format: " [INTR] id [HEAD] value [END] ".
    queue: VecDeque<String>,
    /// Paper invariant: false while inside a CML block, true otherwise.
    in_critical_section: bool,
}

impl InterruptManager {
    fn new() -> Self {
        InterruptManager { queue: VecDeque::new(), in_critical_section: false }
    }

    fn enqueue(&mut self, id: &str, result: &str) {
        self.queue.push_back(format!(" [INTR] {} [HEAD] {} [END] ", id, result));
    }

    fn set_critical(&mut self, val: bool) {
        self.in_critical_section = val;
    }

    /// Returns frames to inject this step.  Empty while in a critical section.
    fn drain(&mut self) -> Vec<String> {
        if self.in_critical_section {
            return Vec::new();
        }
        self.queue.drain(..).collect()
    }

    fn pending_count(&self) -> usize {
        self.queue.len()
    }
}

// ============================================================================
// §5.4 Checkpoint Manager — fork() the context before a CML block
// ============================================================================

struct Checkpoint {
    ctx: PieContext,
    committed_len: usize,
    generated_len: usize,
}

struct CheckpointManager {
    current: Option<Checkpoint>,
}

impl CheckpointManager {
    fn new() -> Self { CheckpointManager { current: None } }

    fn save(&mut self, ctx: &PieContext, generated_len: usize) {
        self.current = Some(Checkpoint {
            ctx: ctx.fork(),
            committed_len: ctx.token_ids.len(),
            generated_len,
        });
    }

    fn has(&self) -> bool { self.current.is_some() }

    fn take(&mut self) -> Option<Checkpoint> { self.current.take() }

    fn discard(&mut self) { self.current = None; }

    fn committed_len(&self) -> Option<usize> {
        self.current.as_ref().map(|c| c.committed_len)
    }
}

// ============================================================================
// §5.4 Trap Handler — paper's 3-way Keep / Recompute / Swap comparator
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq)]
enum TrapStrategy { Keep, Recompute, Swap }

struct CostModel {
    /// Recompute scales O(n²): paper's stated cost.  ns_per_tok2 scales the
    /// quadratic term so total ms ≈ alpha * n² / 1e6.
    alpha_ns_per_tok2: f64,
    /// Swap scales O(n): linear.  ms ≈ beta * n / 1e6.
    beta_ns_per_tok: f64,
}

impl CostModel {
    fn recompute_ms(&self, n: usize) -> f64 {
        self.alpha_ns_per_tok2 * (n as f64) * (n as f64) / 1.0e6
    }
    fn swap_ms(&self, n: usize) -> f64 {
        self.beta_ns_per_tok * (n as f64) / 1.0e6
    }
}

/// §5.4 decision.  Implements the paper's rule verbatim:
///
/// > the trap handler keeps the KV cache in GPU memory if both recompute and
/// > swap times exceed the estimated wait time. Otherwise, it opts to
/// > recompute if recompute latency is lower, or to swap if swap latency is
/// > lower.
fn decide_trap(
    has_checkpoint: bool,
    tokens_since_checkpoint: usize,
    wait_ms: f64,
    cost: &CostModel,
) -> (TrapStrategy, f64, f64) {
    let r_ms = cost.recompute_ms(tokens_since_checkpoint);
    let s_ms = cost.swap_ms(tokens_since_checkpoint);

    if !has_checkpoint {
        // No restore point → can only Keep.
        return (TrapStrategy::Keep, r_ms, s_ms);
    }
    if r_ms > wait_ms && s_ms > wait_ms {
        return (TrapStrategy::Keep, r_ms, s_ms);
    }
    if r_ms <= s_ms {
        (TrapStrategy::Recompute, r_ms, s_ms)
    } else {
        (TrapStrategy::Swap, r_ms, s_ms)
    }
}

// ============================================================================
// Main
// ============================================================================

#[inferlet::main]
async fn main(mut args: Args) -> Result<String> {
    let prompt: String = args.value_from_str(["-p", "--prompt"])?;
    let max_tokens: usize = args.value_from_str(["-n", "--max-tokens"]).unwrap_or(512);
    let system: String = args
        .value_from_str(["-s", "--system"])
        .unwrap_or_else(|_| {
            "You are an assistant that uses async function calls (AsyncLM / CML\n\
             protocol) to answer user questions.\n\
             \n\
             CML syntax (use these delimiters EXACTLY):\n\
             - Dispatch (non-blocking): [CALL] <id> [HEAD] <code> [END]\n\
             - Wait for pending calls:  [TRAP][END]\n\
             - Result (runtime-injected, you only read): [INTR] <id> [HEAD] <value> [END]\n\
             \n\
             Tools available:\n\
             - get_weather(city: str) -> {\"temp_f\": int, \"sky\": str}\n\
             - get_stock_price(ticker: str) -> {\"ticker\": str, \"price_usd\": float}\n\
             - convert_currency(amount: float, from: str, to: str) -> {\"rate\": float, \"quote\": str}\n\
             - search_restaurants(city: str) -> [str, ...]\n\
             - get_reviews(name: str) -> {\"stars\": float, \"summary\": str}\n\
             - get_time(timezone: str) -> {\"time\": str, \"tz\": str}\n\
             \n\
             Semantics:\n\
             - [CALL] blocks are non-blocking. Dispatch all independent calls\n\
               back-to-back so they run in parallel.\n\
             - [TRAP][END] pauses generation until every pending [CALL] has\n\
               returned an [INTR]. Only emit [TRAP][END] when you actually need\n\
               a result to decide what to do next.\n\
             - BETWEEN your [CALL] dispatches and the [TRAP][END], write any\n\
               reasoning or prose that does NOT depend on the call results.\n\
               Calls run in parallel with your writing, so prose before [TRAP]\n\
               is free latency-wise.  Delay [TRAP][END] as long as you still\n\
               have result-independent things to say.\n\
             - After [INTR] frames appear you may EITHER (a) dispatch another\n\
               round of [CALL]s whose inputs depend on those results, then\n\
               [TRAP][END] again, OR (b) write the final natural-language\n\
               answer and stop. There is no fixed number of rounds.\n\
             - If part of the question is outside these tools, answer that\n\
               part from your own knowledge — do NOT invent a [CALL] for it.\n\
             - Do not put CML syntax inside <think> tags.\n\
             - Do not fabricate [INTR] frames yourself; the runtime produces\n\
               them.\n\
             - Do NOT emit [TRAP][END] when you have no pending calls.\n\
             \n\
             Example 1 — single round, two parallel calls:\n\
             User: What's the weather in NYC and London?\n\
             Assistant: [CALL] w1 [HEAD] get_weather(\"New York\") [END]\n\
             [CALL] w2 [HEAD] get_weather(\"London\") [END]\n\
             [TRAP][END]\n\
             [INTR] w1 [HEAD] {\"temp_f\": 72, \"sky\": \"sunny\"} [END]\n\
             [INTR] w2 [HEAD] {\"temp_f\": 60, \"sky\": \"cloudy\"} [END]\n\
             NYC is 72°F and sunny; London is 60°F and cloudy.\n\
             \n\
             Example 2 — interleaved prose hides call latency:\n\
             User: Get the weather in Paris and the time in Tokyo, then tell\n\
             me if it's a reasonable hour to video-call Tokyo from Paris.\n\
             Assistant: [CALL] w1 [HEAD] get_weather(\"Paris\") [END]\n\
             [CALL] t1 [HEAD] get_time(\"Asia/Tokyo\") [END]\n\
             Both calls dispatched in parallel. While they run: Paris weather\n\
             tells us whether the caller is indoors and free; Tokyo local time\n\
             tells us whether the other side is awake — Tokyo is roughly 7-8\n\
             hours ahead of Paris, so a Paris afternoon maps to a Tokyo late\n\
             night.\n\
             [TRAP][END]\n\
             [INTR] w1 [HEAD] {\"temp_f\": 62, \"sky\": \"overcast\"} [END]\n\
             [INTR] t1 [HEAD] {\"time\": \"07:30\", \"tz\": \"Asia/Tokyo\"} [END]\n\
             Paris is 62°F and overcast; Tokyo is 07:30 — good window."
                .to_string()
        });
    let temperature: f32 = args.value_from_str(["-t", "--temperature"]).unwrap_or(0.6);
    let top_p: f32 = args.value_from_str("--top-p").unwrap_or(0.95);
    let fake_wait_ms: u64 = args.value_from_str("--fake-wait-ms").unwrap_or(500);
    // Cost model knobs — tune to make Keep/Recompute/Swap branches reachable
    // in the demo.  Defaults pick small values so short contexts stay in Keep.
    let alpha: f64 = args.value_from_str("--alpha-recompute-ns-per-tok2").unwrap_or(800.0);
    let beta: f64 = args.value_from_str("--beta-swap-ns-per-tok").unwrap_or(400_000.0);
    // §5.4 checkpointing requires `ctx.fork()`, which is non-trivial.  Default
    // off so the basic demo path stays cheap; turn on only when you want to
    // exercise the Swap / Recompute branches.  Without checkpoints, every
    // trap falls through to Keep, identical in behavior to lib2.rs.
    let enable_checkpoints: bool = args.contains("--enable-checkpoints");
    // §5.3 mid-stream injection.  Off by default: with weak few-shot
    // adherence, models can loop forever — emit [CALL], get a mid-stream
    // [INTR] back, emit another [CALL], ad infinitum, until OOM.  Trap-only
    // injection (lib2.rs behavior) bounds the loop to TRAP boundaries.
    let enable_midstream: bool = args.contains("--enable-midstream");
    // Hard ceiling on decode steps regardless of generated.len().  Catches
    // runaway loops where most tokens are absorbed by the parser (CML
    // delimiter prefixes, [CALL] bodies) so `generated` never reaches
    // max_tokens.  Defaults to 4× max_tokens, override with --max-steps.
    let max_steps: u32 = args
        .value_from_str("--max-steps")
        .unwrap_or((max_tokens.saturating_mul(4)) as u32);

    let model = inferlet::get_auto_model();
    let tokenizer = model.get_tokenizer();
    let registry = CmlRegistry::new(&tokenizer);

    inferlet::send(&format!(
        "[Async3] call={:?} head={:?} end={:?} trap={:?} intr={:?}",
        registry.call_ids, registry.head_ids, registry.end_ids, registry.trap_ids, registry.intr_ids
    ));

    let sampler = Sampler::Custom {
        temperature,
        sampler: Box::new(SuppressingSampler {
            suppressed: registry.suppressed.clone(),
            top_p,
        }),
        penalties: Default::default(),
    };
    inferlet::send("[Async3] sampler built");

    let cost = CostModel { alpha_ns_per_tok2: alpha, beta_ns_per_tok: beta };

    let mut ctx = model.create_context();
    inferlet::send("[Async3] context created");
    ctx.fill_system(&system);
    inferlet::send(&format!("[Async3] system filled, committed={} pending={}", ctx.token_ids.len(), ctx.token_ids_pending.len()));
    ctx.fill_user(&prompt);
    inferlet::send(&format!("[Async3] user filled, committed={} pending={}", ctx.token_ids.len(), ctx.token_ids_pending.len()));

    let eos_tokens = model.eos_tokens();
    let stop = max_len(max_tokens).or(ends_with_any(eos_tokens.clone()));

    let mut parser = Parser::new(&registry);
    let mut generated: Vec<u32> = Vec::new();

    let mut pending: Vec<Pending> = Vec::new();
    let mut interrupts = InterruptManager::new();
    let mut checkpoints = CheckpointManager::new();

    let mut step: u32 = 0;
    loop {
        if step < 5 || step % 50 == 0 {
            inferlet::send(&format!("[Async3] decode_step #{} starting", step));
        }
        let token_id = ctx.decode_step(&sampler).await;
        if step < 5 || step % 50 == 0 {
            inferlet::send(&format!("[Async3] decode_step #{} → tok={}", step, token_id));
        }
        step += 1;
        ctx.fill_token(token_id);

        // Poll every in-flight call once — concurrent with token generation.
        pending.retain_mut(|p| match poll_once(p) {
            Some(result) => {
                inferlet::send(&format!(
                    "[Async3] id={} ready after {}ms",
                    p.id,
                    p.dispatched_at.elapsed().as_millis()
                ));
                interrupts.enqueue(&p.id, &result);
                false
            }
            None => true,
        });

        for event in parser.feed(token_id, &tokenizer) {
            match event {
                Event::Passthrough(t) => generated.push(t),
                Event::EnteringBlockHint => {
                    // §5.4 — snapshot context before the block enters KV so a
                    // later Swap or Recompute can restore to a clean state.
                    // Gated: ctx.fork() is non-trivial, and Keep needs no
                    // checkpoint, so default builds skip this path.
                    if enable_checkpoints && !checkpoints.has() {
                        checkpoints.save(&ctx, generated.len());
                        inferlet::send(&format!(
                            "[Async3] checkpoint @ committed={} generated={}",
                            ctx.token_ids.len(), generated.len()
                        ));
                    }
                }
                Event::EnterCriticalSection => {
                    interrupts.set_critical(true);
                }
                Event::ExitCriticalSection => {
                    interrupts.set_critical(false);
                }
                Event::Call { id, code } => {
                    inferlet::send(&format!(
                        "[Async3] dispatching id={} code={} (fake_wait={}ms)",
                        id, code, fake_wait_ms
                    ));
                    pending.push(Pending {
                        id: id.clone(),
                        dispatched_at: Instant::now(),
                        wait_ms: fake_wait_ms,
                        fut: Box::pin(execute_call(id, code, fake_wait_ms)),
                    });
                }
                Event::Trap => {
                    handle_trap(
                        &mut ctx,
                        &mut generated,
                        &mut pending,
                        &mut interrupts,
                        &mut checkpoints,
                        &cost,
                    ).await;
                }
            }
        }

        // §5.3 mid-stream injection: only when not in a critical section AND
        // the parser is in a clean Normal state (buf empty).  This is the
        // paper's "during each decoding step ... appends them" insertion.
        // Gated: with weak prompt adherence, mid-stream injection can loop.
        if enable_midstream && !interrupts.in_critical_section && parser.is_clean() {
            for frame in interrupts.drain() {
                inferlet::send(&format!("[Async3] mid-stream inject: {}", frame.trim()));
                ctx.fill(&frame);
            }
        }

        if stop.check(&generated) { break; }
        if step >= max_steps {
            inferlet::send(&format!(
                "[Async3] hit max_steps={} (generated={}); breaking",
                max_steps, generated.len()
            ));
            break;
        }
    }

    for ev in parser.flush(&tokenizer) {
        if let Event::Passthrough(t) = ev { generated.push(t); }
    }

    Ok(tokenizer.detokenize(&generated))
}

/// §5.4 trap handler.  Drains pending futures and decides Keep/Recompute/Swap.
async fn handle_trap(
    ctx: &mut PieContext,
    generated: &mut Vec<u32>,
    pending: &mut Vec<Pending>,
    interrupts: &mut InterruptManager,
    checkpoints: &mut CheckpointManager,
    cost: &CostModel,
) {
    if pending.is_empty() && interrupts.pending_count() == 0 {
        inferlet::send("[Async3] trap with nothing pending — no-op");
        return;
    }

    // wait_t = max remaining wait across in-flight futures.  If everything is
    // already ready, wait_t = 0 and Keep is forced (any non-zero cost > 0).
    let wait_ms: f64 = pending
        .iter()
        .map(|p| p.remaining_ms() as f64)
        .fold(0.0f64, f64::max);

    let context_len = ctx.token_ids.len() + ctx.token_ids_pending.len();
    let tokens_since_checkpoint = checkpoints
        .committed_len()
        .map(|cp| context_len.saturating_sub(cp))
        .unwrap_or(context_len);

    let (strategy, r_ms, s_ms) = decide_trap(
        checkpoints.has(),
        tokens_since_checkpoint,
        wait_ms,
        cost,
    );
    inferlet::send(&format!(
        "[Async3] trap decision: {:?}  wait_t={:.1}ms recompute_t={:.1}ms swap_t={:.1}ms n={}",
        strategy, wait_ms, r_ms, s_ms, tokens_since_checkpoint
    ));

    // Drain all pending futures (paper: trap blocks until results arrive).
    let trap_start = Instant::now();
    while !pending.is_empty() {
        pending.retain_mut(|p| match poll_once(p) {
            Some(result) => {
                interrupts.enqueue(&p.id, &result);
                false
            }
            None => true,
        });
    }
    inferlet::send(&format!(
        "[Async3] trap drained in {}ms",
        trap_start.elapsed().as_millis()
    ));

    match strategy {
        TrapStrategy::Keep => {
            interrupts.set_critical(false);
            for frame in interrupts.drain() {
                inferlet::send(&format!("[Async3] keep-inject: {}", frame.trim()));
                ctx.fill(&frame);
            }
            checkpoints.discard();
        }
        TrapStrategy::Swap => {
            if let Some(cp) = checkpoints.take() {
                *ctx = cp.ctx;
                generated.truncate(cp.generated_len);
                interrupts.set_critical(false);
                for frame in interrupts.drain() {
                    inferlet::send(&format!("[Async3] swap-inject: {}", frame.trim()));
                    ctx.fill(&frame);
                }
            }
        }
        TrapStrategy::Recompute => {
            if let Some(cp) = checkpoints.take() {
                let target = cp.committed_len;
                let committed_to_remove = ctx.token_ids.len().saturating_sub(target);
                if committed_to_remove > 0 {
                    ctx.shrink_kv_pages(committed_to_remove);
                    ctx.token_ids.truncate(target);
                    ctx.position_ids.truncate(target);
                }
                ctx.token_ids_pending.clear();
                ctx.token_mask_pending.clear();
                let mask_len = target + committed_to_remove + 1;
                ctx.token_mask_current.remove_range(target, mask_len);
                generated.truncate(cp.generated_len);
                interrupts.set_critical(false);
                for frame in interrupts.drain() {
                    inferlet::send(&format!("[Async3] recompute-inject: {}", frame.trim()));
                    ctx.fill(&frame);
                }
            }
        }
    }
}
