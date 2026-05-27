//! AsyncLM inferlet — paper-faithful port of arXiv:2412.07017 (current SDK).
//!
//! Extends v2 with the §5 features the paper insists on:
//!   * Critical-section flag (§5.1, §5.3): parser emits Enter/Exit events on
//!     state transitions in/out of CML blocks.
//!   * InterruptManager (§5.3): queues completed results; flushes them at the
//!     next decode-step boundary when not in a critical section.
//!   * Mid-stream injection (§5.3): each decode step drains the queue if the
//!     parser is in a clean Normal state — not only at [TRAP][END].
//!   * TrapHandler classifier (§5.4): paper's 3-way Keep / Recompute / Swap
//!     decision using `if recompute_t > wait_t AND swap_t > wait_t → Keep`,
//!     with recompute O(n²) and swap O(n). Decision is recorded for
//!     observability; the current SDK's `Context` does not expose the
//!     restore primitives needed to execute Swap/Recompute (committed-page
//!     truncation + in-place context replacement under a `Generator` borrow),
//!     so all branches collapse to Keep at execution time.

use std::collections::{HashMap, HashSet, VecDeque};
use std::future::{poll_fn, Future};
use std::pin::Pin;
use std::task::{Context as TaskCtx, Poll, RawWaker, RawWakerVTable, Waker};
use std::time::{Duration, Instant};

use inferlet::model::{Model, Tokenizer};
use inferlet::sample::Sampler;
use inferlet::{chat, runtime, Constrain, Context, Result};
use serde::Deserialize;

// ============================================================================
// Input
// ============================================================================

#[derive(Deserialize)]
struct Input {
    prompt: String,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default = "default_system")]
    system: String,
    #[serde(default = "default_temperature")]
    temperature: f32,
    #[serde(default = "default_top_p")]
    top_p: f32,
    #[serde(default = "default_fake_wait_ms")]
    fake_wait_ms: u64,
    /// Recompute O(n²) scaling: ms ≈ alpha * n² / 1e6.
    #[serde(default = "default_alpha")]
    alpha_recompute_ns_per_tok2: f64,
    /// Swap O(n) scaling: ms ≈ beta * n / 1e6.
    #[serde(default = "default_beta")]
    beta_swap_ns_per_tok: f64,
    /// §5.3 mid-stream injection. Off by default — with weak few-shot
    /// adherence, models can loop forever (call → mid-stream intr → call → …).
    /// Trap-only injection bounds the loop to TRAP boundaries.
    #[serde(default)]
    enable_midstream: bool,
    /// Hard ceiling on decode steps regardless of generated.len(). Catches
    /// runaway loops where tokens are absorbed by the parser so `generated`
    /// never reaches max_tokens. Defaults to 4× max_tokens.
    #[serde(default)]
    max_steps: Option<u32>,
}

fn default_max_tokens() -> usize { 512 }
fn default_temperature() -> f32 { 0.6 }
fn default_top_p() -> f32 { 0.95 }
fn default_fake_wait_ms() -> u64 { 500 }
fn default_alpha() -> f64 { 800.0 }
fn default_beta() -> f64 { 400_000.0 }
fn default_system() -> String {
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
       is free latency-wise. Delay [TRAP][END] as long as you still have\n\
       result-independent things to say.\n\
     - After [INTR] frames appear you may EITHER (a) dispatch another\n\
       round of [CALL]s whose inputs depend on those results, then\n\
       [TRAP][END] again, OR (b) write the final natural-language answer\n\
       and stop. There is no fixed number of rounds.\n\
     - If part of the question is outside these tools, answer that part\n\
       from your own knowledge — do NOT invent a [CALL] for it.\n\
     - Do not put CML syntax inside <think> tags.\n\
     - Do not fabricate [INTR] frames yourself; the runtime produces them.\n\
     - Do NOT emit [TRAP][END] when you have no pending calls.\n\
     \n\
     Example — single round, two parallel calls:\n\
     User: What's the weather in NYC and London?\n\
     Assistant: [CALL] w1 [HEAD] get_weather(\"New York\") [END]\n\
     [CALL] w2 [HEAD] get_weather(\"London\") [END]\n\
     [TRAP][END]\n\
     [INTR] w1 [HEAD] {\"temp_f\": 72, \"sky\": \"sunny\"} [END]\n\
     [INTR] w2 [HEAD] {\"temp_f\": 60, \"sky\": \"cloudy\"} [END]\n\
     NYC is 72°F and sunny; London is 60°F and cloudy."
        .to_string()
}

// ============================================================================
// CML token registry
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
        let (special_ids, special_bytes) = tokenizer.special_tokens();
        let special_map: HashMap<Vec<u8>, u32> = special_bytes
            .into_iter()
            .zip(special_ids.into_iter())
            .collect();

        let resolve = |s: &str| -> Vec<u32> {
            if let Some(&id) = special_map.get(s.as_bytes()) {
                vec![id]
            } else {
                let ids = tokenizer.encode(s);
                assert!(!ids.is_empty(), "CML token '{}' tokenized to empty", s);
                ids
            }
        };

        let call_ids = resolve("[CALL]");
        let end_ids = resolve("[END]");
        let head_ids = resolve("[HEAD]");
        let trap_ids = resolve("[TRAP]");
        let intr_ids = resolve("[INTR]");

        let mut trap_end_ids = vec![trap_ids
            .iter()
            .chain(end_ids.iter())
            .cloned()
            .collect::<Vec<u32>>()];
        let merge = tokenizer.encode("][");
        if merge.len() == 1 {
            let mut merged: Vec<u32> = trap_ids[..trap_ids.len() - 1].to_vec();
            merged.push(merge[0]);
            merged.extend_from_slice(&end_ids[1..]);
            trap_end_ids.push(merged);
        }

        let shared: HashSet<u32> = call_ids
            .iter()
            .chain(end_ids.iter())
            .chain(head_ids.iter())
            .chain(trap_ids.iter())
            .copied()
            .collect();
        let suppressed: HashSet<u32> = intr_ids
            .iter()
            .copied()
            .filter(|id| !shared.contains(id))
            .collect();

        CmlRegistry {
            call_ids,
            end_ids,
            head_ids,
            trap_ids,
            intr_ids,
            trap_end_ids,
            suppressed,
        }
    }
}

// ============================================================================
// Constraint — block [INTR]-unique token ids (BRLE mask, static)
// ============================================================================

struct SuppressMask {
    mask: Vec<u32>,
}

impl SuppressMask {
    fn new(vocab_size: u32, suppressed: &HashSet<u32>) -> Self {
        let mut buf: Vec<u32> = Vec::new();
        let mut current_val = false;
        let mut current_count: u32 = 0;
        for i in 0..vocab_size {
            let allowed = !suppressed.contains(&i);
            if allowed == current_val {
                current_count += 1;
            } else {
                buf.push(current_count);
                current_val = !current_val;
                current_count = 1;
            }
        }
        buf.push(current_count);
        Self { mask: buf }
    }
}

impl Constrain for SuppressMask {
    fn step(&mut self, _accepted: &[u32]) -> &[u32] {
        &self.mask
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
    /// Parser entered a CML block (§5.3 critical-section: interrupts deferred).
    EnterCriticalSection,
    /// Parser left a CML block back to Normal (interrupts may inject again).
    ExitCriticalSection,
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
        }
    }

    /// True iff Normal state with no pending delimiter prefix — safe seam
    /// for mid-stream injection.
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

    fn detokenize_one(tokenizer: &Tokenizer, t: u32) -> String {
        tokenizer.decode(&[t]).unwrap_or_default()
    }

    fn strip_trailing_bracket(tokens: &mut Vec<u32>, tokenizer: &Tokenizer) {
        if let Some(&last) = tokens.last() {
            if Self::detokenize_one(tokenizer, last).ends_with('[') {
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
                    }
                    Delim::None => {
                        while !self.buf.is_empty() && !self.is_prefix_of_any(&self.buf) {
                            let t = self.buf.remove(0);
                            if Self::detokenize_one(tokenizer, t).ends_with('[') { continue; }
                            events.push(Event::Passthrough(t));
                        }
                    }
                    _ => {
                        for &t in &self.buf {
                            if Self::detokenize_one(tokenizer, t).ends_with('[') { continue; }
                            events.push(Event::Passthrough(t));
                        }
                        self.buf.clear();
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
                        let id = tokenizer.decode(&self.id_tokens).unwrap_or_default().trim().to_string();
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
                        let id = tokenizer.decode(&self.id_tokens).unwrap_or_default().trim().to_string();
                        let code = tokenizer.decode(&self.body_tokens).unwrap_or_default().trim().to_string();
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

        // §5.3 critical-section transitions: Enter on Normal→non-Normal,
        // Exit on non-Normal→Normal. Order matters — flush after the
        // Call/Trap events produced this step.
        if prev_state == State::Normal && self.state != State::Normal {
            events.push(Event::EnterCriticalSection);
        } else if prev_state != State::Normal && self.state == State::Normal {
            events.push(Event::ExitCriticalSection);
        }

        events
    }

    fn flush(&mut self, tokenizer: &Tokenizer) -> Vec<Event> {
        self.buf
            .drain(..)
            .filter(|&t| !Self::detokenize_one(tokenizer, t).ends_with('['))
            .map(Event::Passthrough)
            .collect()
    }
}

// ============================================================================
// Dummy executor
// ============================================================================

fn noop_waker() -> Waker {
    const VTABLE: RawWakerVTable = RawWakerVTable::new(
        |_| RawWaker::new(std::ptr::null(), &VTABLE),
        |_| {},
        |_| {},
        |_| {},
    );
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

    println!("[Async3] call id={} completed (wait {}ms)", id, wait_ms);
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
    wait_ms: u64,
    fut: Pin<Box<dyn Future<Output = String>>>,
}

impl Pending {
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
    queue: VecDeque<String>,
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

    fn drain(&mut self) -> Vec<String> {
        if self.in_critical_section { return Vec::new(); }
        self.queue.drain(..).collect()
    }

    fn pending_count(&self) -> usize { self.queue.len() }
}

// ============================================================================
// §5.4 Trap classifier — paper's 3-way Keep / Recompute / Swap decision
//
// Records the strategy for observability; Swap/Recompute execution requires
// SDK surface not currently exposed (committed-page truncation + in-place
// context replacement while a `Generator` holds the mutable borrow), so all
// branches collapse to Keep when applied.
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq)]
enum TrapStrategy { Keep, Recompute, Swap }

struct CostModel {
    alpha_ns_per_tok2: f64,
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

/// Paper rule: if recompute_t > wait_t AND swap_t > wait_t → Keep; else if
/// recompute_t ≤ swap_t → Recompute; else → Swap.
fn classify_trap(n: usize, wait_ms: f64, cost: &CostModel) -> (TrapStrategy, f64, f64) {
    let r_ms = cost.recompute_ms(n);
    let s_ms = cost.swap_ms(n);
    let strategy = if r_ms > wait_ms && s_ms > wait_ms {
        TrapStrategy::Keep
    } else if r_ms <= s_ms {
        TrapStrategy::Recompute
    } else {
        TrapStrategy::Swap
    };
    (strategy, r_ms, s_ms)
}

// ============================================================================
// Main
// ============================================================================

#[inferlet::main]
async fn main(input: Input) -> Result<String> {
    let max_steps = input.max_steps.unwrap_or_else(|| input.max_tokens.saturating_mul(4) as u32);

    let model_name = runtime::models()
        .first()
        .cloned()
        .ok_or("No models available")?;
    let model = Model::load(&model_name)?;
    let tokenizer = model.tokenizer();
    let registry = CmlRegistry::new(&tokenizer);
    let vocab_size = tokenizer.vocabs().0.len() as u32;

    println!(
        "[Async3] call={:?} head={:?} end={:?} trap={:?} intr={:?}",
        registry.call_ids,
        registry.head_ids,
        registry.end_ids,
        registry.trap_ids,
        registry.intr_ids
    );

    let cost = CostModel {
        alpha_ns_per_tok2: input.alpha_recompute_ns_per_tok2,
        beta_ns_per_tok: input.beta_swap_ns_per_tok,
    };

    let mut ctx = Context::new(&model)?;
    ctx.system(&input.system).user(&input.prompt).cue();

    let stops = chat::stop_tokens(&model);
    let suppress = SuppressMask::new(vocab_size, &registry.suppressed);

    let mut g = ctx
        .generate(Sampler::TopP {
            temperature: input.temperature,
            p: input.top_p,
        })
        .max_tokens(input.max_tokens)
        .stop(&stops)
        .constrain(suppress);

    let mut parser = Parser::new(&registry);
    let mut generated: Vec<u32> = Vec::new();
    let mut pending: Vec<Pending> = Vec::new();
    let mut interrupts = InterruptManager::new();
    let mut step_count: u32 = 0;

    while let Some(step) = g.next()? {
        let out = step.execute().await?;
        if out.tokens.is_empty() { continue; }
        step_count += out.tokens.len() as u32;

        // Advance each in-flight call by one poll — concurrent with token gen.
        pending.retain_mut(|p| match poll_once(p) {
            Some(result) => {
                println!(
                    "[Async3] id={} ready after {}ms",
                    p.id,
                    p.dispatched_at.elapsed().as_millis()
                );
                interrupts.enqueue(&p.id, &result);
                false
            }
            None => true,
        });

        for &tok in &out.tokens {
            for event in parser.feed(tok, &tokenizer) {
                match event {
                    Event::Passthrough(t) => generated.push(t),
                    Event::EnterCriticalSection => interrupts.set_critical(true),
                    Event::ExitCriticalSection => interrupts.set_critical(false),
                    Event::Call { id, code } => {
                        println!(
                            "[Async3] dispatching id={} code={} (fake_wait={}ms)",
                            id, code, input.fake_wait_ms
                        );
                        pending.push(Pending {
                            id: id.clone(),
                            dispatched_at: Instant::now(),
                            wait_ms: input.fake_wait_ms,
                            fut: Box::pin(execute_call(id, code, input.fake_wait_ms)),
                        });
                    }
                    Event::Trap => {
                        if pending.is_empty() && interrupts.pending_count() == 0 {
                            println!("[Async3] trap with nothing pending — no-op");
                            continue;
                        }

                        let wait_ms: f64 = pending
                            .iter()
                            .map(|p| p.remaining_ms() as f64)
                            .fold(0.0f64, f64::max);
                        let context_len = (g.tokens_generated() + generated.len()) as usize;
                        let (strategy, r_ms, s_ms) =
                            classify_trap(context_len, wait_ms, &cost);
                        println!(
                            "[Async3] trap decision: {:?}  wait_t={:.1}ms recompute_t={:.1}ms swap_t={:.1}ms n={}",
                            strategy, wait_ms, r_ms, s_ms, context_len
                        );

                        // Drain all pending futures (paper: trap blocks).
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
                        println!(
                            "[Async3] trap drained in {}ms",
                            trap_start.elapsed().as_millis()
                        );

                        // Execute Keep regardless of classification — see
                        // module docs. Critical section ends at [TRAP][END].
                        interrupts.set_critical(false);
                        let mut injected: Vec<u32> = Vec::new();
                        for frame in interrupts.drain() {
                            println!("[Async3] keep-inject: {}", frame.trim());
                            injected.extend(tokenizer.encode(&frame));
                        }
                        if !injected.is_empty() {
                            let accepted = g.accept(&injected);
                            for &t in &accepted {
                                for ev in parser.feed(t, &tokenizer) {
                                    if let Event::Passthrough(p) = ev {
                                        generated.push(p);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // §5.3 mid-stream injection: only when not in a critical section AND
        // the parser is in clean Normal state (buf empty).
        if input.enable_midstream && !interrupts.in_critical_section && parser.is_clean() {
            let mut injected: Vec<u32> = Vec::new();
            for frame in interrupts.drain() {
                println!("[Async3] mid-stream inject: {}", frame.trim());
                injected.extend(tokenizer.encode(&frame));
            }
            if !injected.is_empty() {
                let accepted = g.accept(&injected);
                for &t in &accepted {
                    for ev in parser.feed(t, &tokenizer) {
                        if let Event::Passthrough(p) = ev {
                            generated.push(p);
                        }
                    }
                }
            }
        }

        if step_count >= max_steps {
            println!(
                "[Async3] hit max_steps={} (generated={}); breaking",
                max_steps, generated.len()
            );
            break;
        }
    }

    for ev in parser.flush(&tokenizer) {
        if let Event::Passthrough(t) = ev { generated.push(t); }
    }

    Ok(tokenizer.decode(&generated).unwrap_or_default())
}
