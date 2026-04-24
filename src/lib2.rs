//! Minimal AsyncLM inferlet.
//!
//! Demonstrates the core async function-calling loop:
//!   • Parse `[CALL] id [HEAD] code [END]` blocks emitted by the model
//!   • Parse `[TRAP][END]` — wait for pending calls and inject `[INTR]` results
//!   • Suppress the `[INTR]` entry token so the model never fabricates one itself
//!
//! Uses a dummy synchronous executor that returns a canned string per call —
//! no KV-page checkpoint / shrink / swap logic, no real broadcast/subscribe.
//!
//! To build this file instead of `lib.rs`, set `path = "src/lib2.rs"` in the
//! `[lib]` section of Cargo.toml.

use inferlet::sampler::Sample;
use inferlet::stop_condition::{ends_with_any, max_len, StopCondition};
use inferlet::{Args, Result, Sampler, Tokenizer};
use std::collections::{HashMap, HashSet};
use std::future::{poll_fn, Future};
use std::pin::Pin;
use std::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};
use std::time::{Duration, Instant};

// ============================================================================
// CML token registry
// ============================================================================

struct CmlRegistry {
    call_ids: Vec<u32>,
    end_ids: Vec<u32>,
    head_ids: Vec<u32>,
    trap_ids: Vec<u32>,
    intr_ids: Vec<u32>,

    /// Full `[TRAP][END]` sequences; second variant handles BPE-merged `][` (e.g. 1404).
    trap_end_ids: Vec<Vec<u32>>,

    /// Token IDs unique to `[INTR]` — suppressed during sampling so the model
    /// cannot hallucinate an interrupt frame.  Bracket tokens shared with other
    /// delimiters are NOT suppressed.
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
        let end_ids  = resolve("[END]");
        let head_ids = resolve("[HEAD]");
        let trap_ids = resolve("[TRAP]");
        let intr_ids = resolve("[INTR]");

        // Combined [TRAP][END] — and BPE-merged `][` variant if present.
        let mut trap_end_ids = vec![trap_ids.iter().chain(end_ids.iter()).cloned().collect()];
        let merge = tokenizer.tokenize("][");
        if merge.len() == 1 {
            let mut merged: Vec<u32> = trap_ids[..trap_ids.len() - 1].to_vec();
            merged.push(merge[0]);
            merged.extend_from_slice(&end_ids[1..]);
            trap_end_ids.push(merged);
        }

        // Suppress only tokens *unique* to [INTR]. Shared `[` and `]` must remain sampleable.
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
// CML parser (FSM)
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq)]
enum State { Normal, InCallId, InCallBody, InTrap }

#[derive(Debug)]
enum Event {
    Passthrough(u32),
    Call { id: String, code: String },
    Trap,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Delim { Call, End, Head, Trap, TrapEnd, None }

struct Parser {
    // Full sequences (Normal state matching)
    call_ids: Vec<u32>,
    end_ids: Vec<u32>,
    head_ids: Vec<u32>,
    trap_ids: Vec<u32>,
    trap_end_ids: Vec<Vec<u32>>,
    // Bracket-free inner suffixes — used inside CML blocks, since the `\n+[`
    // BPE merge makes the leading bracket unpredictable once we're already in
    // a critical section.  Also consulted in Normal state so we catch [CALL] /
    // [TRAP] / [TRAP][END] when the leading `[` has been absorbed into the
    // previous token (e.g. ` [` or `\n[` merged into one BPE piece).
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
            end_inner:  reg.end_ids[1..].to_vec(),
            head_inner: reg.head_ids[1..].to_vec(),
            trap_inner: reg.trap_ids[1..].to_vec(),
            trap_end_inner: reg.trap_end_ids.iter().map(|s| s[1..].to_vec()).collect(),
            call_ids: reg.call_ids.clone(),
            end_ids:  reg.end_ids.clone(),
            head_ids: reg.head_ids.clone(),
            trap_ids: reg.trap_ids.clone(),
            trap_end_ids: reg.trap_end_ids.clone(),
            state: State::Normal,
            id_tokens: Vec::new(),
            body_tokens: Vec::new(),
            buf: Vec::new(),
        }
    }

    fn ends_with(buf: &[u32], pat: &[u32]) -> bool {
        !pat.is_empty() && buf.len() >= pat.len() && buf[buf.len() - pat.len()..] == *pat
    }

    fn is_prefix(buf: &[u32], pat: &[u32]) -> bool {
        !pat.is_empty() && buf.len() <= pat.len() && pat[..buf.len()] == *buf
    }

    fn match_full(&self, buf: &[u32]) -> (Delim, usize) {
        if Self::ends_with(buf, &self.call_ids) { return (Delim::Call, self.call_ids.len()); }
        if Self::ends_with(buf, &self.end_ids)  { return (Delim::End,  self.end_ids.len());  }
        for s in &self.trap_end_ids {
            if Self::ends_with(buf, s) { return (Delim::TrapEnd, s.len()); }
        }
        if Self::ends_with(buf, &self.trap_ids) { return (Delim::Trap, self.trap_ids.len()); }
        if Self::ends_with(buf, &self.head_ids) { return (Delim::Head, self.head_ids.len()); }
        (Delim::None, 0)
    }

    fn match_inner(&self, buf: &[u32]) -> (Delim, usize) {
        if Self::ends_with(buf, &self.call_inner) { return (Delim::Call, self.call_inner.len()); }
        if Self::ends_with(buf, &self.end_inner)  { return (Delim::End,  self.end_inner.len());  }
        if Self::ends_with(buf, &self.trap_inner) { return (Delim::Trap, self.trap_inner.len()); }
        if Self::ends_with(buf, &self.head_inner) { return (Delim::Head, self.head_inner.len()); }
        (Delim::None, 0)
    }

    /// Like `match_inner` but restricted to the delimiters that can appear at
    /// the top level (Normal state): [CALL], [TRAP], [TRAP][END].  End and
    /// Head alone have no meaning in Normal state.  Checked after `match_full`
    /// fails to catch the case where the leading `[` of the delimiter has been
    /// BPE-merged into the previous token.
    fn match_normal_inner(&self, buf: &[u32]) -> (Delim, usize) {
        // TrapEnd first — longest match wins.
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

        match self.state {
            State::Normal => {
                self.buf.push(token_id);
                // Try full match (leading `[` present as token 58) first; fall
                // back to bracket-free inner match to catch BPE-merged leading
                // brackets like ` [` or `\n[`.
                let (d, dlen) = {
                    let (d, l) = self.match_full(&self.buf);
                    if d != Delim::None { (d, l) } else { self.match_normal_inner(&self.buf) }
                };
                match d {
                    Delim::Call => {
                        let end = self.buf.len() - dlen;
                        let mut prefix: Vec<u32> = self.buf[..end].to_vec();
                        // If we matched inner, the last prefix token is the
                        // bracket-absorbed BPE piece — strip it.
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
                            // A token that decodes ending with `[` was buffered
                            // as a possible CML delimiter start but no delimiter
                            // formed — it's an aborted attempt (typically from
                            // [INTR] suppression).  Drop it instead of leaking
                            // a stray `[` into the passthrough stream.  Real
                            // delimiters are consumed by matching, not by this
                            // drain path, so this only catches leakage.
                            if tokenizer.detokenize(&[t]).ends_with('[') {
                                continue;
                            }
                            events.push(Event::Passthrough(t));
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
// Sampler — top-p with INTR suppression
// ============================================================================

struct SuppressingSampler {
    suppressed: HashSet<u32>,
    top_p: f32,
}

impl Sample for SuppressingSampler {
    fn sample(&self, ids: &[u32], probs: &[f32]) -> u32 {
        // Build (id, prob) pairs, zeroing suppressed tokens.
        let mut pairs: Vec<(u32, f32)> = ids.iter().zip(probs.iter())
            .map(|(&id, &p)| if self.suppressed.contains(&id) { (id, 0.0) } else { (id, p) })
            .collect();
        pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Top-p nucleus — pick argmax within the cumulative-prob cutoff.
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
// Async dummy executor
//
// `execute_call` is an `async fn` that yields `Poll::Pending` until a wall-clock
// deadline passes, then returns a canned result.  We manually poll these futures
// once per main-loop iteration with a no-op waker, so calls make progress
// between `decode_step` awaits — i.e. while the model is generating tokens.
// This is the key async-LM property: dispatched tool calls run concurrently with
// token generation and only block at `[TRAP][END]`.
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
            // Ask to be re-polled — our top-level loop polls us each iteration anyway,
            // but this keeps things correct under any scheduler.
            cx.waker().wake_by_ref();
            Poll::Pending
        }
    })
    .await;

    inferlet::send(&format!("[MinAsync] call id={} completed (wait {}ms)", id, wait_ms));
    let lc = code.to_lowercase();

    // get_weather(city) — standalone or chainable (temp becomes input to another tool)
    if lc.contains("get_weather") || lc.contains("weather(") {
        if lc.contains("london") {
            "{\"temp_f\": 60, \"sky\": \"cloudy\"}".to_string()
        } else if lc.contains("paris") {
            "{\"temp_f\": 59, \"sky\": \"rainy\"}".to_string()
        } else if lc.contains("boston") {
            "{\"temp_f\": 68, \"sky\": \"clear\"}".to_string()
        } else if lc.contains("tokyo") {
            "{\"temp_f\": 75, \"sky\": \"humid\"}".to_string()
        } else {
            "{\"temp_f\": 72, \"sky\": \"sunny\"}".to_string()
        }
    }
    // get_stock_price(ticker) — standalone; chainable via convert_currency
    else if lc.contains("stock_price") || lc.contains("get_stock") {
        if lc.contains("aapl") {
            "{\"ticker\": \"AAPL\", \"price_usd\": 189.50}".to_string()
        } else if lc.contains("goog") {
            "{\"ticker\": \"GOOG\", \"price_usd\": 141.20}".to_string()
        } else if lc.contains("tsla") {
            "{\"ticker\": \"TSLA\", \"price_usd\": 258.30}".to_string()
        } else {
            "{\"ticker\": \"UNKNOWN\", \"price_usd\": 100.00}".to_string()
        }
    }
    // convert_currency(amount, from, to) — chains off stock / weather-temp
    else if lc.contains("convert_currency") || lc.contains("exchange_rate") {
        if lc.contains("eur") {
            "{\"rate\": 0.92, \"quote\": \"USD->EUR\"}".to_string()
        } else if lc.contains("jpy") {
            "{\"rate\": 155.40, \"quote\": \"USD->JPY\"}".to_string()
        } else if lc.contains("gbp") {
            "{\"rate\": 0.79, \"quote\": \"USD->GBP\"}".to_string()
        } else {
            "{\"rate\": 1.00, \"quote\": \"USD->USD\"}".to_string()
        }
    }
    // search_restaurants(city) — standalone; chainable via get_reviews
    else if lc.contains("search_restaurants") || lc.contains("find_restaurants") {
        if lc.contains("tokyo") {
            "[\"Sukiyabashi Jiro\", \"Ichiran\", \"Sushi Saito\"]".to_string()
        } else if lc.contains("paris") {
            "[\"Le Jules Verne\", \"L'Ami Jean\", \"Septime\"]".to_string()
        } else {
            "[\"Katz's Deli\", \"Joe's Pizza\", \"Lombardi's\"]".to_string()
        }
    }
    // get_reviews(name) — chains off search_restaurants
    else if lc.contains("get_reviews") || lc.contains("reviews(") {
        if lc.contains("katz") {
            "{\"stars\": 4.6, \"summary\": \"Iconic pastrami sandwiches\"}".to_string()
        } else if lc.contains("jiro") {
            "{\"stars\": 4.9, \"summary\": \"Legendary omakase, tiny counter\"}".to_string()
        } else if lc.contains("joe") {
            "{\"stars\": 4.4, \"summary\": \"Classic NY slice, cheap and fast\"}".to_string()
        } else {
            "{\"stars\": 4.0, \"summary\": \"Solid, well-reviewed spot\"}".to_string()
        }
    }
    // get_time(timezone) — standalone
    else if lc.contains("get_time") || lc.contains("time_in") || lc.contains("current_time") {
        if lc.contains("tokyo") || lc.contains("jst") {
            "{\"time\": \"2026-04-21T23:32+09:00\", \"tz\": \"JST\"}".to_string()
        } else if lc.contains("london") || lc.contains("gmt") {
            "{\"time\": \"2026-04-21T15:32+01:00\", \"tz\": \"BST\"}".to_string()
        } else if lc.contains("new_york") || lc.contains("nyc") || lc.contains("est") {
            "{\"time\": \"2026-04-21T10:32-04:00\", \"tz\": \"EDT\"}".to_string()
        } else {
            "{\"time\": \"2026-04-21T14:32+00:00\", \"tz\": \"UTC\"}".to_string()
        }
    } else {
        format!("result({})", code)
    }
}

/// A call that has been dispatched but not yet observed as complete.
struct Pending {
    id: String,
    dispatched_at: Instant,
    fut: Pin<Box<dyn Future<Output = String>>>,
}

/// Poll a pending future once.  Returns the result if ready.
fn poll_once(p: &mut Pending) -> Option<String> {
    let waker = noop_waker();
    let mut cx = Context::from_waker(&waker);
    match p.fut.as_mut().poll(&mut cx) {
        Poll::Ready(r) => Some(r),
        Poll::Pending => None,
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
               This is the whole point of async: the calls run in parallel\n\
               with your writing, so prose before [TRAP] is free latency-wise.\n\
               Delay [TRAP][END] as long as you still have result-independent\n\
               things to say.\n\
             - After [INTR] frames appear you may EITHER (a) dispatch another\n\
               round of [CALL]s whose inputs depend on those results, then\n\
               [TRAP][END] again, OR (b) write the final natural-language answer\n\
               and stop. There is no fixed number of rounds.\n\
             - If part of the question is outside these tools (general knowledge,\n\
               math, unrelated topics), answer that part from your own knowledge\n\
               — do NOT invent a [CALL] for it.\n\
             - Do not put CML syntax inside <think> tags.\n\
             - Do not fabricate [INTR] frames yourself; the runtime produces them.\n\
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
             Example 2 — two rounds, round 2 depends on round 1 results:\n\
             User: Get the weather in Boston, then also get the weather for a\n\
             city whose name is that temperature (as a string).\n\
             Assistant: [CALL] b1 [HEAD] get_weather(\"Boston\") [END]\n\
             [TRAP][END]\n\
             [INTR] b1 [HEAD] {\"temp_f\": 68, \"sky\": \"clear\"} [END]\n\
             Boston is 68°F. Now looking up \"68\".\n\
             [CALL] b2 [HEAD] get_weather(\"68\") [END]\n\
             [TRAP][END]\n\
             [INTR] b2 [HEAD] {\"temp_f\": 72, \"sky\": \"sunny\"} [END]\n\
             Boston is 68°F and clear; \"68\" is 72°F and sunny.\n\
             \n\
             Example 3 — mixed tool + knowledge question:\n\
             User: What's the weather in Paris and what is the capital of Japan?\n\
             Assistant: [CALL] p1 [HEAD] get_weather(\"Paris\") [END]\n\
             [TRAP][END]\n\
             [INTR] p1 [HEAD] {\"temp_f\": 59, \"sky\": \"rainy\"} [END]\n\
             Paris is 59°F and rainy. The capital of Japan is Tokyo.\n\
             \n\
             Example 4 — interleaved prose hides call latency (preferred style\n\
             whenever you have anything to say that does not depend on the\n\
             results). Note how reasoning appears BEFORE [TRAP][END]:\n\
             User: Get the weather in Paris and the time in Tokyo, then tell\n\
             me if it's a reasonable hour to video-call Tokyo from Paris.\n\
             Assistant: [CALL] w1 [HEAD] get_weather(\"Paris\") [END]\n\
             [CALL] t1 [HEAD] get_time(\"Asia/Tokyo\") [END]\n\
             Both calls are dispatched in parallel. While they run, here is the\n\
             shape of the answer: Paris weather tells us whether the caller is\n\
             likely indoors and free, and Tokyo local time tells us whether the\n\
             other side is awake — Tokyo is roughly 7-8 hours ahead of Paris,\n\
             so a Paris afternoon maps to a Tokyo late-night which is bad, but\n\
             a Paris morning maps to a Tokyo afternoon which is ideal.\n\
             [TRAP][END]\n\
             [INTR] w1 [HEAD] {\"temp_f\": 62, \"sky\": \"overcast\"} [END]\n\
             [INTR] t1 [HEAD] {\"time\": \"07:30\", \"tz\": \"Asia/Tokyo\"} [END]\n\
             Paris is 62°F and overcast — comfortable indoor conditions. Tokyo\n\
             is 07:30, just starting the workday, so this is a good window."
                .to_string()
        });
    let temperature: f32 = args.value_from_str(["-t", "--temperature"]).unwrap_or(0.6);
    let top_p: f32 = args.value_from_str("--top-p").unwrap_or(0.95);
    let fake_wait_ms: u64 = args.value_from_str("--fake-wait-ms").unwrap_or(500);

    let model = inferlet::get_auto_model();
    let tokenizer = model.get_tokenizer();
    let registry = CmlRegistry::new(&tokenizer);

    inferlet::send(&format!(
        "[MinAsync] call={:?} head={:?} end={:?} trap={:?} intr={:?}",
        registry.call_ids, registry.head_ids, registry.end_ids, registry.trap_ids, registry.intr_ids
    ));

    let sampler = Sampler::Custom {
        temperature,
        sampler: Box::new(SuppressingSampler {
            suppressed: registry.suppressed.clone(),
            top_p,
        }),
    };

    let mut ctx = model.create_context();
    ctx.fill_system(&system);
    ctx.fill_user(&prompt);

    let eos_tokens = model.eos_tokens();
    let stop = max_len(max_tokens).or(ends_with_any(eos_tokens.clone()));

    let mut parser = Parser::new(&registry);
    let mut generated: Vec<u32> = Vec::new();

    // In-flight async calls, and results that have completed but not yet injected.
    let mut pending: Vec<Pending> = Vec::new();
    let mut ready_frames: Vec<String> = Vec::new();

    loop {
        let token_id = ctx.decode_step(&sampler).await;
        ctx.fill_token(token_id);

        // Advance each in-flight call by one poll — concurrent with token generation.
        // This is the core async-LM property: tool calls progress while tokens are
        // being sampled, not blocking serial execution.
        pending.retain_mut(|p| match poll_once(p) {
            Some(result) => {
                inferlet::send(&format!(
                    "[MinAsync] id={} ready after {}ms",
                    p.id,
                    p.dispatched_at.elapsed().as_millis()
                ));
                ready_frames.push(format!(" [INTR] {} [HEAD] {} [END] ", p.id, result));
                false
            }
            None => true,
        });

        for event in parser.feed(token_id, &tokenizer) {
            match event {
                Event::Passthrough(t) => generated.push(t),
                Event::Call { id, code } => {
                    inferlet::send(&format!(
                        "[MinAsync] dispatching id={} code={} (fake_wait={}ms)",
                        id, code, fake_wait_ms
                    ));
                    pending.push(Pending {
                        id: id.clone(),
                        dispatched_at: Instant::now(),
                        fut: Box::pin(execute_call(id, code, fake_wait_ms)),
                    });
                }
                Event::Trap => {
                    // TRAP blocks: force any still-pending futures to completion.
                    let trap_start = Instant::now();
                    while !pending.is_empty() {
                        pending.retain_mut(|p| match poll_once(p) {
                            Some(result) => {
                                ready_frames.push(format!(
                                    " [INTR] {} [HEAD] {} [END] ",
                                    p.id, result
                                ));
                                false
                            }
                            None => true,
                        });
                    }
                    inferlet::send(&format!(
                        "[MinAsync] trap drained in {}ms; injecting {} frames",
                        trap_start.elapsed().as_millis(),
                        ready_frames.len()
                    ));
                    for frame in ready_frames.drain(..) {
                        inferlet::send(&format!("[MinAsync] injecting: {}", frame.trim()));
                        ctx.fill(&frame);
                    }
                }
            }
        }

        if stop.check(&generated) { break; }
    }

    for ev in parser.flush(&tokenizer) {
        if let Event::Passthrough(t) = ev { generated.push(t); }
    }

    Ok(tokenizer.detokenize(&generated))
}
