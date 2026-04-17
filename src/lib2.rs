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
    // a critical section.
    call_inner: Vec<u32>,
    end_inner: Vec<u32>,
    head_inner: Vec<u32>,
    trap_inner: Vec<u32>,

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
                let (d, dlen) = self.match_full(&self.buf);
                match d {
                    Delim::Call => {
                        let end = self.buf.len() - dlen;
                        for i in 0..end { events.push(Event::Passthrough(self.buf[i])); }
                        self.buf.clear();
                        self.id_tokens.clear();
                        self.body_tokens.clear();
                        self.state = State::InCallId;
                    }
                    Delim::Trap => {
                        let end = self.buf.len() - dlen;
                        for i in 0..end { events.push(Event::Passthrough(self.buf[i])); }
                        self.buf.clear();
                        self.state = State::InTrap;
                    }
                    Delim::TrapEnd => {
                        let end = self.buf.len() - dlen;
                        for i in 0..end { events.push(Event::Passthrough(self.buf[i])); }
                        self.buf.clear();
                        events.push(Event::Trap);
                    }
                    Delim::None => {
                        while !self.buf.is_empty() && !self.is_prefix_of_any(&self.buf) {
                            let t = self.buf.remove(0);
                            events.push(Event::Passthrough(t));
                        }
                    }
                    _ => {
                        for &t in &self.buf { events.push(Event::Passthrough(t)); }
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

    fn flush(&mut self) -> Vec<Event> {
        let out: Vec<Event> = self.buf.drain(..).map(Event::Passthrough).collect();
        out
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
// Dummy executor — canned results, immediate.
// ============================================================================

fn execute_call(id: &str, code: &str) -> String {
    inferlet::send(&format!("[MinAsync] dispatching call id={} code={}", id, code));
    // Canned response: echo the code with a stub answer.
    if code.to_lowercase().contains("weather") {
        format!("{{\"temp_f\": 72, \"sky\": \"sunny\"}}")
    } else {
        format!("result({})", code)
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
            "You are an assistant that MUST use async function calls to answer.\n\
             \n\
             Syntax (use these delimiters EXACTLY):\n\
             - Dispatch a call:  [CALL] <id> [HEAD] <code> [END]\n\
             - Wait for results: [TRAP][END]\n\
             \n\
             You have one available function: get_weather(city).\n\
             \n\
             WORKFLOW — you MUST follow this exact sequence for every request:\n\
             1. Acknowledge the request briefly.\n\
             2. Emit one or more [CALL] blocks to dispatch the needed tools.\n\
             3. Emit [TRAP][END] to wait for the results. DO NOT write a final\n\
                answer before the [TRAP][END]. You will not know the result until\n\
                after the trap, so any answer before it is guessing.\n\
             4. After [TRAP][END], the results appear as [INTR] <id> [HEAD] <result> [END].\n\
                Read them and then write a concise natural-language answer.\n\
             \n\
             Example:\n\
             User: What's the weather in NYC and London?\n\
             Assistant: Let me check both cities.\n\
             [CALL] w1 [HEAD] get_weather(\"New York\") [END]\n\
             [CALL] w2 [HEAD] get_weather(\"London\") [END]\n\
             [TRAP][END]\n\
             (results arrive here as [INTR] frames)\n\
             NYC is 72°F and sunny; London is 60°F and cloudy.\n\
             \n\
             Do NOT put the [CALL] / [TRAP] syntax inside <think> tags. Emit them\n\
             as your actual response. Never answer without issuing a [TRAP][END]\n\
             first when a tool call is needed."
                .to_string()
        });
    let temperature: f32 = args.value_from_str(["-t", "--temperature"]).unwrap_or(0.6);
    let top_p: f32 = args.value_from_str("--top-p").unwrap_or(0.95);

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
    let mut pending_results: Vec<String> = Vec::new();

    loop {
        let token_id = ctx.decode_step(&sampler).await;
        ctx.fill_token(token_id);

        for event in parser.feed(token_id, &tokenizer) {
            match event {
                Event::Passthrough(t) => generated.push(t),
                Event::Call { id, code } => {
                    // Synchronous dummy executor — result ready immediately.
                    let result = execute_call(&id, &code);
                    pending_results.push(format!(" [INTR] {} [HEAD] {} [END] ", id, result));
                }
                Event::Trap => {
                    // Inject any pending results as [INTR] frames.
                    if pending_results.is_empty() {
                        inferlet::send("[MinAsync] trap with no pending results, continuing");
                    }
                    for frame in pending_results.drain(..) {
                        inferlet::send(&format!("[MinAsync] injecting: {}", frame.trim()));
                        ctx.fill(&frame);
                    }
                }
            }
        }

        if stop.check(&generated) { break; }
    }

    for ev in parser.flush() {
        if let Event::Passthrough(t) = ev { generated.push(t); }
    }

    Ok(tokenizer.detokenize(&generated))
}
