//! Minimal AsyncLM inferlet — faithful port of the reference `lib2.rs`
//! (https://github.com/evanc7007/asyncLM/blob/master/src/lib2.rs) to the
//! current Pie SDK.
//!
//! Core loop, unchanged from the reference:
//!   * Parse `[CALL] id [HEAD] code [END]` blocks the model emits.
//!   * Parse `[TRAP][END]` — block until all in-flight calls return.
//!   * Inject `[INTR]` result frames into the KV cache.
//!   * Suppress `[INTR]`-unique token ids so the model can't fabricate one.
//!
//! Async dummy executor: each call yields `Poll::Pending` until a wall-clock
//! deadline, then returns a canned string. Hand-polled with a no-op waker
//! once per decode step — calls progress concurrently with token generation.
//!
//! SDK-port deltas vs. the reference (forced by the new API surface):
//!   * `Args`/pico_args → serde `Input` struct (the macro now deserializes JSON).
//!   * `Sampler::Custom { Sample }` is gone. INTR suppression goes through a
//!     `Constrain` BRLE mask paired with `Sampler::TopP`. This loses the
//!     reference's argmax-within-nucleus determinism, but functionally
//!     reproduces "top-p with these ids zeroed out".
//!   * `ctx.decode_step` + `ctx.fill_token` → `ctx.generate(...)` →
//!     `g.next()` / `step.execute().await`. The SDK seeds the next step's
//!     input from the last sampled token automatically.
//!   * `ctx.fill(&frame)` (string injection) → `g.accept(&encoded)`. Same
//!     semantics for KV injection; `accept` also bumps `tokens_generated`
//!     against `max_tokens`, which slightly overcounts (acceptable for our
//!     ~25-token INTR frames against typical 2k budgets).
//!   * `model.eos_tokens()` → `chat::stop_tokens(&model)`.
//!   * `tokenizer.tokenize/detokenize` → `tokenizer.encode/decode`.
//!   * `inferlet::send` is gone; logging goes through `println!`.

use std::collections::{HashMap, HashSet};
use std::future::{poll_fn, Future};
use std::pin::Pin;
use std::task::{Context as TaskContext, Poll, RawWaker, RawWakerVTable, Waker};
use std::time::{Duration, Instant};

use inferlet::model::{Model, Tokenizer};
use inferlet::sample::Sampler;
use inferlet::{chat, runtime, Constrain, Context, Result};
use serde::Deserialize;

const DEBUG_TRACE: bool = false;

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
}

fn default_max_tokens() -> usize { 2048 }
fn default_temperature() -> f32 { 0.6 }
fn default_top_p() -> f32 { 0.95 }
fn default_fake_wait_ms() -> u64 { 500 }
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
     - Do not put CML syntax inside <think> tags. Reason out what to call\n\
       briefly, exit </think>, THEN emit the actual [CALL] / [TRAP][END]\n\
       frames. Describing a call inside <think> is not the same as making\n\
       the call — you must always emit the literal CML frames outside.\n\
     - Do not fabricate [INTR] frames yourself; the runtime produces them.\n\
     - Never put plain prose or explanations inside [CALL]; [CALL] bodies must be one of the listed tool functions.\n\
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

    /// Full `[TRAP][END]` sequences; second variant handles BPE-merged `][`.
    trap_end_ids: Vec<Vec<u32>>,

    /// Token IDs unique to `[INTR]` — masked out during sampling so the
    /// model cannot hallucinate an interrupt frame. Bracket tokens shared
    /// with other delimiters are NOT suppressed.
    suppressed: HashSet<u32>,

    /// Tokens that open / close a Qwen3 thinking block. Empty if absent.
    /// The parser uses these to suspend CML matching while the model is
    /// inside `<think>…</think>`, otherwise it eats literal CML delimiters
    /// that the model quotes in its reasoning.
    think_open_ids: Vec<u32>,
    think_close_ids: Vec<u32>,

    /// Every token whose decoded bytes start with `]`. Used by the suffix
    /// matcher as alternates for the canonical `]` in the final position
    /// of each delimiter pattern, so that BPE-fused tokens like `]\n` or
    /// `][` close out a delimiter correctly.
    close_bracket_alts: Vec<u32>,
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

        // Combined [TRAP][END] — plus BPE-merged `][` variant if present.
        let mut trap_end_ids = vec![trap_ids
            .iter()
            .chain(end_ids.iter())
            .cloned()
            .collect::<Vec<u32>>()];
        let merge = tokenizer.encode("][");
        if merge.len() == 1 && !trap_ids.is_empty() && !end_ids.is_empty() {
            let mut merged: Vec<u32> = trap_ids[..trap_ids.len() - 1].to_vec();
            merged.push(merge[0]);
            merged.extend_from_slice(&end_ids[1..]);
            trap_end_ids.push(merged);
        }

        // BPE-merged closing-bracket variants. The matchers expect every
        // delimiter to end at the canonical `]` token, but the tokenizer
        // routinely fuses `]` with the immediately-following byte into a
        // single token (`]\n`, `][`, `] `, etc.). Without these alternates
        // the suffix match silently fails on the final position and the
        // FSM stays stuck in InCallBody / InCallId / InTrap forever.
        let (vocab_ids, vocab_bytes) = tokenizer.vocabs();
        let mut close_bracket_alts: Vec<u32> = Vec::new();
        for (&id, bytes) in vocab_ids.iter().zip(vocab_bytes.iter()) {
            if !bytes.is_empty() && bytes[0] == b']' {
                close_bracket_alts.push(id);
            }
        }

        // Suppress only tokens *unique* to [INTR]. Shared `[` / `]` must
        // stay sampleable — they're needed for [CALL] / [TRAP] / [END].
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

        // Qwen3 think delimiters. Resolved via the same special-map →
        // encode fallback as the CML delimiters. Empty Vec on tokenizers
        // that lack the tag (Qwen2.5 etc.) — the bypass becomes a no-op.
        let resolve_optional = |s: &str| -> Vec<u32> {
            if let Some(&id) = special_map.get(s.as_bytes()) {
                vec![id]
            } else {
                let ids = tokenizer.encode(s);
                if ids.is_empty() {
                    Vec::new()
                } else {
                    ids
                }
            }
        };
        let think_open_ids = resolve_optional("<think>");
        let think_close_ids = resolve_optional("</think>");

        CmlRegistry {
            call_ids,
            end_ids,
            head_ids,
            trap_ids,
            intr_ids,
            trap_end_ids,
            suppressed,
            think_open_ids,
            think_close_ids,
            close_bracket_alts,
        }
    }
}

// ============================================================================
// Constraint — block the [INTR]-unique token ids
//
// The new SDK has no custom-sampler hook. INTR suppression goes through the
// constraint mask path: emit a BRLE that allows every vocab id except those
// in `suppressed`. The ban list is static, so we compute the mask once and
// hand back the same slice every step.
// ============================================================================

struct SuppressMask {
    mask: Vec<u32>,
}

impl SuppressMask {
    fn new(mask_size: u32, suppressed: &HashSet<u32>) -> Self {
        // BRLE alternates false/true runs starting with `false`.
        let mut buf: Vec<u32> = Vec::new();
        let mut current_val = false;
        let mut current_count: u32 = 0;
        for i in 0..mask_size {
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
// CML parser (FSM)
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq)]
enum State {
    Normal,
    InCallId,
    InCallBody,
    InTrap,
    InIntrId,
    InIntrBody,
}

#[derive(Debug)]
enum Event {
    Passthrough(u32),
    Call { id: String, code: String },
    Trap,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Delim {
    Call,
    End,
    Head,
    Trap,
    TrapEnd,
    Intr,
    None,
}

struct Parser {
    // Full sequences (Normal-state matching).
    call_ids: Vec<u32>,
    end_ids: Vec<u32>,
    head_ids: Vec<u32>,
    trap_ids: Vec<u32>,
    intr_ids: Vec<u32>,
    trap_end_ids: Vec<Vec<u32>>,
    // Bracket-free inner suffixes — same delimiters with the leading bracket
    // token dropped. Used to catch BPE-merged leading brackets (`\n[`, ` [`)
    // where the `[` is fused into the previous token.
    call_inner: Vec<u32>,
    end_inner: Vec<u32>,
    head_inner: Vec<u32>,
    trap_inner: Vec<u32>,
    intr_inner: Vec<u32>,
    trap_end_inner: Vec<Vec<u32>>,

    // Qwen3 think-block bypass. While in_think is true, all incoming tokens
    // are passed through unchanged and CML matching is suspended — the model
    // routinely quotes literal `[CALL]` / `[TRAP][END]` inside <think>, and
    // letting the FSM act on those corrupts the run.
    think_open_ids: Vec<u32>,
    think_close_ids: Vec<u32>,
    in_think: bool,
    think_buf: Vec<u32>,

    /// Tokens that decode to a string starting with `]`. Treated as
    /// equivalent to the canonical `]` token when matching the final
    /// position of a delimiter, so e.g. `]\n` (a single fused token)
    /// closes `[END]` correctly.
    close_bracket_alts: Vec<u32>,

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
            intr_inner: reg.intr_ids[1..].to_vec(),
            trap_end_inner: reg.trap_end_ids.iter().map(|s| s[1..].to_vec()).collect(),
            call_ids: reg.call_ids.clone(),
            end_ids: reg.end_ids.clone(),
            head_ids: reg.head_ids.clone(),
            trap_ids: reg.trap_ids.clone(),
            intr_ids: reg.intr_ids.clone(),
            trap_end_ids: reg.trap_end_ids.clone(),
            think_open_ids: reg.think_open_ids.clone(),
            think_close_ids: reg.think_close_ids.clone(),
            in_think: false,
            think_buf: Vec::new(),
            close_bracket_alts: reg.close_bracket_alts.clone(),
            state: State::Normal,
            id_tokens: Vec::new(),
            body_tokens: Vec::new(),
            buf: Vec::new(),
        }
    }

    /// Observe a freshly-emitted token for `<think>` / `</think>` boundaries
    /// and update `in_think`. Returns true if this token was part of a think
    /// boundary (caller will still pass it through).
    fn observe_think_boundary(&mut self, token_id: u32) {
        // Single-token boundary fast path.
        if self.think_open_ids.len() == 1 && self.think_open_ids[0] == token_id {
            self.in_think = true;
            self.think_buf.clear();
            return;
        }
        if self.think_close_ids.len() == 1 && self.think_close_ids[0] == token_id {
            self.in_think = false;
            self.think_buf.clear();
            return;
        }
        // Multi-token boundaries: accumulate and check.
        self.think_buf.push(token_id);
        let max_len = self.think_open_ids.len().max(self.think_close_ids.len());
        if self.think_buf.len() > max_len {
            let drop = self.think_buf.len() - max_len;
            self.think_buf.drain(..drop);
        }
        if !self.think_open_ids.is_empty()
            && self.think_buf.ends_with(&self.think_open_ids)
        {
            self.in_think = true;
            self.think_buf.clear();
            return;
        }
        if !self.think_close_ids.is_empty()
            && self.think_buf.ends_with(&self.think_close_ids)
        {
            self.in_think = false;
            self.think_buf.clear();
        }
    }

    /// True if `buf` ends with `pat`, treating any `close_bracket_alts`
    /// token as equivalent to the canonical `]` (token IDs in
    /// `close_bracket_alts`). Specifically, positions where `pat` is one of
    /// those alts may be filled by any of the alts in `buf`. The non-`]`
    /// positions still match strictly.
    fn ends_with(&self, buf: &[u32], pat: &[u32]) -> bool {
        if pat.is_empty() || buf.len() < pat.len() {
            return false;
        }
        let off = buf.len() - pat.len();
        for (i, &p) in pat.iter().enumerate() {
            let b = buf[off + i];
            if b == p {
                continue;
            }
            // Allow any closing-bracket variant to stand in for the
            // canonical `]` (or vice versa).
            if self.close_bracket_alts.contains(&p)
                && self.close_bracket_alts.contains(&b)
            {
                continue;
            }
            return false;
        }
        true
    }

    /// True if `buf` is a prefix of `pat`, with the same `]`-equivalence
    /// rule as `ends_with`.
    fn is_prefix(&self, buf: &[u32], pat: &[u32]) -> bool {
        if pat.is_empty() || buf.len() > pat.len() {
            return false;
        }
        for (i, &b) in buf.iter().enumerate() {
            let p = pat[i];
            if b == p {
                continue;
            }
            if self.close_bracket_alts.contains(&p)
                && self.close_bracket_alts.contains(&b)
            {
                continue;
            }
            return false;
        }
        true
    }

    fn match_full(&self, buf: &[u32]) -> (Delim, usize) {
        if self.ends_with(buf, &self.call_ids) {
            return (Delim::Call, self.call_ids.len());
        }
        if self.ends_with(buf, &self.end_ids) {
            return (Delim::End, self.end_ids.len());
        }
        for s in &self.trap_end_ids {
            if self.ends_with(buf, s) {
                return (Delim::TrapEnd, s.len());
            }
        }
        if self.ends_with(buf, &self.trap_ids) {
            return (Delim::Trap, self.trap_ids.len());
        }
        if self.ends_with(buf, &self.intr_ids) {
            return (Delim::Intr, self.intr_ids.len());
        }
        if self.ends_with(buf, &self.head_ids) {
            return (Delim::Head, self.head_ids.len());
        }
        (Delim::None, 0)
    }

    fn match_inner(&self, buf: &[u32]) -> (Delim, usize) {
        if self.ends_with(buf, &self.call_inner) {
            return (Delim::Call, self.call_inner.len());
        }
        if self.ends_with(buf, &self.end_inner) {
            return (Delim::End, self.end_inner.len());
        }
        if self.ends_with(buf, &self.trap_inner) {
            return (Delim::Trap, self.trap_inner.len());
        }
        if self.ends_with(buf, &self.intr_inner) {
            return (Delim::Intr, self.intr_inner.len());
        }
        if self.ends_with(buf, &self.head_inner) {
            return (Delim::Head, self.head_inner.len());
        }
        (Delim::None, 0)
    }

    /// Like `match_inner` but restricted to delimiters that can appear at
    /// the top level in Normal state: [CALL], [TRAP], [TRAP][END]. Used to
    /// catch the case where the leading `[` of a top-level delimiter has
    /// been BPE-merged into the previous token.
    fn match_normal_inner(&self, buf: &[u32]) -> (Delim, usize) {
        for s in &self.trap_end_inner {
            if self.ends_with(buf, s) {
                return (Delim::TrapEnd, s.len());
            }
        }
        if self.ends_with(buf, &self.call_inner) {
            return (Delim::Call, self.call_inner.len());
        }
        if self.ends_with(buf, &self.trap_inner) {
            return (Delim::Trap, self.trap_inner.len());
        }
        if self.ends_with(buf, &self.intr_inner) {
            return (Delim::Intr, self.intr_inner.len());
        }
        (Delim::None, 0)
    }

    fn is_prefix_of_any(&self, buf: &[u32]) -> bool {
        self.is_prefix(buf, &self.call_ids)
            || self.is_prefix(buf, &self.end_ids)
            || self.is_prefix(buf, &self.trap_ids)
            || self.is_prefix(buf, &self.intr_ids)
            || self.is_prefix(buf, &self.head_ids)
            || self.trap_end_ids.iter().any(|s| self.is_prefix(buf, s))
            || self.is_prefix(buf, &self.call_inner)
            || self.is_prefix(buf, &self.end_inner)
            || self.is_prefix(buf, &self.trap_inner)
            || self.is_prefix(buf, &self.intr_inner)
            || self.is_prefix(buf, &self.head_inner)
            || self.trap_end_inner.iter().any(|s| self.is_prefix(buf, s))
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

        // Think-block bypass: while the model is inside <think>…</think>,
        // suspend CML matching and pass tokens through verbatim. The model
        // routinely quotes literal `[CALL]` / `[TRAP][END]` while reasoning
        // about how to format its answer, and the parser must not act on
        // those.
        let was_in_think = self.in_think;
        self.observe_think_boundary(token_id);
        if was_in_think || self.in_think {
            events.push(Event::Passthrough(token_id));
            return events;
        }

        match self.state {
            State::Normal => {
                self.buf.push(token_id);
                // Try full match (leading `[` present as its own token)
                // first; fall back to bracket-free inner match to catch
                // BPE-merged leading brackets like ` [` or `\n[`.
                let (d, dlen) = {
                    let (d, l) = self.match_full(&self.buf);
                    if d != Delim::None {
                        (d, l)
                    } else {
                        self.match_normal_inner(&self.buf)
                    }
                };
                match d {
                    Delim::Call => {
                        let end = self.buf.len() - dlen;
                        let mut prefix: Vec<u32> = self.buf[..end].to_vec();
                        Self::strip_trailing_bracket(&mut prefix, tokenizer);
                        for t in prefix {
                            events.push(Event::Passthrough(t));
                        }
                        self.buf.clear();
                        self.id_tokens.clear();
                        self.body_tokens.clear();
                        self.state = State::InCallId;
                    }
                    Delim::Trap => {
                        let end = self.buf.len() - dlen;
                        let mut prefix: Vec<u32> = self.buf[..end].to_vec();
                        Self::strip_trailing_bracket(&mut prefix, tokenizer);
                        for t in prefix {
                            events.push(Event::Passthrough(t));
                        }
                        self.buf.clear();
                        self.state = State::InTrap;
                    }
                    Delim::TrapEnd => {
                        let end = self.buf.len() - dlen;
                        let mut prefix: Vec<u32> = self.buf[..end].to_vec();
                        Self::strip_trailing_bracket(&mut prefix, tokenizer);
                        for t in prefix {
                            events.push(Event::Passthrough(t));
                        }
                        self.buf.clear();
                        events.push(Event::Trap);
                    }
                    Delim::Intr => {
                        let end = self.buf.len() - dlen;
                        let mut prefix: Vec<u32> = self.buf[..end].to_vec();
                        Self::strip_trailing_bracket(&mut prefix, tokenizer);
                        for t in prefix {
                            events.push(Event::Passthrough(t));
                        }
                        self.buf.clear();
                        self.state = State::InIntrId;
                    }
                    Delim::None => {
                        while !self.buf.is_empty() && !self.is_prefix_of_any(&self.buf) {
                            let t = self.buf.remove(0);
                            // A token that decodes ending with `[` was held
                            // as a potential delimiter start but no delim
                            // formed — typically the aborted result of
                            // [INTR] suppression. Drop rather than leak a
                            // stray `[` into the passthrough stream.
                            if Self::detokenize_one(tokenizer, t).ends_with('[') {
                                continue;
                            }
                            events.push(Event::Passthrough(t));
                        }
                    }
                    _ => {
                        for &t in &self.buf {
                            if Self::detokenize_one(tokenizer, t).ends_with('[') {
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
                    if d != Delim::None {
                        (d, l)
                    } else {
                        self.match_inner(&self.buf)
                    }
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
                        let id = tokenizer
                            .decode(&self.id_tokens)
                            .unwrap_or_default()
                            .trim()
                            .to_string();
                        events.push(Event::Call {
                            id,
                            code: String::new(),
                        });
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
                    if d != Delim::None {
                        (d, l)
                    } else {
                        self.match_inner(&self.buf)
                    }
                };
                match d {
                    Delim::End => {
                        let body_end = self.buf.len() - dlen;
                        self.body_tokens.extend_from_slice(&self.buf[..body_end]);
                        self.buf.clear();
                        Self::strip_trailing_bracket(&mut self.body_tokens, tokenizer);
                        let id = tokenizer
                            .decode(&self.id_tokens)
                            .unwrap_or_default()
                            .trim()
                            .to_string();
                        let code = tokenizer
                            .decode(&self.body_tokens)
                            .unwrap_or_default()
                            .trim()
                            .to_string();
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
                    if d != Delim::None {
                        (d, l)
                    } else {
                        self.match_inner(&self.buf)
                    }
                };
                match d {
                    Delim::End => {
                        self.buf.clear();
                        events.push(Event::Trap);
                        self.state = State::Normal;
                    }
                    Delim::None => {
                        if !self.is_prefix_of_any(&self.buf) {
                            self.buf.clear();
                        }
                    }
                    _ => {}
                }
            }

            State::InIntrId => {
                self.buf.push(token_id);
                let (d, _) = {
                    let (d, l) = self.match_full(&self.buf);
                    if d != Delim::None {
                        (d, l)
                    } else {
                        self.match_inner(&self.buf)
                    }
                };
                match d {
                    Delim::Head => {
                        self.buf.clear();
                        self.state = State::InIntrBody;
                    }
                    Delim::End => {
                        self.buf.clear();
                        self.state = State::Normal;
                    }
                    Delim::None => {
                        if !self.is_prefix_of_any(&self.buf) {
                            self.buf.clear();
                        }
                    }
                    _ => {}
                }
            }

            State::InIntrBody => {
                self.buf.push(token_id);
                let (d, _) = {
                    let (d, l) = self.match_full(&self.buf);
                    if d != Delim::None {
                        (d, l)
                    } else {
                        self.match_inner(&self.buf)
                    }
                };
                match d {
                    Delim::End => {
                        self.buf.clear();
                        self.state = State::Normal;
                    }
                    Delim::None => {
                        if !self.is_prefix_of_any(&self.buf) {
                            self.buf.clear();
                        }
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
            .filter(|&t| !Self::detokenize_one(tokenizer, t).ends_with('['))
            .map(Event::Passthrough)
            .collect()
    }
}

// ============================================================================
// Async dummy executor
//
// `execute_call` yields `Poll::Pending` until a wall-clock deadline, then
// returns a canned string. Hand-polled with a no-op waker once per main-loop
// iteration so calls progress while tokens are being sampled — the core
// AsyncLM property.
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

    println!("[MinAsync] call id={} completed (wait {}ms)", id, wait_ms);
    let lc = code.to_lowercase();

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
    } else if lc.contains("stock_price") || lc.contains("get_stock") {
        if lc.contains("aapl") {
            "{\"ticker\": \"AAPL\", \"price_usd\": 189.50}".to_string()
        } else if lc.contains("goog") {
            "{\"ticker\": \"GOOG\", \"price_usd\": 141.20}".to_string()
        } else if lc.contains("tsla") {
            "{\"ticker\": \"TSLA\", \"price_usd\": 258.30}".to_string()
        } else {
            "{\"ticker\": \"UNKNOWN\", \"price_usd\": 100.00}".to_string()
        }
    } else if lc.contains("convert_currency") || lc.contains("exchange_rate") {
        if lc.contains("eur") {
            "{\"rate\": 0.92, \"quote\": \"USD->EUR\"}".to_string()
        } else if lc.contains("jpy") {
            "{\"rate\": 155.40, \"quote\": \"USD->JPY\"}".to_string()
        } else if lc.contains("gbp") {
            "{\"rate\": 0.79, \"quote\": \"USD->GBP\"}".to_string()
        } else {
            "{\"rate\": 1.00, \"quote\": \"USD->USD\"}".to_string()
        }
    } else if lc.contains("search_restaurants") || lc.contains("find_restaurants") {
        if lc.contains("tokyo") {
            "[\"Sukiyabashi Jiro\", \"Ichiran\", \"Sushi Saito\"]".to_string()
        } else if lc.contains("paris") {
            "[\"Le Jules Verne\", \"L'Ami Jean\", \"Septime\"]".to_string()
        } else {
            "[\"Katz's Deli\", \"Joe's Pizza\", \"Lombardi's\"]".to_string()
        }
    } else if lc.contains("get_reviews") || lc.contains("reviews(") {
        if lc.contains("katz") {
            "{\"stars\": 4.6, \"summary\": \"Iconic pastrami sandwiches\"}".to_string()
        } else if lc.contains("jiro") {
            "{\"stars\": 4.9, \"summary\": \"Legendary omakase, tiny counter\"}".to_string()
        } else if lc.contains("joe") {
            "{\"stars\": 4.4, \"summary\": \"Classic NY slice, cheap and fast\"}".to_string()
        } else {
            "{\"stars\": 4.0, \"summary\": \"Solid, well-reviewed spot\"}".to_string()
        }
    } else if lc.contains("get_time") || lc.contains("time_in") || lc.contains("current_time") {
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

/// Poll a pending future once. Returns the result if ready.
fn poll_once(p: &mut Pending) -> Option<String> {
    let waker = noop_waker();
    let mut cx = TaskContext::from_waker(&waker);
    match p.fut.as_mut().poll(&mut cx) {
        Poll::Ready(r) => Some(r),
        Poll::Pending => None,
    }
}

// ============================================================================
// Main
// ============================================================================

#[inferlet::main]
async fn main(input: Input) -> Result<String> {
    let model_name = runtime::models()
        .first()
        .cloned()
        .ok_or("No models available")?;
    let model = Model::load(&model_name)?;
    let tokenizer = model.tokenizer();
    let registry = CmlRegistry::new(&tokenizer);

    println!(
        "[MinAsync] call={:?} head={:?} end={:?} trap={:?} intr={:?}",
        registry.call_ids,
        registry.head_ids,
        registry.end_ids,
        registry.trap_ids,
        registry.intr_ids
    );

    let mut ctx = Context::new(&model)?;
    ctx.system(&input.system).user(&input.prompt).cue();

    // BRLE has to cover the model's FULL logit width — specials (e.g.
    // `<|im_end|>`, `</think>`) often live above the regular BPE vocab.
    // Truncating to `vocabs().len()` would leave the tail implicitly masked
    // and the model could never produce its stop/end tokens.
    let (sp_ids, _) = tokenizer.special_tokens();
    let vocab_size = tokenizer.vocabs().0.len() as u32;
    let mask_size = sp_ids
        .iter()
        .copied()
        .max()
        .map(|m| m + 1)
        .unwrap_or(vocab_size)
        .max(vocab_size);
    let suppress = SuppressMask::new(mask_size, &registry.suppressed);
    let stops = chat::stop_tokens(&model);

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

    // In-flight async calls, and completed results awaiting injection.
    let mut pending: Vec<Pending> = Vec::new();
    let mut ready_frames: Vec<String> = Vec::new();
    let mut flush_injected_prefix = false;
    let mut deferred_injected_tail: Option<u32> = None;

    let mut step_count: usize = 0;
    while let Some(mut step) = g.next()? {
        step_count += 1;
        let flushing_injected_prefix = flush_injected_prefix;
        if flushing_injected_prefix {
            step.clear_sampler();
            flush_injected_prefix = false;
        }
        let out = step.execute().await?;
        if DEBUG_TRACE {
            let dbg_tokens: Vec<String> = out
                .tokens
                .iter()
                .map(|&t| format!("{}={:?}", t, tokenizer.decode(&[t]).unwrap_or_default()))
                .collect();
            println!(
                "[dbg] step={} ntoks={} raw_slots={} tokens={:?}",
                step_count,
                out.tokens.len(),
                out.raw().slots.len(),
                dbg_tokens
            );
        }

        if flushing_injected_prefix {
            if let Some(tail) = deferred_injected_tail.take() {
                if DEBUG_TRACE {
                    println!(
                        "[dbg] staged final injected token {}={:?} for sampled resume",
                        tail,
                        tokenizer.decode(&[tail]).unwrap_or_default()
                    );
                }
                let _ = g.accept(&[tail]);
            }
            continue;
        }

        // Advance each in-flight call by one poll — concurrent with token
        // generation. Core AsyncLM property: tool calls progress while
        // tokens are being sampled.
        pending.retain_mut(|p| match poll_once(p) {
            Some(result) => {
                println!(
                    "[MinAsync] id={} ready after {}ms",
                    p.id,
                    p.dispatched_at.elapsed().as_millis()
                );
                ready_frames.push(format!(" [INTR] {} [HEAD] {} [END] ", p.id, result));
                false
            }
            None => true,
        });

        for &tok in &out.tokens {
            for event in parser.feed(tok, &tokenizer) {
                match event {
                    Event::Passthrough(t) => generated.push(t),
                    Event::Call { id, code } => {
                        println!(
                            "[MinAsync] dispatching id={} code={} (fake_wait={}ms)",
                            id, code, input.fake_wait_ms
                        );
                        pending.push(Pending {
                            id: id.clone(),
                            dispatched_at: Instant::now(),
                            fut: Box::pin(execute_call(id, code, input.fake_wait_ms)),
                        });
                    }
                    Event::Trap => {
                        // TRAP blocks: drain every remaining future to
                        // completion before injection.
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
                        println!(
                            "[MinAsync] trap drained in {}ms; injecting {} frames",
                            trap_start.elapsed().as_millis(),
                            ready_frames.len()
                        );
                        // Bundle all ready frames into a single `accept`
                        // call. v1/v3 use this pattern; multiple consecutive
                        // accepts work in theory but the runtime behaves
                        // oddly with the resulting large pending buffer.
                        let mut injected: Vec<u32> = Vec::new();
                        for frame in ready_frames.drain(..) {
                            println!("[MinAsync] injecting: {}", frame.trim());
                            injected.extend(tokenizer.encode(&frame));
                        }
                        if !injected.is_empty() {
                            if DEBUG_TRACE {
                                println!(
                                    "[dbg] accept injecting {} tokens",
                                    injected.len()
                                );
                            }
                            if injected.len() > 1 {
                                let tail = *injected.last().unwrap();
                                let prefix_len = injected.len() - 1;
                                let accepted = g.accept(&injected[..prefix_len]);
                                if DEBUG_TRACE {
                                    println!(
                                        "[dbg] accepted injected prefix {} tokens; deferring final token",
                                        accepted.len()
                                    );
                                }
                                deferred_injected_tail = Some(tail);
                                flush_injected_prefix = true;
                            } else {
                                let _ = g.accept(&injected);
                            }
                        }
                    }
                }
            }
        }
    }
    if DEBUG_TRACE {
        println!(
            "[dbg] loop exited after {} steps; generated.len={}; pending.len={}",
            step_count,
            generated.len(),
            pending.len()
        );
    }

    for ev in parser.flush(&tokenizer) {
        if let Event::Passthrough(t) = ev {
            generated.push(t);
        }
    }

    Ok(tokenizer.decode(&generated).unwrap_or_default())
}
