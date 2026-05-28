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
        let special_map = build_special_map(tokenizer);
        let resolve = |s: &str| resolve_token(s, &special_map, tokenizer);
        let require = |s: &str| {
            let ids = resolve(s);
            assert!(!ids.is_empty(), "CML token '{}' tokenized to empty", s);
            ids
        };

        let call_ids = require("[CALL]");
        let end_ids = require("[END]");
        let head_ids = require("[HEAD]");
        let trap_ids = require("[TRAP]");
        let intr_ids = require("[INTR]");

        let trap_end_ids = build_trap_end_variants(&trap_ids, &end_ids, tokenizer);
        let close_bracket_alts = collect_close_bracket_alts(tokenizer);
        let suppressed = compute_intr_suppression(&call_ids, &end_ids, &head_ids, &trap_ids, &intr_ids);

        // Qwen3 think delimiters — empty Vec on tokenizers that lack them
        // (Qwen2.5 etc.), in which case the bypass is a no-op.
        let think_open_ids = resolve("<think>");
        let think_close_ids = resolve("</think>");

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

fn build_special_map(tokenizer: &Tokenizer) -> HashMap<Vec<u8>, u32> {
    let (special_ids, special_bytes) = tokenizer.special_tokens();
    special_bytes
        .into_iter()
        .zip(special_ids.into_iter())
        .collect()
}

fn resolve_token(s: &str, special_map: &HashMap<Vec<u8>, u32>, tokenizer: &Tokenizer) -> Vec<u32> {
    if let Some(&id) = special_map.get(s.as_bytes()) {
        vec![id]
    } else {
        tokenizer.encode(s)
    }
}

fn build_trap_end_variants(trap_ids: &[u32], end_ids: &[u32], tokenizer: &Tokenizer) -> Vec<Vec<u32>> {
    let mut variants = vec![trap_ids.iter().chain(end_ids.iter()).copied().collect::<Vec<u32>>()];
    let merge = tokenizer.encode("][");
    if merge.len() == 1 && !trap_ids.is_empty() && !end_ids.is_empty() {
        let mut merged: Vec<u32> = trap_ids[..trap_ids.len() - 1].to_vec();
        merged.push(merge[0]);
        merged.extend_from_slice(&end_ids[1..]);
        variants.push(merged);
    }
    variants
}

/// Every vocab token whose decoded bytes start with `]`. The suffix matcher
/// accepts any of these in place of the canonical `]` so BPE-fused trailing
/// brackets (`]\n`, `][`, `] `, …) still close out a delimiter. Without
/// this the FSM jams in InCallBody/InCallId forever — failures are silent.
fn collect_close_bracket_alts(tokenizer: &Tokenizer) -> Vec<u32> {
    let (vocab_ids, vocab_bytes) = tokenizer.vocabs();
    vocab_ids
        .iter()
        .zip(vocab_bytes.iter())
        .filter(|(_, bytes)| !bytes.is_empty() && bytes[0] == b']')
        .map(|(&id, _)| id)
        .collect()
}

/// Suppress only tokens *unique* to `[INTR]`. Shared `[` / `]` must stay
/// sampleable — they're needed for `[CALL]` / `[TRAP]` / `[END]`.
fn compute_intr_suppression(
    call_ids: &[u32],
    end_ids: &[u32],
    head_ids: &[u32],
    trap_ids: &[u32],
    intr_ids: &[u32],
) -> HashSet<u32> {
    let shared: HashSet<u32> = call_ids
        .iter()
        .chain(end_ids.iter())
        .chain(head_ids.iter())
        .chain(trap_ids.iter())
        .copied()
        .collect();
    intr_ids
        .iter()
        .copied()
        .filter(|id| !shared.contains(id))
        .collect()
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
    /// BRLE: alternating run lengths starting from `false` (disallowed).
    /// An initial run-length of 0 is used when id 0 happens to be allowed.
    fn new(mask_size: u32, suppressed: &HashSet<u32>) -> Self {
        let mut mask: Vec<u32> = Vec::new();
        let mut run_value = false;
        let mut run_len: u32 = 0;
        for id in 0..mask_size {
            let allowed = !suppressed.contains(&id);
            if allowed == run_value {
                run_len += 1;
            } else {
                mask.push(run_len);
                run_value = !run_value;
                run_len = 1;
            }
        }
        mask.push(run_len);
        Self { mask }
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
    // Full sequences (leading `[` present as its own token).
    call_ids: Vec<u32>,
    end_ids: Vec<u32>,
    head_ids: Vec<u32>,
    trap_ids: Vec<u32>,
    intr_ids: Vec<u32>,
    trap_end_ids: Vec<Vec<u32>>,
    // Bracket-free inner suffixes — leading-bracket token dropped. Used to
    // catch BPE-merged leading brackets (`\n[`, ` [`) where the `[` is
    // fused into the previous token.
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
    /// and update `in_think`.
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

    /// Position-level equivalence used by both `ends_with` and `is_prefix`:
    /// equal token ids match, and any `close_bracket_alts` token is
    /// interchangeable with another (so `]\n` stands in for the canonical
    /// `]`).
    fn pos_eq(&self, b: u32, p: u32) -> bool {
        b == p
            || (self.close_bracket_alts.contains(&p) && self.close_bracket_alts.contains(&b))
    }

    fn ends_with(&self, buf: &[u32], pat: &[u32]) -> bool {
        if pat.is_empty() || buf.len() < pat.len() {
            return false;
        }
        let off = buf.len() - pat.len();
        pat.iter()
            .enumerate()
            .all(|(i, &p)| self.pos_eq(buf[off + i], p))
    }

    fn is_prefix(&self, buf: &[u32], pat: &[u32]) -> bool {
        if pat.is_empty() || buf.len() > pat.len() {
            return false;
        }
        buf.iter()
            .enumerate()
            .all(|(i, &b)| self.pos_eq(b, pat[i]))
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

    /// `match_inner` restricted to delimiters that can appear at top level
    /// in Normal state ([CALL], [TRAP], [TRAP][END], [INTR]). Catches the
    /// case where the leading `[` of a top-level delimiter has been
    /// BPE-merged into the previous token.
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

    /// Try full-match first; fall back to inner-match (BPE-merged leading
    /// bracket). Used by every state except Normal.
    fn match_full_or_inner(&self, buf: &[u32]) -> (Delim, usize) {
        let r = self.match_full(buf);
        if r.0 != Delim::None { r } else { self.match_inner(buf) }
    }

    /// Normal-state variant: full-match first, then `match_normal_inner`
    /// (only delimiters legal at top level).
    fn match_full_or_normal_inner(&self, buf: &[u32]) -> (Delim, usize) {
        let r = self.match_full(buf);
        if r.0 != Delim::None { r } else { self.match_normal_inner(buf) }
    }

    fn is_prefix_of_any(&self, buf: &[u32]) -> bool {
        self.all_delim_patterns().any(|pat| self.is_prefix(buf, pat))
    }

    fn all_delim_patterns(&self) -> impl Iterator<Item = &[u32]> {
        let singles: [&[u32]; 10] = [
            &self.call_ids,
            &self.end_ids,
            &self.trap_ids,
            &self.intr_ids,
            &self.head_ids,
            &self.call_inner,
            &self.end_inner,
            &self.trap_inner,
            &self.intr_inner,
            &self.head_inner,
        ];
        let trap_end_full = self.trap_end_ids.iter().map(|s| s.as_slice());
        let trap_end_inner = self.trap_end_inner.iter().map(|s| s.as_slice());
        singles.into_iter().chain(trap_end_full).chain(trap_end_inner)
    }

    fn detokenize_one(tokenizer: &Tokenizer, t: u32) -> String {
        tokenizer.decode(&[t]).unwrap_or_default()
    }

    /// Tokens whose decoded bytes end with `[` are potential delimiter
    /// starts; if the delimiter never forms (usually because the [INTR]
    /// continuation was suppressed), we drop them rather than leak a stray
    /// `[` into passthrough.
    fn is_partial_delim_start(t: u32, tokenizer: &Tokenizer) -> bool {
        Self::detokenize_one(tokenizer, t).ends_with('[')
    }

    fn strip_trailing_bracket(tokens: &mut Vec<u32>, tokenizer: &Tokenizer) {
        if tokens.last().is_some_and(|&t| Self::is_partial_delim_start(t, tokenizer)) {
            tokens.pop();
        }
    }

    /// Emit `self.buf[..buf.len() - dlen]` as Passthrough events (stripping
    /// the trailing `[` that opens the matched delimiter), then clear buf.
    fn emit_prefix_and_consume(
        &mut self,
        dlen: usize,
        tokenizer: &Tokenizer,
        events: &mut Vec<Event>,
    ) {
        let end = self.buf.len() - dlen;
        let mut prefix: Vec<u32> = self.buf[..end].to_vec();
        Self::strip_trailing_bracket(&mut prefix, tokenizer);
        events.extend(prefix.into_iter().map(Event::Passthrough));
        self.buf.clear();
    }

    /// Pop one token from the front of `self.buf` and emit it as
    /// Passthrough unless it's a partial-delimiter start (in which case
    /// drop it). Used to drain `buf` token-by-token while it can't be the
    /// start of any delimiter.
    fn drain_front_as_passthrough(&mut self, tokenizer: &Tokenizer, events: &mut Vec<Event>) {
        let t = self.buf.remove(0);
        if !Self::is_partial_delim_start(t, tokenizer) {
            events.push(Event::Passthrough(t));
        }
    }

    fn feed(&mut self, token_id: u32, tokenizer: &Tokenizer) -> Vec<Event> {
        let mut events = Vec::new();

        // Think-block bypass: while inside <think>…</think>, suspend CML
        // matching and pass tokens through verbatim — the model routinely
        // quotes literal CML while reasoning about how to format its
        // answer, and the parser must not act on those.
        let was_in_think = self.in_think;
        self.observe_think_boundary(token_id);
        if was_in_think || self.in_think {
            events.push(Event::Passthrough(token_id));
            return events;
        }

        match self.state {
            State::Normal => self.feed_normal(token_id, tokenizer, &mut events),
            State::InCallId => self.feed_in_call_id(token_id, tokenizer, &mut events),
            State::InCallBody => self.feed_in_call_body(token_id, tokenizer, &mut events),
            State::InTrap => self.feed_in_trap(token_id, &mut events),
            State::InIntrId => self.feed_in_intr_id(token_id),
            State::InIntrBody => self.feed_in_intr_body(token_id),
        }

        events
    }

    fn feed_normal(&mut self, token_id: u32, tokenizer: &Tokenizer, events: &mut Vec<Event>) {
        self.buf.push(token_id);
        let (d, dlen) = self.match_full_or_normal_inner(&self.buf);
        match d {
            Delim::Call => {
                self.emit_prefix_and_consume(dlen, tokenizer, events);
                self.id_tokens.clear();
                self.body_tokens.clear();
                self.state = State::InCallId;
            }
            Delim::Trap => {
                self.emit_prefix_and_consume(dlen, tokenizer, events);
                self.state = State::InTrap;
            }
            Delim::TrapEnd => {
                self.emit_prefix_and_consume(dlen, tokenizer, events);
                events.push(Event::Trap);
            }
            Delim::Intr => {
                self.emit_prefix_and_consume(dlen, tokenizer, events);
                self.state = State::InIntrId;
            }
            Delim::None => {
                while !self.buf.is_empty() && !self.is_prefix_of_any(&self.buf) {
                    self.drain_front_as_passthrough(tokenizer, events);
                }
            }
            // End/Head matched at top level — should not occur, but drain
            // conservatively rather than leaving the buffer indefinitely.
            _ => {
                let drained: Vec<u32> = self.buf.drain(..).collect();
                for t in drained {
                    if !Self::is_partial_delim_start(t, tokenizer) {
                        events.push(Event::Passthrough(t));
                    }
                }
            }
        }
    }

    fn feed_in_call_id(&mut self, token_id: u32, tokenizer: &Tokenizer, events: &mut Vec<Event>) {
        self.buf.push(token_id);
        let (d, dlen) = self.match_full_or_inner(&self.buf);
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
                // Decode error → empty id, runtime keeps going (best-effort).
                let id = tokenizer
                    .decode(&self.id_tokens)
                    .unwrap_or_default()
                    .trim()
                    .to_string();
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

    fn feed_in_call_body(&mut self, token_id: u32, tokenizer: &Tokenizer, events: &mut Vec<Event>) {
        self.buf.push(token_id);
        let (d, dlen) = self.match_full_or_inner(&self.buf);
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

    fn feed_in_trap(&mut self, token_id: u32, events: &mut Vec<Event>) {
        self.buf.push(token_id);
        let (d, _) = self.match_full_or_inner(&self.buf);
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

    fn feed_in_intr_id(&mut self, token_id: u32) {
        self.buf.push(token_id);
        let (d, _) = self.match_full_or_inner(&self.buf);
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

    fn feed_in_intr_body(&mut self, token_id: u32) {
        self.buf.push(token_id);
        let (d, _) = self.match_full_or_inner(&self.buf);
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

    fn flush(&mut self, tokenizer: &Tokenizer) -> Vec<Event> {
        self.buf
            .drain(..)
            .filter(|&t| !Self::is_partial_delim_start(t, tokenizer))
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

/// Yield `Poll::Pending` until `wait_ms` has elapsed. The cooperative
/// `wake_by_ref` is what makes the hand-polled futures progress once per
/// decode step.
async fn wait_until_deadline(wait_ms: u64) {
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
}

async fn execute_call(id: String, code: String, wait_ms: u64) -> String {
    wait_until_deadline(wait_ms).await;
    println!("[MinAsync] call id={} completed (wait {}ms)", id, wait_ms);
    canned_result(&code.to_lowercase())
        .map(str::to_owned)
        .unwrap_or_else(|| format!("result({})", code))
}

/// Dispatch the lowercased call body to the right canned-result function.
/// Returns `None` when no tool name matches; the caller falls back to
/// `result(<code>)`.
fn canned_result(lc: &str) -> Option<&'static str> {
    if lc.contains("get_weather") || lc.contains("weather(") {
        Some(weather_result(lc))
    } else if lc.contains("stock_price") || lc.contains("get_stock") {
        Some(stock_result(lc))
    } else if lc.contains("convert_currency") || lc.contains("exchange_rate") {
        Some(currency_result(lc))
    } else if lc.contains("search_restaurants") || lc.contains("find_restaurants") {
        Some(restaurants_result(lc))
    } else if lc.contains("get_reviews") || lc.contains("reviews(") {
        Some(reviews_result(lc))
    } else if lc.contains("get_time") || lc.contains("time_in") || lc.contains("current_time") {
        Some(time_result(lc))
    } else {
        None
    }
}

fn weather_result(lc: &str) -> &'static str {
    if lc.contains("london") {
        r#"{"temp_f": 60, "sky": "cloudy"}"#
    } else if lc.contains("paris") {
        r#"{"temp_f": 59, "sky": "rainy"}"#
    } else if lc.contains("boston") {
        r#"{"temp_f": 68, "sky": "clear"}"#
    } else if lc.contains("tokyo") {
        r#"{"temp_f": 75, "sky": "humid"}"#
    } else {
        r#"{"temp_f": 72, "sky": "sunny"}"#
    }
}

fn stock_result(lc: &str) -> &'static str {
    if lc.contains("aapl") {
        r#"{"ticker": "AAPL", "price_usd": 189.50}"#
    } else if lc.contains("goog") {
        r#"{"ticker": "GOOG", "price_usd": 141.20}"#
    } else if lc.contains("tsla") {
        r#"{"ticker": "TSLA", "price_usd": 258.30}"#
    } else {
        r#"{"ticker": "UNKNOWN", "price_usd": 100.00}"#
    }
}

fn currency_result(lc: &str) -> &'static str {
    if lc.contains("eur") {
        r#"{"rate": 0.92, "quote": "USD->EUR"}"#
    } else if lc.contains("jpy") {
        r#"{"rate": 155.40, "quote": "USD->JPY"}"#
    } else if lc.contains("gbp") {
        r#"{"rate": 0.79, "quote": "USD->GBP"}"#
    } else {
        r#"{"rate": 1.00, "quote": "USD->USD"}"#
    }
}

fn restaurants_result(lc: &str) -> &'static str {
    if lc.contains("tokyo") {
        r#"["Sukiyabashi Jiro", "Ichiran", "Sushi Saito"]"#
    } else if lc.contains("paris") {
        r#"["Le Jules Verne", "L'Ami Jean", "Septime"]"#
    } else {
        r#"["Katz's Deli", "Joe's Pizza", "Lombardi's"]"#
    }
}

fn reviews_result(lc: &str) -> &'static str {
    if lc.contains("katz") {
        r#"{"stars": 4.6, "summary": "Iconic pastrami sandwiches"}"#
    } else if lc.contains("jiro") {
        r#"{"stars": 4.9, "summary": "Legendary omakase, tiny counter"}"#
    } else if lc.contains("joe") {
        r#"{"stars": 4.4, "summary": "Classic NY slice, cheap and fast"}"#
    } else {
        r#"{"stars": 4.0, "summary": "Solid, well-reviewed spot"}"#
    }
}

fn time_result(lc: &str) -> &'static str {
    if lc.contains("tokyo") || lc.contains("jst") {
        r#"{"time": "2026-04-21T23:32+09:00", "tz": "JST"}"#
    } else if lc.contains("london") || lc.contains("gmt") {
        r#"{"time": "2026-04-21T15:32+01:00", "tz": "BST"}"#
    } else if lc.contains("new_york") || lc.contains("nyc") || lc.contains("est") {
        r#"{"time": "2026-04-21T10:32-04:00", "tz": "EDT"}"#
    } else {
        r#"{"time": "2026-04-21T14:32+00:00", "tz": "UTC"}"#
    }
}

/// A call that has been dispatched but not yet observed as complete.
struct PendingCall {
    id: String,
    dispatched_at: Instant,
    fut: Pin<Box<dyn Future<Output = String>>>,
}

/// Poll a pending future once. Returns the result if ready.
fn poll_once(p: &mut PendingCall) -> Option<String> {
    let waker = noop_waker();
    let mut cx = TaskContext::from_waker(&waker);
    match p.fut.as_mut().poll(&mut cx) {
        Poll::Ready(r) => Some(r),
        Poll::Pending => None,
    }
}

// ============================================================================
// Main-loop helpers
// ============================================================================

fn intr_frame(id: &str, result: &str) -> String {
    format!(" [INTR] {} [HEAD] {} [END] ", id, result)
}

/// Poll each pending call once. Completed calls are removed from `pending`
/// and their result frames are pushed to `frames`. When `log_each_ready` is
/// true, a per-completion log line is emitted (used by the per-decode-step
/// drain; the TRAP-drain caller suppresses it and emits a single summary).
fn poll_and_collect_ready(
    pending: &mut Vec<PendingCall>,
    frames: &mut Vec<String>,
    log_each_ready: bool,
) {
    pending.retain_mut(|p| match poll_once(p) {
        Some(result) => {
            if log_each_ready {
                println!(
                    "[MinAsync] id={} ready after {}ms",
                    p.id,
                    p.dispatched_at.elapsed().as_millis()
                );
            }
            frames.push(intr_frame(&p.id, &result));
            false
        }
        None => true,
    });
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

    let mut generator = ctx
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
    let mut pending: Vec<PendingCall> = Vec::new();
    let mut ready_frames: Vec<String> = Vec::new();

    // CUDA-driver workaround state: when we inject >1 token at TRAP, we
    // accept all but the last this step, then the next step runs
    // sampler-cleared (prefill-only) and finally accepts the deferred tail
    // before resuming normal sampling. See asyncLM/CLAUDE.md for the
    // motivation.
    let mut next_step_flushes_injection = false;
    let mut deferred_injected_tail: Option<u32> = None;

    let mut step_count: usize = 0;
    while let Some(mut step) = generator.next()? {
        step_count += 1;

        let is_flush_step = next_step_flushes_injection;
        next_step_flushes_injection = false;
        if is_flush_step {
            step.clear_sampler();
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

        if is_flush_step {
            if let Some(tail) = deferred_injected_tail.take() {
                if DEBUG_TRACE {
                    println!(
                        "[dbg] staged final injected token {}={:?} for sampled resume",
                        tail,
                        tokenizer.decode(&[tail]).unwrap_or_default()
                    );
                }
                let _ = generator.accept(&[tail]);
            }
            continue;
        }

        // Advance each in-flight call by one poll — concurrent with token
        // generation. Core AsyncLM property: tool calls progress while
        // tokens are being sampled.
        poll_and_collect_ready(&mut pending, &mut ready_frames, true);

        for &tok in &out.tokens {
            for event in parser.feed(tok, &tokenizer) {
                match event {
                    Event::Passthrough(t) => generated.push(t),
                    Event::Call { id, code } => {
                        println!(
                            "[MinAsync] dispatching id={} code={} (fake_wait={}ms)",
                            id, code, input.fake_wait_ms
                        );
                        pending.push(PendingCall {
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
                            poll_and_collect_ready(&mut pending, &mut ready_frames, false);
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
                        if injected.is_empty() {
                            continue;
                        }
                        if DEBUG_TRACE {
                            println!("[dbg] accept injecting {} tokens", injected.len());
                        }
                        if injected.len() > 1 {
                            let tail = *injected.last().unwrap();
                            let prefix_len = injected.len() - 1;
                            let accepted = generator.accept(&injected[..prefix_len]);
                            if DEBUG_TRACE {
                                println!(
                                    "[dbg] accepted injected prefix {} tokens; deferring final token",
                                    accepted.len()
                                );
                            }
                            deferred_injected_tail = Some(tail);
                            next_step_flushes_injection = true;
                        } else {
                            let _ = generator.accept(&injected);
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
