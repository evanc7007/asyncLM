//! AsyncLM inferlet — v1 (reference/archive, current SDK).
//!
//! Earliest variant. Compared to v2/v3 it adds:
//!   * Multi-variant CML token registry: each delimiter can match one of
//!     several token-id sequences, covering BPE merges like `\n[CALL]`
//!     (`[498, 24622, 60]`) vs bare `[CALL]` (`[58, 24622, 60]`).
//!   * Real broadcast/subscribe executor: dispatches each call on the
//!     `asynclm/call` topic and awaits replies on `asynclm/result`. A
//!     companion worker (separate inferlet or external process) is
//!     required for end-to-end runs.
//!
//! Drops vs. the original v1:
//!   * Custom `AsyncLmSampler` with repetition penalty — the current SDK
//!     has no custom-sampler hook. Suppression of `[INTR]`-unique token
//!     ids is implemented via a static `Constrain` mask; repetition
//!     penalty would require a probe + manual-sample + accept pattern.
//!   * In-place Swap/Recompute context restoration — `Generator` holds
//!     `Context` mutably and the SDK does not expose committed-page
//!     truncation. The trap classifier is preserved for observability;
//!     all branches execute as Keep at apply time.

use std::collections::{HashMap, HashSet, VecDeque};

use inferlet::messaging;
use inferlet::model::{Model, Tokenizer};
use inferlet::sample::Sampler;
use inferlet::{chat, runtime, Constrain, Context, Result, SubscriptionExt};
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
    #[serde(default = "default_max_context")]
    max_context: usize,
    #[serde(default = "default_dispatch_topic")]
    dispatch_topic: String,
    #[serde(default = "default_result_topic")]
    result_topic: String,
}

fn default_max_tokens() -> usize { 512 }
fn default_temperature() -> f32 { 0.6 }
fn default_top_p() -> f32 { 0.95 }
fn default_max_context() -> usize { 4096 }
fn default_dispatch_topic() -> String { "asynclm/call".to_string() }
fn default_result_topic() -> String { "asynclm/result".to_string() }
fn default_system() -> String {
    "You are an assistant that uses async function calls. You MUST use the exact syntax below.\n\
     \n\
     To call a function: [CALL] <id> [HEAD] <function_call> [END]\n\
     To wait for results: [TRAP][END]\n\
     \n\
     Example:\n\
     User: What's the weather in NYC and London?\n\
     Assistant: Let me check both.\n\
     [CALL] w1 [HEAD] get_weather(\"New York\") [END]\n\
     [CALL] w2 [HEAD] get_weather(\"London\") [END]\n\
     [TRAP][END]"
        .to_string()
}

// ============================================================================
// Section 1: CML Token Registry — multi-variant
// ============================================================================

struct CmlTokenRegistry {
    call_ids: Vec<Vec<u32>>,
    end_ids: Vec<Vec<u32>>,
    intr_ids: Vec<Vec<u32>>,
    trap_ids: Vec<Vec<u32>>,
    head_ids: Vec<Vec<u32>>,
    trap_end_ids: Vec<Vec<u32>>,
    suppressed_ids: HashSet<u32>,
}

impl CmlTokenRegistry {
    fn new(tokenizer: &Tokenizer) -> Self {
        let (special_ids, special_bytes) = tokenizer.special_tokens();
        let special_map: HashMap<Vec<u8>, u32> = special_bytes
            .into_iter()
            .zip(special_ids.into_iter())
            .collect();

        let call_ids = Self::resolve_variants(tokenizer, &special_map, "[CALL]");
        let end_ids = Self::resolve_variants(tokenizer, &special_map, "[END]");
        let intr_ids = Self::resolve_variants(tokenizer, &special_map, "[INTR]");
        let trap_ids = Self::resolve_variants(tokenizer, &special_map, "[TRAP]");
        let head_ids = Self::resolve_variants(tokenizer, &special_map, "[HEAD]");

        let trap_primary = &trap_ids[0];
        let end_primary = &end_ids[0];
        let trap_end_full: Vec<u32> = trap_primary
            .iter()
            .chain(end_primary.iter())
            .cloned()
            .collect();
        let mut trap_end_ids = vec![trap_end_full];
        let bracket_merge = tokenizer.encode("][");
        if bracket_merge.len() == 1 {
            let merged = bracket_merge[0];
            let mut merged_seq: Vec<u32> = trap_primary[..trap_primary.len() - 1].to_vec();
            merged_seq.push(merged);
            merged_seq.extend_from_slice(&end_primary[1..]);
            println!(
                "[AsyncLM] ][ BPE merge detected: token={} trap_end_merged={:?}",
                merged, merged_seq
            );
            trap_end_ids.push(merged_seq);
        }

        let shared_ids: HashSet<u32> = call_ids[0]
            .iter()
            .chain(end_ids[0].iter())
            .chain(trap_ids[0].iter())
            .chain(head_ids[0].iter())
            .copied()
            .collect();
        let suppressed_ids: HashSet<u32> = intr_ids[0]
            .iter()
            .copied()
            .filter(|id| !shared_ids.contains(id))
            .collect();

        CmlTokenRegistry {
            call_ids,
            end_ids,
            intr_ids,
            trap_ids,
            head_ids,
            trap_end_ids,
            suppressed_ids,
        }
    }

    /// Resolve `tag` to a primary token-id sequence plus any context-merged
    /// variants we can detect via offline tokenization probes (`\n` + tag,
    /// ` ` + tag).
    fn resolve_variants(
        tokenizer: &Tokenizer,
        special_map: &HashMap<Vec<u8>, u32>,
        tag: &str,
    ) -> Vec<Vec<u32>> {
        let primary = if let Some(&id) = special_map.get(tag.as_bytes()) {
            vec![id]
        } else {
            tokenizer.encode(tag)
        };
        assert!(!primary.is_empty(), "tag {} tokenized to empty", tag);

        let mut variants = vec![primary.clone()];
        for prefix in ["\n", " "] {
            let with_prefix = tokenizer.encode(&format!("{}{}", prefix, tag));
            // Keep only variants that don't already start with the primary
            // sequence's first token — those would re-match through the
            // primary path.
            if !with_prefix.is_empty()
                && with_prefix.first() != primary.first()
                && !variants.iter().any(|v| v == &with_prefix)
            {
                variants.push(with_prefix);
            }
        }
        variants
    }
}

// ============================================================================
// Section 2: Constraint — suppress [INTR]-unique tokens (BRLE mask, static)
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
// Section 3: CML parser (FSM) — multi-variant matching
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq)]
enum CmlState {
    Normal,
    InCallId,
    InCallBody,
    InTrap,
}

#[derive(Debug)]
enum CmlEvent {
    CallDetected { id: String, code: String },
    TrapDetected,
    EnterCriticalSection,
    ExitCriticalSection,
    PassthroughToken { token_id: u32 },
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum CmlDelimiter {
    Call,
    End,
    Trap,
    TrapEnd,
    Head,
    None,
}

struct CmlParser {
    call_ids: Vec<Vec<u32>>,
    end_ids: Vec<Vec<u32>>,
    trap_ids: Vec<Vec<u32>>,
    head_ids: Vec<Vec<u32>>,
    trap_end_ids: Vec<Vec<u32>>,
    call_inner: Vec<u32>,
    head_inner: Vec<u32>,
    end_inner: Vec<u32>,
    trap_inner: Vec<u32>,
    state: CmlState,
    id_tokens: Vec<u32>,
    body_tokens: Vec<u32>,
    match_buf: Vec<u32>,
}

impl CmlParser {
    fn new(registry: &CmlTokenRegistry) -> Self {
        let call_primary = &registry.call_ids[0];
        let head_primary = &registry.head_ids[0];
        let end_primary = &registry.end_ids[0];
        let trap_primary = &registry.trap_ids[0];
        CmlParser {
            call_ids: registry.call_ids.clone(),
            end_ids: registry.end_ids.clone(),
            trap_ids: registry.trap_ids.clone(),
            head_ids: registry.head_ids.clone(),
            trap_end_ids: registry.trap_end_ids.clone(),
            call_inner: call_primary[1..].to_vec(),
            head_inner: head_primary[1..].to_vec(),
            end_inner: end_primary[1..].to_vec(),
            trap_inner: trap_primary[1..].to_vec(),
            state: CmlState::Normal,
            id_tokens: Vec::new(),
            body_tokens: Vec::new(),
            match_buf: Vec::new(),
        }
    }

    fn buf_ends_with(buf: &[u32], pattern: &[u32]) -> bool {
        !pattern.is_empty()
            && buf.len() >= pattern.len()
            && buf[buf.len() - pattern.len()..] == *pattern
    }

    fn match_delimiter(&self, buf: &[u32]) -> (CmlDelimiter, usize) {
        for seq in &self.call_ids {
            if Self::buf_ends_with(buf, seq) {
                return (CmlDelimiter::Call, seq.len());
            }
        }
        for seq in &self.end_ids {
            if Self::buf_ends_with(buf, seq) {
                return (CmlDelimiter::End, seq.len());
            }
        }
        for seq in &self.trap_end_ids {
            if Self::buf_ends_with(buf, seq) {
                return (CmlDelimiter::TrapEnd, seq.len());
            }
        }
        for seq in &self.trap_ids {
            if Self::buf_ends_with(buf, seq) {
                return (CmlDelimiter::Trap, seq.len());
            }
        }
        for seq in &self.head_ids {
            if Self::buf_ends_with(buf, seq) {
                return (CmlDelimiter::Head, seq.len());
            }
        }
        (CmlDelimiter::None, 0)
    }

    fn match_inner_delimiter(&self, buf: &[u32]) -> (CmlDelimiter, usize) {
        if Self::buf_ends_with(buf, &self.call_inner) {
            return (CmlDelimiter::Call, self.call_inner.len());
        }
        if Self::buf_ends_with(buf, &self.end_inner) {
            return (CmlDelimiter::End, self.end_inner.len());
        }
        if Self::buf_ends_with(buf, &self.trap_inner) {
            return (CmlDelimiter::Trap, self.trap_inner.len());
        }
        if Self::buf_ends_with(buf, &self.head_inner) {
            return (CmlDelimiter::Head, self.head_inner.len());
        }
        (CmlDelimiter::None, 0)
    }

    fn is_prefix_of_any_delimiter(&self, buf: &[u32]) -> bool {
        for variants in [&self.call_ids, &self.end_ids, &self.trap_ids, &self.head_ids] {
            for seq in variants {
                if buf.len() <= seq.len() && seq[..buf.len()] == *buf {
                    return true;
                }
            }
        }
        for seq in &self.trap_end_ids {
            if buf.len() <= seq.len() && seq[..buf.len()] == *buf {
                return true;
            }
        }
        false
    }

    fn is_prefix_of_any_inner(&self, buf: &[u32]) -> bool {
        for pattern in [&self.call_inner, &self.end_inner, &self.trap_inner, &self.head_inner] {
            if !pattern.is_empty() && buf.len() <= pattern.len() && pattern[..buf.len()] == *buf {
                return true;
            }
        }
        false
    }

    fn strip_trailing_bracket(tokens: &mut Vec<u32>, tokenizer: &Tokenizer) {
        if let Some(&last) = tokens.last() {
            let decoded = tokenizer.decode(&[last]).unwrap_or_default();
            if decoded.ends_with('[') {
                tokens.pop();
            }
        }
    }

    fn feed(&mut self, token_id: u32, tokenizer: &Tokenizer) -> Vec<CmlEvent> {
        let mut events = Vec::new();

        match self.state {
            CmlState::Normal => {
                self.match_buf.push(token_id);
                let (delim, delim_len) = self.match_delimiter(&self.match_buf);
                match delim {
                    CmlDelimiter::Call => {
                        let passthrough_end = self.match_buf.len() - delim_len;
                        for i in 0..passthrough_end {
                            events.push(CmlEvent::PassthroughToken {
                                token_id: self.match_buf[i],
                            });
                        }
                        self.match_buf.clear();
                        events.push(CmlEvent::EnterCriticalSection);
                        self.id_tokens.clear();
                        self.body_tokens.clear();
                        self.state = CmlState::InCallId;
                    }
                    CmlDelimiter::Trap => {
                        let passthrough_end = self.match_buf.len() - delim_len;
                        for i in 0..passthrough_end {
                            events.push(CmlEvent::PassthroughToken {
                                token_id: self.match_buf[i],
                            });
                        }
                        self.match_buf.clear();
                        events.push(CmlEvent::EnterCriticalSection);
                        self.state = CmlState::InTrap;
                    }
                    CmlDelimiter::TrapEnd => {
                        let passthrough_end = self.match_buf.len() - delim_len;
                        for i in 0..passthrough_end {
                            events.push(CmlEvent::PassthroughToken {
                                token_id: self.match_buf[i],
                            });
                        }
                        self.match_buf.clear();
                        events.push(CmlEvent::EnterCriticalSection);
                        events.push(CmlEvent::TrapDetected);
                        events.push(CmlEvent::ExitCriticalSection);
                    }
                    CmlDelimiter::None => {
                        if !self.is_prefix_of_any_delimiter(&self.match_buf) {
                            let first = self.match_buf.remove(0);
                            events.push(CmlEvent::PassthroughToken { token_id: first });
                            while !self.match_buf.is_empty()
                                && !self.is_prefix_of_any_delimiter(&self.match_buf)
                            {
                                let t = self.match_buf.remove(0);
                                events.push(CmlEvent::PassthroughToken { token_id: t });
                            }
                        }
                    }
                    _ => {
                        for &t in &self.match_buf {
                            events.push(CmlEvent::PassthroughToken { token_id: t });
                        }
                        self.match_buf.clear();
                    }
                }
            }

            CmlState::InCallId => {
                self.match_buf.push(token_id);
                let (delim, delim_len) = {
                    let (d, l) = self.match_delimiter(&self.match_buf);
                    if d != CmlDelimiter::None {
                        (d, l)
                    } else {
                        self.match_inner_delimiter(&self.match_buf)
                    }
                };
                match delim {
                    CmlDelimiter::Head => {
                        let id_end = self.match_buf.len() - delim_len;
                        self.id_tokens.extend_from_slice(&self.match_buf[..id_end]);
                        self.match_buf.clear();
                        Self::strip_trailing_bracket(&mut self.id_tokens, tokenizer);
                        self.state = CmlState::InCallBody;
                    }
                    CmlDelimiter::End => {
                        let id_end = self.match_buf.len() - delim_len;
                        self.id_tokens.extend_from_slice(&self.match_buf[..id_end]);
                        self.match_buf.clear();
                        Self::strip_trailing_bracket(&mut self.id_tokens, tokenizer);
                        let id = tokenizer
                            .decode(&self.id_tokens)
                            .unwrap_or_default()
                            .trim()
                            .to_string();
                        events.push(CmlEvent::CallDetected {
                            id,
                            code: String::new(),
                        });
                        events.push(CmlEvent::ExitCriticalSection);
                        self.id_tokens.clear();
                        self.state = CmlState::Normal;
                    }
                    CmlDelimiter::Call => {
                        self.match_buf.clear();
                        self.id_tokens.clear();
                        self.body_tokens.clear();
                    }
                    CmlDelimiter::None => {
                        let is_prefix = self.is_prefix_of_any_delimiter(&self.match_buf)
                            || self.is_prefix_of_any_inner(&self.match_buf);
                        if !is_prefix {
                            self.id_tokens.extend_from_slice(&self.match_buf);
                            self.match_buf.clear();
                        }
                    }
                    _ => {}
                }
            }

            CmlState::InCallBody => {
                self.match_buf.push(token_id);
                let (delim, delim_len) = {
                    let (d, l) = self.match_delimiter(&self.match_buf);
                    if d != CmlDelimiter::None {
                        (d, l)
                    } else {
                        self.match_inner_delimiter(&self.match_buf)
                    }
                };
                match delim {
                    CmlDelimiter::End => {
                        let body_end = self.match_buf.len() - delim_len;
                        self.body_tokens.extend_from_slice(&self.match_buf[..body_end]);
                        self.match_buf.clear();
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
                        events.push(CmlEvent::CallDetected { id, code });
                        events.push(CmlEvent::ExitCriticalSection);
                        self.id_tokens.clear();
                        self.body_tokens.clear();
                        self.state = CmlState::Normal;
                    }
                    CmlDelimiter::None => {
                        let is_prefix = self.is_prefix_of_any_delimiter(&self.match_buf)
                            || self.is_prefix_of_any_inner(&self.match_buf);
                        if !is_prefix {
                            self.body_tokens.extend_from_slice(&self.match_buf);
                            self.match_buf.clear();
                        }
                    }
                    _ => {}
                }
            }

            CmlState::InTrap => {
                self.match_buf.push(token_id);
                let (delim, _) = {
                    let (d, l) = self.match_delimiter(&self.match_buf);
                    if d != CmlDelimiter::None {
                        (d, l)
                    } else {
                        self.match_inner_delimiter(&self.match_buf)
                    }
                };
                match delim {
                    CmlDelimiter::End => {
                        self.match_buf.clear();
                        events.push(CmlEvent::TrapDetected);
                        events.push(CmlEvent::ExitCriticalSection);
                        self.state = CmlState::Normal;
                    }
                    CmlDelimiter::None => {
                        let is_prefix = self.is_prefix_of_any_delimiter(&self.match_buf)
                            || self.is_prefix_of_any_inner(&self.match_buf);
                        if !is_prefix {
                            self.match_buf.clear();
                        }
                    }
                    _ => {}
                }
            }
        }

        events
    }

    fn in_critical_section(&self) -> bool {
        self.state != CmlState::Normal
    }

    fn flush_pending(&mut self) -> Vec<CmlEvent> {
        let mut events = Vec::new();
        for &t in &self.match_buf {
            events.push(CmlEvent::PassthroughToken { token_id: t });
        }
        self.match_buf.clear();
        events
    }
}

// ============================================================================
// Section 4: Message Executor — broadcast/subscribe IPC
//
// Companion worker contract:
//   - Subscribe to `dispatch_topic` (default `asynclm/call`)
//   - For each message of form `id|code`, execute and broadcast on
//     `result_topic` (default `asynclm/result`) as `id|is_error|output`
// ============================================================================

#[derive(Debug, Clone)]
struct ExecutionResult {
    id: String,
    output: String,
    is_error: bool,
}

struct MessageExecutor {
    pending: HashSet<String>,
    completed: VecDeque<ExecutionResult>,
    dispatch_topic: String,
    sub: messaging::Subscription,
}

impl MessageExecutor {
    fn new(dispatch_topic: String, result_topic: String) -> Self {
        // Subscribe BEFORE any dispatch so we don't miss replies.
        let sub = messaging::subscribe(&result_topic);
        MessageExecutor {
            pending: HashSet::new(),
            completed: VecDeque::new(),
            dispatch_topic,
            sub,
        }
    }

    fn dispatch(&mut self, id: &str, code: &str) {
        let payload = format!("{}|{}", id, code);
        messaging::broadcast(&self.dispatch_topic, &payload);
        self.pending.insert(id.to_string());
        println!("[AsyncLM] dispatched id={} on {}", id, self.dispatch_topic);
    }

    /// Non-blocking poll: returns any results that arrived since the last call.
    fn poll_completed(&mut self) -> Vec<ExecutionResult> {
        let mut out = Vec::new();
        out.extend(self.completed.drain(..));
        while let Some(msg) = self.sub.get() {
            let res = Self::parse_result(&msg);
            self.pending.remove(&res.id);
            out.push(res);
        }
        out
    }

    /// Block until every pending call has produced a result.
    async fn wait_all(&mut self) -> Vec<ExecutionResult> {
        let mut results: Vec<ExecutionResult> = self.completed.drain(..).collect();
        while !self.pending.is_empty() {
            let Some(msg) = self.sub.get_async().await else {
                println!("[AsyncLM] subscription closed; aborting wait_all");
                break;
            };
            let res = Self::parse_result(&msg);
            self.pending.remove(&res.id);
            results.push(res);
        }
        results
    }

    fn parse_result(msg: &str) -> ExecutionResult {
        let mut parts = msg.splitn(3, '|');
        let id = parts.next().unwrap_or("unknown").to_string();
        let is_error = parts.next().unwrap_or("false") == "true";
        let output = parts.next().unwrap_or("").to_string();
        ExecutionResult { id, output, is_error }
    }

    fn pending_count(&self) -> usize {
        self.pending.len()
    }

    fn has_pending(&self) -> bool {
        !self.pending.is_empty()
    }
}

// ============================================================================
// Section 5: Interrupt Manager
// ============================================================================

struct InterruptManager {
    pending: VecDeque<ExecutionResult>,
    in_critical_section: bool,
}

impl InterruptManager {
    fn new() -> Self {
        InterruptManager {
            pending: VecDeque::new(),
            in_critical_section: false,
        }
    }

    fn enqueue(&mut self, result: ExecutionResult) {
        self.pending.push_back(result);
    }

    fn set_critical_section(&mut self, val: bool) {
        self.in_critical_section = val;
    }

    fn drain(&mut self) -> Vec<String> {
        if self.in_critical_section {
            return Vec::new();
        }
        let mut out = Vec::new();
        while let Some(r) = self.pending.pop_front() {
            let body = if r.is_error {
                format!("ERROR: {}", r.output)
            } else {
                r.output
            };
            out.push(format!(" [INTR] {} [HEAD] {} [END] ", r.id, body));
        }
        out
    }

    fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }
}

// ============================================================================
// Section 6: Trap Handler — classifier (Keep / Swap / Recompute)
//
// Records the decision for observability; only Keep is executed (see module
// docs for why Swap/Recompute aren't supported in this port).
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq)]
enum TrapStrategy {
    Keep,
    Swap,
    Recompute,
}

struct TrapConfig {
    context_pressure_threshold: f32,
    keep_max_pending: usize,
    stale_token_threshold: usize,
    max_context_length: usize,
}

struct TrapHandler {
    config: TrapConfig,
}

impl TrapHandler {
    fn new(config: TrapConfig) -> Self {
        TrapHandler { config }
    }

    fn decide(
        &self,
        has_checkpoint: bool,
        pending_calls: usize,
        context_len: usize,
        checkpoint_token_count: Option<usize>,
    ) -> TrapStrategy {
        if !has_checkpoint {
            return TrapStrategy::Keep;
        }
        let context_ratio = context_len as f32 / self.config.max_context_length as f32;
        if pending_calls <= self.config.keep_max_pending
            && context_ratio < self.config.context_pressure_threshold
        {
            return TrapStrategy::Keep;
        }
        if context_ratio >= self.config.context_pressure_threshold {
            let stale_tokens = checkpoint_token_count
                .map(|cp| context_len.saturating_sub(cp))
                .unwrap_or(0);
            if stale_tokens > self.config.stale_token_threshold {
                return TrapStrategy::Recompute;
            }
            return TrapStrategy::Swap;
        }
        TrapStrategy::Keep
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
    let registry = CmlTokenRegistry::new(&tokenizer);
    let vocab_size = tokenizer.vocabs().0.len() as u32;

    println!(
        "[AsyncLM] CML tokens: call={:?} end={:?} head={:?} trap={:?} intr={:?}",
        registry.call_ids[0],
        registry.end_ids[0],
        registry.head_ids[0],
        registry.trap_ids[0],
        registry.intr_ids[0]
    );
    println!("[AsyncLM] temperature={} top_p={}", input.temperature, input.top_p);

    let trap_config = TrapConfig {
        context_pressure_threshold: 0.75,
        keep_max_pending: 2,
        stale_token_threshold: 64,
        max_context_length: input.max_context,
    };
    let trap_handler = TrapHandler::new(trap_config);

    let mut ctx = Context::new(&model)?;
    ctx.system(&input.system).user(&input.prompt).cue();

    let stops = chat::stop_tokens(&model);
    let suppress = SuppressMask::new(vocab_size, &registry.suppressed_ids);

    let mut g = ctx
        .generate(Sampler::TopP {
            temperature: input.temperature,
            p: input.top_p,
        })
        .max_tokens(input.max_tokens)
        .stop(&stops)
        .constrain(suppress);

    let mut parser = CmlParser::new(&registry);
    let mut executor = MessageExecutor::new(input.dispatch_topic.clone(), input.result_topic.clone());
    let mut interrupts = InterruptManager::new();
    let mut generated: Vec<u32> = Vec::new();

    while let Some(step) = g.next()? {
        let out = step.execute().await?;
        if out.tokens.is_empty() {
            continue;
        }

        // Non-blocking pull from the result topic.
        for result in executor.poll_completed() {
            println!("[AsyncLM] result id={} is_error={}", result.id, result.is_error);
            interrupts.enqueue(result);
        }

        for &tok in &out.tokens {
            for event in parser.feed(tok, &tokenizer) {
                match event {
                    CmlEvent::PassthroughToken { token_id } => generated.push(token_id),
                    CmlEvent::EnterCriticalSection => interrupts.set_critical_section(true),
                    CmlEvent::ExitCriticalSection => interrupts.set_critical_section(false),
                    CmlEvent::CallDetected { id, code } => {
                        executor.dispatch(&id, &code);
                    }
                    CmlEvent::TrapDetected => {
                        if !executor.has_pending() && interrupts.is_empty() {
                            println!("[AsyncLM] trap with nothing pending — no-op");
                            continue;
                        }
                        let context_len = g.tokens_generated();
                        let strategy = trap_handler.decide(
                            false, // no checkpoint support in this port
                            executor.pending_count(),
                            context_len,
                            None,
                        );
                        println!(
                            "[AsyncLM] trap decision: {:?} (Keep executed regardless)",
                            strategy
                        );

                        let results = executor.wait_all().await;
                        for r in results {
                            interrupts.enqueue(r);
                        }

                        interrupts.set_critical_section(false);
                        let mut injected: Vec<u32> = Vec::new();
                        for frame in interrupts.drain() {
                            println!("[AsyncLM] inject: {}", frame.trim());
                            injected.extend(tokenizer.encode(&frame));
                        }
                        if !injected.is_empty() {
                            let accepted = g.accept(&injected);
                            for &t in &accepted {
                                for ev in parser.feed(t, &tokenizer) {
                                    if let CmlEvent::PassthroughToken { token_id } = ev {
                                        generated.push(token_id);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Mid-stream injection — drains anything safe to inject between calls.
        if !interrupts.in_critical_section && parser.match_buf.is_empty() {
            let mut injected: Vec<u32> = Vec::new();
            for frame in interrupts.drain() {
                println!("[AsyncLM] mid-stream inject: {}", frame.trim());
                injected.extend(tokenizer.encode(&frame));
            }
            if !injected.is_empty() {
                let accepted = g.accept(&injected);
                for &t in &accepted {
                    for ev in parser.feed(t, &tokenizer) {
                        if let CmlEvent::PassthroughToken { token_id } = ev {
                            generated.push(token_id);
                        }
                    }
                }
            }
        }
    }

    for ev in parser.flush_pending() {
        if let CmlEvent::PassthroughToken { token_id } = ev {
            generated.push(token_id);
        }
    }
    if parser.in_critical_section() {
        println!("[AsyncLM] Warning: generation stopped mid-CML block; partial block discarded");
    }

    Ok(tokenizer.decode(&generated).unwrap_or_default())
}
