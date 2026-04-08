//! AsyncLM Inferlet — Async function calling with parallel tool execution
//!
//! Implements the AsyncLM paper (https://arxiv.org/pdf/2412.07017): the LLM continues
//! generating while tool calls execute in parallel, with results injected back via interrupts.

use inferlet::context::Context;
use inferlet::sampler::Sample;
use inferlet::stop_condition::{ends_with_any, max_len, StopCondition};
use inferlet::{Args, Result, Sampler, Tokenizer};
use std::cell::RefCell;
use std::collections::{HashMap, HashSet, VecDeque};
use std::rc::Rc;

// ============================================================================
// Section 1: CML Token Registry
// ============================================================================

/// Resolves CML special tokens to their token IDs.
///
/// CML tokens may be single special tokens (if registered in the vocabulary)
/// or multi-token sequences (if the model doesn't have them as specials).
///
/// Each CML token is stored as a list of **variant** sequences to handle BPE
/// context-merging: e.g. `\n[CALL]` often tokenizes as `[498, 24622, 60]` while
/// `[CALL]` in isolation is `[58, 24622, 60]`.  Both variants are registered so
/// the parser matches either form.
struct CmlTokenRegistry {
    /// Each entry is one valid token-ID sequence for [CALL] (primary first).
    call_ids: Vec<Vec<u32>>,
    end_ids: Vec<Vec<u32>>,
    intr_ids: Vec<Vec<u32>>,
    trap_ids: Vec<Vec<u32>>,
    head_ids: Vec<Vec<u32>>,

    /// Combined [TRAP][END] sequences matched atomically.
    ///
    /// BPE tokenizers often merge `][` (closing bracket of [TRAP] + opening bracket
    /// of [END]) into a single token (e.g. 1404 = `][`).  Matching [TRAP][END] as
    /// a unit handles this without requiring InTrap state.
    trap_end_ids: Vec<Vec<u32>>,

    suppressed_ids: HashSet<u32>,
}

impl CmlTokenRegistry {
    fn new(tokenizer: &Tokenizer) -> Self {
        // Build a lookup from byte representation to token ID for special tokens
        let (special_ids, special_bytes) = tokenizer.get_special_tokens();
        let special_map: HashMap<Vec<u8>, u32> = special_bytes
            .into_iter()
            .zip(special_ids.into_iter())
            .collect();

        let call_ids = Self::resolve_variants(tokenizer, &special_map, "[CALL]");
        let end_ids  = Self::resolve_variants(tokenizer, &special_map, "[END]");
        let intr_ids = Self::resolve_variants(tokenizer, &special_map, "[INTR]");
        let trap_ids = Self::resolve_variants(tokenizer, &special_map, "[TRAP]");
        let head_ids = Self::resolve_variants(tokenizer, &special_map, "[HEAD]");

        // Build combined [TRAP][END] sequences matched atomically.
        // When `][` is a single BPE token (e.g. 1404), the full sequence becomes
        // [trap_primary[..-1], merged_bracket, end_primary[1..]] instead of
        // [trap_primary.., end_primary..].
        let trap_primary = &trap_ids[0];
        let end_primary  = &end_ids[0];
        let trap_end_full: Vec<u32> = trap_primary.iter().chain(end_primary.iter()).cloned().collect();
        let mut trap_end_ids = vec![trap_end_full];
        {
            let bracket_merge = tokenizer.tokenize("][");
            if bracket_merge.len() == 1 {
                let merged = bracket_merge[0];
                // [TRAP_without_close] + [merged_][_bracket] + [END_without_open]
                let mut merged_seq: Vec<u32> = trap_primary[..trap_primary.len() - 1].to_vec();
                merged_seq.push(merged);
                merged_seq.extend_from_slice(&end_primary[1..]);
                inferlet::send(&format!(
                    "[AsyncLM] ][ BPE merge detected: token={} trap_end_merged={:?}",
                    merged, merged_seq
                ));
                trap_end_ids.push(merged_seq);
            }
        }

        // Suppress only the token IDs *unique* to [INTR] primary sequence.
        // Shared bracket tokens (`[`, `]`) appear in every CML sequence and
        // must NOT be suppressed — they're needed for other delimiters.
        let shared_ids: HashSet<u32> = call_ids[0].iter()
            .chain(end_ids[0].iter())
            .chain(trap_ids[0].iter())
            .chain(head_ids[0].iter())
            .copied()
            .collect();
        let suppressed_ids: HashSet<u32> = intr_ids[0].iter()
            .copied()
            .filter(|id| !shared_ids.contains(id))
            .collect();

        CmlTokenRegistry { call_ids, end_ids, intr_ids, trap_ids, head_ids, trap_end_ids, suppressed_ids }
    }

    /// Resolve a CML token string to ALL valid token-ID sequences.
    ///
    /// BPE tokenizers frequently merge a preceding `\n` with `[` into a single
    /// token (e.g. `\n[CALL]` → `[498, 24622, 60]` vs `[CALL]` → `[58, 24622, 60]`).
    /// This method detects such merges and adds the merged form as an alternate
    /// variant so the parser recognises both.
    fn resolve_variants(
        tokenizer: &Tokenizer,
        special_map: &HashMap<Vec<u8>, u32>,
        text: &str,
    ) -> Vec<Vec<u32>> {
        let primary = Self::resolve_primary(tokenizer, special_map, text);

        // Tokenize with a newline prefix to detect \n+[ BPE merging.
        let with_nl = tokenizer.tokenize(&format!("\n{}", text));

        // Merged form: same length as primary but different first token,
        // with identical suffix.  This means `\n` collapsed into primary[0].
        if with_nl.len() == primary.len()
            && !with_nl.is_empty()
            && with_nl[0] != primary[0]
            && with_nl[1..] == primary[1..]
        {
            inferlet::send(&format!(
                "[AsyncLM] BPE merge detected for '{}': bare={:?} nl-merged={:?}",
                text, primary, with_nl
            ));
            vec![primary, with_nl]
        } else {
            vec![primary]
        }
    }

    /// Returns the primary (isolated) token-ID sequence for a CML token string.
    fn resolve_primary(
        tokenizer: &Tokenizer,
        special_map: &HashMap<Vec<u8>, u32>,
        text: &str,
    ) -> Vec<u32> {
        if let Some(&id) = special_map.get(text.as_bytes()) {
            return vec![id];
        }
        let ids = tokenizer.tokenize(text);
        assert!(!ids.is_empty(), "CML token '{}' tokenized to empty sequence", text);
        ids
    }
}

// ============================================================================
// Section 2: CML Parser (FSM)
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

/// Which CML delimiter was matched
#[derive(Debug, Clone, Copy, PartialEq)]
enum CmlDelimiter {
    Call,
    End,
    Trap,
    /// Matched full [TRAP][END] sequence atomically (handles BPE-merged `][` token).
    TrapEnd,
    Head,
    None,
}

struct CmlParser {
    /// Full sequences including bracket token — used in Normal state.
    call_ids: Vec<Vec<u32>>,
    end_ids: Vec<Vec<u32>>,
    trap_ids: Vec<Vec<u32>>,
    head_ids: Vec<Vec<u32>>,

    /// Combined [TRAP][END] sequences matched atomically in Normal state.
    /// Handles BPE `][` merge (e.g. token 1404 = `][`).
    trap_end_ids: Vec<Vec<u32>>,

    /// Bracket-free inner suffixes: primary[1..].
    ///
    /// BPE tokenizers often merge `\n` + `[` into a single token (e.g. 498 = `\n[`).
    /// We cannot reliably detect this via offline `tokenize()` probing because the merge
    /// only applies in context.  Inside InCallId/InCallBody/InTrap we already KNOW we are
    /// inside a CML block, so the leading bracket is redundant — we match only the
    /// tag-identifier + closing bracket.  This is unambiguous because these inner tokens
    /// (e.g. 24622 = `CALL`, 34261 = `HEAD`, 4537 = `END`) are highly model-specific.
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
        // Primary sequences (index 0 in each variant list)
        let call_primary = &registry.call_ids[0];
        let head_primary = &registry.head_ids[0];
        let end_primary  = &registry.end_ids[0];
        let trap_primary = &registry.trap_ids[0];

        // Inner suffix = everything after the first (bracket) token
        let call_inner = call_primary[1..].to_vec();
        let head_inner = head_primary[1..].to_vec();
        let end_inner  = end_primary[1..].to_vec();
        let trap_inner = trap_primary[1..].to_vec();

        inferlet::send(&format!(
            "[AsyncLM] CML inner: call={:?} head={:?} end={:?} trap={:?}",
            call_inner, head_inner, end_inner, trap_inner
        ));
        inferlet::send(&format!(
            "[AsyncLM] CML trap_end variants: {:?}",
            registry.trap_end_ids
        ));

        CmlParser {
            call_ids: registry.call_ids.clone(),
            end_ids: registry.end_ids.clone(),
            trap_ids: registry.trap_ids.clone(),
            head_ids: registry.head_ids.clone(),
            trap_end_ids: registry.trap_end_ids.clone(),
            call_inner,
            head_inner,
            end_inner,
            trap_inner,
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

    /// Match full delimiter sequences (for Normal state — requires leading bracket token).
    fn match_delimiter(&self, buf: &[u32]) -> (CmlDelimiter, usize) {
        for seq in &self.call_ids {
            if Self::buf_ends_with(buf, seq) { return (CmlDelimiter::Call, seq.len()); }
        }
        for seq in &self.end_ids {
            if Self::buf_ends_with(buf, seq) { return (CmlDelimiter::End, seq.len()); }
        }
        // Check combined [TRAP][END] before plain [TRAP] so BPE-merged `][` form wins.
        for seq in &self.trap_end_ids {
            if Self::buf_ends_with(buf, seq) { return (CmlDelimiter::TrapEnd, seq.len()); }
        }
        for seq in &self.trap_ids {
            if Self::buf_ends_with(buf, seq) { return (CmlDelimiter::Trap, seq.len()); }
        }
        for seq in &self.head_ids {
            if Self::buf_ends_with(buf, seq) { return (CmlDelimiter::Head, seq.len()); }
        }
        (CmlDelimiter::None, 0)
    }

    /// Match inner (bracket-free) suffixes (for InCallId/InCallBody/InTrap states).
    fn match_inner_delimiter(&self, buf: &[u32]) -> (CmlDelimiter, usize) {
        if Self::buf_ends_with(buf, &self.call_inner) { return (CmlDelimiter::Call, self.call_inner.len()); }
        if Self::buf_ends_with(buf, &self.end_inner)  { return (CmlDelimiter::End,  self.end_inner.len());  }
        if Self::buf_ends_with(buf, &self.trap_inner) { return (CmlDelimiter::Trap, self.trap_inner.len()); }
        if Self::buf_ends_with(buf, &self.head_inner) { return (CmlDelimiter::Head, self.head_inner.len()); }
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
        // Also check combined [TRAP][END] sequences (handles BPE-merged `][` token).
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

    /// Strip a trailing bracket token (`[` or `\n[`) from a token list.
    /// Called after matching a delimiter via its inner suffix so the bracket
    /// that was flushed to id/body tokens is removed.
    fn strip_trailing_bracket(tokens: &mut Vec<u32>, tokenizer: &Tokenizer) {
        if let Some(&last) = tokens.last() {
            let decoded = tokenizer.detokenize(&[last]);
            // Strip any token whose decoded form ends with `[`.
            // This covers bare `[` (58), `\n[` (498), ` [`, `Ċ[`, etc.
            if decoded.ends_with('[') {
                tokens.pop();
            }
        }
    }

    fn feed(&mut self, token_id: u32, tokenizer: &Tokenizer) -> Vec<CmlEvent> {
        inferlet::send(&format!("[Parser] token={} state={:?} buf={:?}", token_id, self.state, self.match_buf));
        let mut events = Vec::new();

        match self.state {
            CmlState::Normal => {
                self.match_buf.push(token_id);

                let (delim, delim_len) = self.match_delimiter(&self.match_buf);
                match delim {
                    CmlDelimiter::Call => {
                        let passthrough_end = self.match_buf.len() - delim_len;
                        for i in 0..passthrough_end {
                            events.push(CmlEvent::PassthroughToken { token_id: self.match_buf[i] });
                        }
                        self.match_buf.clear();
                        events.push(CmlEvent::EnterCriticalSection);
                        self.id_tokens.clear();
                        self.body_tokens.clear();
                        self.state = CmlState::InCallId;
                        inferlet::send("[Parser] MATCHED Call → InCallId");
                    }
                    CmlDelimiter::Trap => {
                        let passthrough_end = self.match_buf.len() - delim_len;
                        for i in 0..passthrough_end {
                            events.push(CmlEvent::PassthroughToken { token_id: self.match_buf[i] });
                        }
                        self.match_buf.clear();
                        events.push(CmlEvent::EnterCriticalSection);
                        self.state = CmlState::InTrap;
                        inferlet::send("[Parser] MATCHED Trap → InTrap");
                    }
                    CmlDelimiter::TrapEnd => {
                        // Full [TRAP][END] matched atomically — no InTrap state needed.
                        // This handles the BPE-merged `][` token case.
                        let passthrough_end = self.match_buf.len() - delim_len;
                        for i in 0..passthrough_end {
                            events.push(CmlEvent::PassthroughToken { token_id: self.match_buf[i] });
                        }
                        self.match_buf.clear();
                        events.push(CmlEvent::EnterCriticalSection);
                        events.push(CmlEvent::TrapDetected);
                        events.push(CmlEvent::ExitCriticalSection);
                        inferlet::send("[Parser] MATCHED TrapEnd → TrapDetected (atomic)");
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

                // Try full-sequence match first, then inner (bracket-free) match.
                let (delim, delim_len) = {
                    let (d, l) = self.match_delimiter(&self.match_buf);
                    if d != CmlDelimiter::None { (d, l) } else { self.match_inner_delimiter(&self.match_buf) }
                };

                match delim {
                    CmlDelimiter::Head => {
                        let id_end = self.match_buf.len() - delim_len;
                        self.id_tokens.extend_from_slice(&self.match_buf[..id_end]);
                        self.match_buf.clear();
                        // Strip trailing bracket token that may have been flushed to id_tokens
                        // before the inner HEAD suffix was recognised.
                        Self::strip_trailing_bracket(&mut self.id_tokens, tokenizer);
                        self.state = CmlState::InCallBody;
                        inferlet::send("[Parser] MATCHED Head → InCallBody");
                    }
                    CmlDelimiter::End => {
                        let id_end = self.match_buf.len() - delim_len;
                        self.id_tokens.extend_from_slice(&self.match_buf[..id_end]);
                        self.match_buf.clear();
                        Self::strip_trailing_bracket(&mut self.id_tokens, tokenizer);
                        let id = tokenizer.detokenize(&self.id_tokens).trim().to_string();
                        inferlet::send(&format!("[Parser] EVENT: CallDetected id={} (no body)", id));
                        events.push(CmlEvent::CallDetected { id, code: String::new() });
                        events.push(CmlEvent::ExitCriticalSection);
                        self.id_tokens.clear();
                        self.state = CmlState::Normal;
                    }
                    CmlDelimiter::Call => {
                        // Nested/restarted [CALL] — reset and stay in InCallId
                        inferlet::send("[AsyncLM] Warning: nested [CALL], resetting");
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
                    _ => {} // Trap in InCallId: ignore
                }
            }

            CmlState::InCallBody => {
                self.match_buf.push(token_id);

                let (delim, delim_len) = {
                    let (d, l) = self.match_delimiter(&self.match_buf);
                    if d != CmlDelimiter::None { (d, l) } else { self.match_inner_delimiter(&self.match_buf) }
                };

                match delim {
                    CmlDelimiter::End => {
                        let body_end = self.match_buf.len() - delim_len;
                        self.body_tokens.extend_from_slice(&self.match_buf[..body_end]);
                        self.match_buf.clear();
                        // Strip trailing bracket that preceded the inner END suffix
                        Self::strip_trailing_bracket(&mut self.body_tokens, tokenizer);
                        let id   = tokenizer.detokenize(&self.id_tokens).trim().to_string();
                        let code = tokenizer.detokenize(&self.body_tokens).trim().to_string();
                        inferlet::send(&format!("[Parser] EVENT: CallDetected id={} code={}", id, code));
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
                    _ => {} // Other delimiters in body: accumulate
                }
            }

            CmlState::InTrap => {
                self.match_buf.push(token_id);

                let (delim, _) = {
                    let (d, l) = self.match_delimiter(&self.match_buf);
                    if d != CmlDelimiter::None { (d, l) } else { self.match_inner_delimiter(&self.match_buf) }
                };

                match delim {
                    CmlDelimiter::End => {
                        self.match_buf.clear();
                        inferlet::send("[Parser] EVENT: TrapDetected");
                        events.push(CmlEvent::TrapDetected);
                        events.push(CmlEvent::ExitCriticalSection);
                        self.state = CmlState::Normal;
                    }
                    CmlDelimiter::None => {
                        let is_prefix = self.is_prefix_of_any_delimiter(&self.match_buf)
                            || self.is_prefix_of_any_inner(&self.match_buf);
                        if !is_prefix {
                            self.match_buf.clear(); // Discard non-delimiter tokens in trap
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

    /// Returns true if `token_id` is the first token of a [CALL] or [TRAP] sequence
    /// (either the leading bracket form or the inner-suffix form).
    /// Used to save a checkpoint *before* the delimiter enters the context.
    fn is_entry_delimiter_start(&self, token_id: u32) -> bool {
        // Full-sequence first tokens (e.g. 58 = `[`)
        self.call_ids.iter().any(|seq| seq.first() == Some(&token_id))
            || self.trap_ids.iter().any(|seq| seq.first() == Some(&token_id))
            // Inner-suffix first tokens (e.g. 24622 = `CALL`, 2301 = `TRAP`)
            // These fire when the bracket merged with a preceding newline and the
            // offline tokenizer missed the BPE merge.
            || self.call_inner.first() == Some(&token_id)
            || self.trap_inner.first() == Some(&token_id)
    }

    /// Returns true if the match buffer has pending tokens (potential partial delimiter)
    fn has_pending(&self) -> bool {
        !self.match_buf.is_empty()
    }

    /// Flush any remaining buffered tokens as passthrough events
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
// Section 3: Anti-Hallucination Sampler
// ============================================================================

/// Custom sampler that suppresses `[INTR]` tokens and applies repetition penalty.
///
/// `recent_tokens` is shared with the coordinator via `Rc<RefCell<...>>` — safe in
/// single-threaded WASM.  The coordinator pushes each sampled token after every
/// `decode_step`; this sampler reads it to downweight already-seen tokens.
struct AsyncLmSampler {
    suppressed_ids: HashSet<u32>,
    top_p: f32,
    /// Multiplicative penalty applied to recently seen tokens (>1.0 reduces their prob).
    repetition_penalty: f32,
    /// Sliding window of recently generated token IDs, shared with the coordinator.
    recent_tokens: Rc<RefCell<VecDeque<u32>>>,
}

impl AsyncLmSampler {
    fn new(
        suppressed_ids: HashSet<u32>,
        top_p: f32,
        repetition_penalty: f32,
        recent_tokens: Rc<RefCell<VecDeque<u32>>>,
    ) -> Self {
        AsyncLmSampler {
            suppressed_ids,
            top_p,
            repetition_penalty,
            recent_tokens,
        }
    }
}

impl Sample for AsyncLmSampler {
    fn sample(&self, ids: &[u32], probs: &[f32]) -> u32 {
        let recent = self.recent_tokens.borrow();

        // Filter suppressed, apply repetition penalty, collect candidates
        let mut candidates: Vec<(u32, f32)> = ids
            .iter()
            .zip(probs.iter())
            .filter(|(id, _)| !self.suppressed_ids.contains(id))
            .map(|(id, p)| {
                let penalized = if self.repetition_penalty > 1.0 && recent.contains(id) {
                    // Divide prob by penalty (never below a tiny epsilon)
                    (*p / self.repetition_penalty).max(1e-9)
                } else {
                    *p
                };
                (*id, penalized)
            })
            .collect();

        drop(recent); // release borrow before any further work

        if candidates.is_empty() {
            return ids[0];
        }

        // Sort by (penalized) probability descending
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Apply top-p (nucleus) filtering on the penalized distribution
        let total: f32 = candidates.iter().map(|(_, p)| p).sum();
        let mut cumulative = 0.0f32;
        let mut cutoff = candidates.len();
        for (i, &(_, p)) in candidates.iter().enumerate() {
            cumulative += p / total;
            if cumulative >= self.top_p {
                cutoff = i + 1;
                break;
            }
        }
        candidates.truncate(cutoff);

        // Pick the highest probability token after filtering.
        // Temperature scaling is applied by the Sampler::Custom wrapper before
        // this method is called; without an RNG in WASM we use greedy selection
        // from the filtered nucleus.
        candidates[0].0
    }
}

// ============================================================================
// Section 4: Executor (Message-based)
// ============================================================================

#[derive(Debug, Clone)]
struct FunctionCall {
    id: String,
    code: String,
    dispatch_token_count: usize,
}

#[derive(Debug, Clone)]
struct ExecutionResult {
    id: String,
    output: String,
    is_error: bool,
}

struct MessageExecutor {
    pending: HashMap<String, FunctionCall>,
    completed: VecDeque<ExecutionResult>,
}

impl MessageExecutor {
    fn new() -> Self {
        MessageExecutor {
            pending: HashMap::new(),
            completed: VecDeque::new(),
        }
    }

    fn dispatch(&mut self, call: &FunctionCall) {
        let payload = format!("{}|{}", call.id, call.code);
        inferlet::broadcast("asynclm/call", &payload);
        self.pending.insert(call.id.clone(), call.clone());
    }

    fn poll_completed(&mut self) -> Vec<ExecutionResult> {
        // Non-blocking: we can't truly poll without an async runtime,
        // so completed results are populated by wait_all or external injection.
        self.completed.drain(..).collect()
    }

    async fn wait_all(&mut self) -> Vec<ExecutionResult> {
        let mut results = Vec::new();

        // Drain any already-completed results
        results.extend(self.completed.drain(..));

        // Wait for remaining pending calls
        while !self.pending.is_empty() {
            let msg = inferlet::subscribe("asynclm/result").await;
            let result = Self::parse_result(&msg);
            self.pending.remove(&result.id);
            results.push(result);
        }

        results
    }

    fn parse_result(msg: &str) -> ExecutionResult {
        // Format: "id|is_error|output"
        let mut parts = msg.splitn(3, '|');
        let id = parts.next().unwrap_or("unknown").to_string();
        let is_error = parts.next().unwrap_or("false") == "true";
        let output = parts.next().unwrap_or("").to_string();
        ExecutionResult {
            id,
            output,
            is_error,
        }
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
    injected_ids: HashSet<String>,
    in_critical_section: bool,
}

impl InterruptManager {
    fn new(_registry: &CmlTokenRegistry) -> Self {
        InterruptManager {
            pending: VecDeque::new(),
            injected_ids: HashSet::new(),
            in_critical_section: false,
        }
    }

    fn enqueue(&mut self, result: ExecutionResult) {
        self.pending.push_back(result);
    }

    fn set_critical_section(&mut self, val: bool) {
        self.in_critical_section = val;
    }

    /// Returns formatted CML interrupt strings, empty if in critical section.
    fn drain(&mut self) -> Vec<String> {
        if self.in_critical_section {
            return Vec::new();
        }

        let mut injections = Vec::new();
        while let Some(result) = self.pending.pop_front() {
            let output = if result.is_error {
                format!("ERROR: {}", result.output)
            } else {
                result.output.clone()
            };
            let formatted = format!("[INTR] {} [HEAD] {} [END]", result.id, output);
            self.injected_ids.insert(result.id);
            injections.push(formatted);
        }
        injections
    }
}

// ============================================================================
// Section 6: Trap Handler (Keep / Swap / Recompute)
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

impl Default for TrapConfig {
    fn default() -> Self {
        TrapConfig {
            context_pressure_threshold: 0.75,
            keep_max_pending: 2,
            stale_token_threshold: 64,
            max_context_length: 4096,
        }
    }
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
            let stale_tokens = if let Some(cp_count) = checkpoint_token_count {
                context_len.saturating_sub(cp_count)
            } else {
                0
            };

            if stale_tokens > self.config.stale_token_threshold {
                return TrapStrategy::Recompute;
            }
            return TrapStrategy::Swap;
        }

        TrapStrategy::Keep
    }
}

// ============================================================================
// Section 7: Checkpoint Manager
// ============================================================================

struct Checkpoint {
    ctx: Context,
    /// Total context_len (committed + pending) at save time — used for stale-token heuristic.
    token_count_at_save: usize,
    /// Only the committed token count at save time — used for KV shrink arithmetic.
    committed_len_at_save: usize,
    generated_len_at_save: usize,
}

struct CheckpointManager {
    current: Option<Checkpoint>,
}

impl CheckpointManager {
    fn new() -> Self {
        CheckpointManager { current: None }
    }

    fn save(&mut self, ctx: &Context, generated_len: usize) {
        let committed = ctx.token_ids.len();
        let token_count = committed + ctx.token_ids_pending.len();
        self.current = Some(Checkpoint {
            ctx: ctx.fork(),
            token_count_at_save: token_count,
            committed_len_at_save: committed,
            generated_len_at_save: generated_len,
        });
    }

    fn has_checkpoint(&self) -> bool {
        self.current.is_some()
    }

    fn take(&mut self) -> Option<Checkpoint> {
        self.current.take()
    }

    fn discard(&mut self) {
        self.current = None;
    }

    fn checkpoint_token_count(&self) -> Option<usize> {
        self.current.as_ref().map(|cp| cp.token_count_at_save)
    }
}

// ============================================================================
// Section 8: AsyncLM Coordinator + Main Entry
// ============================================================================

struct AsyncLmCoordinator {
    parser: CmlParser,
    executor: MessageExecutor,
    interrupt_mgr: InterruptManager,
    checkpoint_mgr: CheckpointManager,
    trap_handler: TrapHandler,
    tokenizer: Tokenizer,
    /// Sliding window of recently generated token IDs, shared with the sampler.
    recent_tokens: Rc<RefCell<VecDeque<u32>>>,
    /// Maximum number of tokens to keep in the repetition window.
    repetition_window: usize,
}

impl AsyncLmCoordinator {
    fn new(
        registry: &CmlTokenRegistry,
        tokenizer: Tokenizer,
        trap_config: TrapConfig,
        recent_tokens: Rc<RefCell<VecDeque<u32>>>,
    ) -> Self {
        AsyncLmCoordinator {
            parser: CmlParser::new(registry),
            executor: MessageExecutor::new(),
            interrupt_mgr: InterruptManager::new(registry),
            checkpoint_mgr: CheckpointManager::new(),
            trap_handler: TrapHandler::new(trap_config),
            tokenizer,
            recent_tokens,
            repetition_window: 64,
        }
    }

    async fn generate(
        &mut self,
        ctx: &mut Context,
        sampler: &Sampler,
        eos_tokens: &[Vec<u32>],
        max_tokens: usize,
    ) -> String {
        let mut generated_token_ids: Vec<u32> = Vec::new();
        let stop_cond = max_len(max_tokens).or(ends_with_any(eos_tokens.to_vec()));

        loop {
            let token_id = ctx.decode_step(sampler).await;

            // Update repetition penalty window
            {
                let mut rt = self.recent_tokens.borrow_mut();
                rt.push_back(token_id);
                if rt.len() > self.repetition_window {
                    rt.pop_front();
                }
            }

            // Save checkpoint before the delimiter enters the context so that Swap/Recompute
            // restore to a clean state (no partial CML block in pending).
            if !self.checkpoint_mgr.has_checkpoint()
                && self.parser.is_entry_delimiter_start(token_id)
            {
                self.checkpoint_mgr.save(ctx, generated_token_ids.len());
            }

            ctx.fill_token(token_id);

            let events = self.parser.feed(token_id, &self.tokenizer);

            for event in events {
                match event {
                    CmlEvent::PassthroughToken { token_id } => {
                        generated_token_ids.push(token_id);
                    }
                    CmlEvent::EnterCriticalSection => {
                        self.interrupt_mgr.set_critical_section(true);
                    }
                    CmlEvent::ExitCriticalSection => {
                        self.interrupt_mgr.set_critical_section(false);
                    }
                    CmlEvent::CallDetected { id, code } => {
                        let call = FunctionCall {
                            id,
                            code,
                            dispatch_token_count: ctx.token_ids.len()
                                + ctx.token_ids_pending.len(),
                        };
                        self.executor.dispatch(&call);
                    }
                    CmlEvent::TrapDetected => {
                        self.handle_trap(ctx, &mut generated_token_ids).await;
                    }
                }
            }

            // Poll for completed results
            for result in self.executor.poll_completed() {
                self.interrupt_mgr.enqueue(result);
            }

            // Inject available interrupts (if not in critical section)
            for intr_text in self.interrupt_mgr.drain() {
                ctx.fill(&intr_text);
            }

            if stop_cond.check(&generated_token_ids) {
                break;
            }
        }

        // Flush any remaining buffered tokens from the parser as passthrough
        if self.parser.has_pending() {
            for event in self.parser.flush_pending() {
                if let CmlEvent::PassthroughToken { token_id } = event {
                    generated_token_ids.push(token_id);
                }
            }
        }

        // If generation stopped mid-CML-block, warn about incomplete block
        if self.parser.in_critical_section() {
            inferlet::send("[AsyncLM] Warning: generation stopped mid-CML block, discarding partial block");
        }

        self.tokenizer.detokenize(&generated_token_ids)
    }

    async fn handle_trap(&mut self, ctx: &mut Context, generated_token_ids: &mut Vec<u32>) {
        if !self.executor.has_pending() && self.interrupt_mgr.pending.is_empty() {
            // No pending calls at trap — no-op
            inferlet::send("[AsyncLM] Trap with no pending calls, continuing");
            return;
        }

        let context_len = ctx.token_ids.len() + ctx.token_ids_pending.len();
        let strategy = self.trap_handler.decide(
            self.checkpoint_mgr.has_checkpoint(),
            self.executor.pending_count(),
            context_len,
            self.checkpoint_mgr.checkpoint_token_count(),
        );

        match strategy {
            TrapStrategy::Keep => {
                // Wait for all pending, drain and inject.
                // Do NOT flush — the injected tokens (and the pending [END]) will be
                // processed in one batch by the next decode_step call.
                let results = self.executor.wait_all().await;
                for result in results {
                    self.interrupt_mgr.enqueue(result);
                }
                for intr_text in self.interrupt_mgr.drain() {
                    ctx.fill(&intr_text);
                }
            }
            TrapStrategy::Swap => {
                let results = self.executor.wait_all().await;
                for result in results {
                    self.interrupt_mgr.enqueue(result);
                }

                if let Some(checkpoint) = self.checkpoint_mgr.take() {
                    // Replace context with the checkpoint fork. The fork's token_ids_pending
                    // may have partial-page tokens to recompute; that is fine — the next
                    // decode_step processes them together with the interrupt tokens.
                    *ctx = checkpoint.ctx;
                    generated_token_ids.truncate(checkpoint.generated_len_at_save);

                    for intr_text in self.interrupt_mgr.drain() {
                        ctx.fill(&intr_text);
                    }
                    // No flush — next decode_step processes all pending in one batch.
                }
            }
            TrapStrategy::Recompute => {
                let results = self.executor.wait_all().await;
                for result in results {
                    self.interrupt_mgr.enqueue(result);
                }

                if let Some(checkpoint) = self.checkpoint_mgr.take() {
                    let target = checkpoint.committed_len_at_save;

                    // KV pages only track committed tokens. Shrink based solely on
                    // committed counts — pending tokens ([END]) are not in KV yet.
                    let committed_to_remove =
                        ctx.token_ids.len().saturating_sub(target);
                    if committed_to_remove > 0 {
                        ctx.shrink_kv_pages(committed_to_remove);
                        ctx.token_ids.truncate(target);
                        ctx.position_ids.truncate(target);
                    }

                    // Clear pending: the [END] trap token and its mask snapshot must not
                    // carry over into the Recomputed context.
                    ctx.token_ids_pending.clear();
                    ctx.token_mask_pending.clear();

                    // Truncate token_mask_current to match the committed + 0 pending state.
                    // Its length was committed + pending (N+1); remove everything after target.
                    let mask_len = target + committed_to_remove + 1; // N+1 before truncation
                    ctx.token_mask_current.remove_range(target, mask_len);

                    generated_token_ids.truncate(checkpoint.generated_len_at_save);

                    for intr_text in self.interrupt_mgr.drain() {
                        ctx.fill(&intr_text);
                    }
                    // No flush — next decode_step processes pending interrupt tokens.
                }
            }
        }
    }
}

// ============================================================================
// Main Entry Point
// ============================================================================

#[inferlet::main]
async fn main(mut args: Args) -> Result<String> {
    let prompt: String = args.value_from_str(["-p", "--prompt"])?;
    let max_tokens: usize = args.value_from_str(["-n", "--max-tokens"]).unwrap_or(512);
    let system: String = args
        .value_from_str(["-s", "--system"])
        .unwrap_or_else(|_| {
            "You are an assistant that uses async function calls. You MUST use the exact syntax below.

            To call a function, write: [CALL] function_name [HEAD] function_name(arguments) [END]
            To wait for results, write: [TRAP][END]
            Example conversation:
            User: What's the weather in NYC and London?
            Assistant: Let me check both cities.
            [CALL] weather1 [HEAD] get_weather(\"New York\") [END]
            [CALL] weather2 [HEAD] get_weather(\"London\") [END]
            [TRAP][END]

            Always use this exact format. Never omit the delimiters. Assume you have a function called get_weather()."
                .to_string()
        });
    let temperature: f32 = args.value_from_str(["-t", "--temperature"]).unwrap_or(0.6);
    let top_p: f32 = args.value_from_str("--top-p").unwrap_or(0.95);
    let repetition_penalty: f32 = args
        .value_from_str("--repetition-penalty")
        .unwrap_or(1.3);
    let max_context: usize = args
        .value_from_str("--max-context")
        .unwrap_or(4096);

    let model = inferlet::get_auto_model();
    let tokenizer = model.get_tokenizer();

    // Build CML token registry
    let registry = CmlTokenRegistry::new(&tokenizer);
    inferlet::send(&format!("[AsyncLM] CML tokens: call={:?} end={:?} head={:?} trap={:?} intr={:?}",
        registry.call_ids[0], registry.end_ids[0], registry.head_ids[0],
        registry.trap_ids[0], registry.intr_ids[0]));
    inferlet::send(&format!("[AsyncLM] repetition_penalty={} temperature={} top_p={}",
        repetition_penalty, temperature, top_p));

    // Shared token history for repetition penalty (Rc: safe in single-threaded WASM)
    let recent_tokens: Rc<RefCell<VecDeque<u32>>> = Rc::new(RefCell::new(VecDeque::new()));

    // Create custom sampler that suppresses [INTR] and applies repetition penalty
    let async_sampler = AsyncLmSampler::new(
        registry.suppressed_ids.clone(),
        top_p,
        repetition_penalty,
        Rc::clone(&recent_tokens),
    );
    let sampler = Sampler::Custom {
        temperature,
        sampler: Box::new(async_sampler),
    };

    // Build trap config
    let trap_config = TrapConfig {
        max_context_length: max_context,
        ..TrapConfig::default()
    };

    // Create coordinator
    let mut coordinator = AsyncLmCoordinator::new(&registry, tokenizer, trap_config, recent_tokens);

    // Prepare context
    let mut ctx = model.create_context();
    ctx.fill_system(&system);
    ctx.fill_user(&prompt);

    // Get EOS tokens
    let eos_tokens = model.eos_tokens();

    // Generate
    let result = coordinator
        .generate(&mut ctx, &sampler, &eos_tokens, max_tokens)
        .await;

    Ok(result)
}
