//! AsyncLM Inferlet — Async function calling with parallel tool execution
//!
//! Implements the AsyncLM paper (https://arxiv.org/pdf/2412.07017): the LLM continues
//! generating while tool calls execute in parallel, with results injected back via interrupts.

use inferlet::context::Context;
use inferlet::sampler::Sample;
use inferlet::stop_condition::{ends_with_any, max_len, StopCondition};
use inferlet::{Args, Result, Sampler, Tokenizer};
use std::collections::{HashMap, HashSet, VecDeque};

// ============================================================================
// Section 1: CML Token Registry
// ============================================================================

/// Resolves CML special tokens to their token IDs.
///
/// CML tokens may be single special tokens (if registered in the vocabulary)
/// or multi-token sequences (if the model doesn't have them as specials).
struct CmlTokenRegistry {
    call_ids: Vec<u32>,
    end_ids: Vec<u32>,
    intr_ids: Vec<u32>,
    trap_ids: Vec<u32>,
    head_ids: Vec<u32>,

    call_str: String,
    end_str: String,
    intr_str: String,
    trap_str: String,
    head_str: String,

    suppressed_ids: HashSet<u32>,
}

impl CmlTokenRegistry {
    fn new(tokenizer: &Tokenizer) -> Self {
        let call_str = "<|call|>".to_string();
        let end_str = "<|end|>".to_string();
        let intr_str = "<|intr|>".to_string();
        let trap_str = "<|trap|>".to_string();
        let head_str = "<|head|>".to_string();

        // Build a lookup from byte representation to token ID for special tokens
        let (special_ids, special_bytes) = tokenizer.get_special_tokens();
        let special_map: HashMap<Vec<u8>, u32> = special_bytes
            .into_iter()
            .zip(special_ids.into_iter())
            .collect();

        let call_ids = Self::resolve_token(tokenizer, &special_map, &call_str);
        let end_ids = Self::resolve_token(tokenizer, &special_map, &end_str);
        let intr_ids = Self::resolve_token(tokenizer, &special_map, &intr_str);
        let trap_ids = Self::resolve_token(tokenizer, &special_map, &trap_str);
        let head_ids = Self::resolve_token(tokenizer, &special_map, &head_str);

        // Suppress all token IDs that make up <|intr|>
        let suppressed_ids: HashSet<u32> = intr_ids.iter().copied().collect();

        CmlTokenRegistry {
            call_ids,
            end_ids,
            intr_ids,
            trap_ids,
            head_ids,
            call_str,
            end_str,
            intr_str,
            trap_str,
            head_str,
            suppressed_ids,
        }
    }

    /// Resolves a CML token string to its token ID(s).
    /// First checks if it exists as a single special token, otherwise falls back to tokenize().
    fn resolve_token(
        tokenizer: &Tokenizer,
        special_map: &HashMap<Vec<u8>, u32>,
        text: &str,
    ) -> Vec<u32> {
        // Try special token lookup first
        if let Some(&id) = special_map.get(text.as_bytes()) {
            return vec![id];
        }
        // Fall back to regular tokenization
        let ids = tokenizer.tokenize(text);
        assert!(
            !ids.is_empty(),
            "CML token '{}' tokenized to empty sequence",
            text
        );
        ids
    }

    /// Returns true if all CML tokens resolve to single IDs (ideal fast path).
    fn all_single_token(&self) -> bool {
        self.call_ids.len() == 1
            && self.end_ids.len() == 1
            && self.intr_ids.len() == 1
            && self.trap_ids.len() == 1
            && self.head_ids.len() == 1
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
    Head,
    None,
}

struct CmlParser {
    call_ids: Vec<u32>,
    end_ids: Vec<u32>,
    trap_ids: Vec<u32>,
    head_ids: Vec<u32>,
    state: CmlState,
    id_tokens: Vec<u32>,
    body_tokens: Vec<u32>,
    /// Buffer for matching multi-token CML delimiters in Normal state
    match_buf: Vec<u32>,
}

impl CmlParser {
    fn new(registry: &CmlTokenRegistry) -> Self {
        CmlParser {
            call_ids: registry.call_ids.clone(),
            end_ids: registry.end_ids.clone(),
            trap_ids: registry.trap_ids.clone(),
            head_ids: registry.head_ids.clone(),
            state: CmlState::Normal,
            id_tokens: Vec::new(),
            body_tokens: Vec::new(),
            match_buf: Vec::new(),
        }
    }

    /// Check if `buf` ends with `pattern`
    fn buf_ends_with(buf: &[u32], pattern: &[u32]) -> bool {
        buf.len() >= pattern.len() && buf[buf.len() - pattern.len()..] == *pattern
    }

    /// Try to match a delimiter at the tail of `buf`. Returns delimiter and its length.
    fn match_delimiter(&self, buf: &[u32]) -> (CmlDelimiter, usize) {
        // Check each delimiter — order matters (check longer patterns first if same prefix)
        if Self::buf_ends_with(buf, &self.call_ids) {
            return (CmlDelimiter::Call, self.call_ids.len());
        }
        if Self::buf_ends_with(buf, &self.end_ids) {
            return (CmlDelimiter::End, self.end_ids.len());
        }
        if Self::buf_ends_with(buf, &self.trap_ids) {
            return (CmlDelimiter::Trap, self.trap_ids.len());
        }
        if Self::buf_ends_with(buf, &self.head_ids) {
            return (CmlDelimiter::Head, self.head_ids.len());
        }
        (CmlDelimiter::None, 0)
    }

    /// Could `buf` be the start of any CML delimiter?
    fn is_prefix_of_any_delimiter(&self, buf: &[u32]) -> bool {
        for pattern in [&self.call_ids, &self.end_ids, &self.trap_ids, &self.head_ids] {
            if buf.len() <= pattern.len() && pattern[..buf.len()] == *buf {
                return true;
            }
        }
        false
    }

    fn feed(&mut self, token_id: u32, tokenizer: &Tokenizer) -> Vec<CmlEvent> {
        let mut events = Vec::new();

        match self.state {
            CmlState::Normal => {
                self.match_buf.push(token_id);

                let (delim, delim_len) = self.match_delimiter(&self.match_buf);
                match delim {
                    CmlDelimiter::Call => {
                        // Flush any tokens before the delimiter as passthrough
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
                    CmlDelimiter::None => {
                        // Check if buffer could still be a prefix of a delimiter
                        if !self.is_prefix_of_any_delimiter(&self.match_buf) {
                            // Flush the first token as passthrough, keep the rest
                            // (they might form a prefix with future tokens)
                            let first = self.match_buf.remove(0);
                            events.push(CmlEvent::PassthroughToken { token_id: first });
                            // Keep draining non-prefix tokens
                            while !self.match_buf.is_empty()
                                && !self.is_prefix_of_any_delimiter(&self.match_buf)
                            {
                                let t = self.match_buf.remove(0);
                                events.push(CmlEvent::PassthroughToken { token_id: t });
                            }
                        }
                    }
                    _ => {
                        // End/Head in Normal state: treat as passthrough
                        for &t in &self.match_buf {
                            events.push(CmlEvent::PassthroughToken { token_id: t });
                        }
                        self.match_buf.clear();
                    }
                }
            }
            CmlState::InCallId => {
                self.match_buf.push(token_id);
                let (delim, delim_len) = self.match_delimiter(&self.match_buf);
                match delim {
                    CmlDelimiter::Head => {
                        // Tokens before <|head|> are part of the call id
                        let id_end = self.match_buf.len() - delim_len;
                        self.id_tokens.extend_from_slice(&self.match_buf[..id_end]);
                        self.match_buf.clear();
                        self.state = CmlState::InCallBody;
                    }
                    CmlDelimiter::End => {
                        // Call with no body: <|call|> id <|end|>
                        let id_end = self.match_buf.len() - delim_len;
                        self.id_tokens.extend_from_slice(&self.match_buf[..id_end]);
                        self.match_buf.clear();
                        let id = tokenizer.detokenize(&self.id_tokens);
                        let id = id.trim().to_string();
                        events.push(CmlEvent::CallDetected {
                            id,
                            code: String::new(),
                        });
                        events.push(CmlEvent::ExitCriticalSection);
                        self.id_tokens.clear();
                        self.state = CmlState::Normal;
                    }
                    CmlDelimiter::Call => {
                        // Nested <|call|> — reset, warn
                        inferlet::send(
                            "[AsyncLM] Warning: nested <|call|> detected, resetting parser",
                        );
                        self.match_buf.clear();
                        self.id_tokens.clear();
                        self.body_tokens.clear();
                    }
                    CmlDelimiter::None => {
                        // If not a prefix of any delimiter, flush to id_tokens
                        if !self.is_prefix_of_any_delimiter(&self.match_buf) {
                            self.id_tokens.extend_from_slice(&self.match_buf);
                            self.match_buf.clear();
                        }
                    }
                    _ => {} // Trap in InCallId: ignore
                }
            }
            CmlState::InCallBody => {
                self.match_buf.push(token_id);
                let (delim, delim_len) = self.match_delimiter(&self.match_buf);
                match delim {
                    CmlDelimiter::End => {
                        let body_end = self.match_buf.len() - delim_len;
                        self.body_tokens.extend_from_slice(&self.match_buf[..body_end]);
                        self.match_buf.clear();
                        let id = tokenizer.detokenize(&self.id_tokens);
                        let id = id.trim().to_string();
                        let code = tokenizer.detokenize(&self.body_tokens);
                        let code = code.trim().to_string();
                        events.push(CmlEvent::CallDetected { id, code });
                        events.push(CmlEvent::ExitCriticalSection);
                        self.id_tokens.clear();
                        self.body_tokens.clear();
                        self.state = CmlState::Normal;
                    }
                    CmlDelimiter::None => {
                        if !self.is_prefix_of_any_delimiter(&self.match_buf) {
                            self.body_tokens.extend_from_slice(&self.match_buf);
                            self.match_buf.clear();
                        }
                    }
                    _ => {} // Other delimiters in body: accumulate
                }
            }
            CmlState::InTrap => {
                self.match_buf.push(token_id);
                let (delim, _) = self.match_delimiter(&self.match_buf);
                match delim {
                    CmlDelimiter::End => {
                        self.match_buf.clear();
                        events.push(CmlEvent::TrapDetected);
                        events.push(CmlEvent::ExitCriticalSection);
                        self.state = CmlState::Normal;
                    }
                    CmlDelimiter::None => {
                        if !self.is_prefix_of_any_delimiter(&self.match_buf) {
                            self.match_buf.clear(); // Discard non-delimiter tokens in trap
                        }
                    }
                    _ => {} // Other delimiters in trap: ignore
                }
            }
        }

        events
    }

    fn in_critical_section(&self) -> bool {
        self.state != CmlState::Normal
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

/// Custom sampler that suppresses `<|intr|>` tokens so the LLM can never
/// generate them — only the system can inject interrupts.
struct AsyncLmSampler {
    suppressed_ids: HashSet<u32>,
    top_p: f32,
}

impl AsyncLmSampler {
    fn new(suppressed_ids: HashSet<u32>, top_p: f32) -> Self {
        AsyncLmSampler {
            suppressed_ids,
            top_p,
        }
    }
}

impl Sample for AsyncLmSampler {
    fn sample(&self, ids: &[u32], probs: &[f32]) -> u32 {
        // Filter out suppressed IDs and collect valid (id, prob) pairs
        let mut candidates: Vec<(u32, f32)> = ids
            .iter()
            .zip(probs.iter())
            .filter(|(id, _)| !self.suppressed_ids.contains(id))
            .map(|(id, p)| (*id, *p))
            .collect();

        if candidates.is_empty() {
            // Fallback: if all tokens are suppressed, return the most probable original
            return ids[0];
        }

        // Sort by probability descending
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Apply top-p (nucleus) filtering
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

        // Renormalize and sample
        let norm: f32 = candidates.iter().map(|(_, p)| p).sum();
        if norm <= 0.0 {
            return candidates[0].0;
        }

        // Simple weighted random using a deterministic approach:
        // pick proportionally — use the probability distribution directly
        // Since we don't have a RNG in WASM, we pick the highest probability token
        // after top-p filtering. The temperature is handled by the Sampler::Custom wrapper.
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
    registry: CmlTokenRegistryRef,
}

/// Lightweight reference to the CML string forms needed for formatting.
struct CmlTokenRegistryRef {
    intr_str: String,
    head_str: String,
    end_str: String,
}

impl InterruptManager {
    fn new(registry: &CmlTokenRegistry) -> Self {
        InterruptManager {
            pending: VecDeque::new(),
            injected_ids: HashSet::new(),
            in_critical_section: false,
            registry: CmlTokenRegistryRef {
                intr_str: registry.intr_str.clone(),
                head_str: registry.head_str.clone(),
                end_str: registry.end_str.clone(),
            },
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
            let formatted = format!(
                "{} {} {} {} {}",
                self.registry.intr_str,
                result.id,
                self.registry.head_str,
                output,
                self.registry.end_str
            );
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
    token_count_at_save: usize,
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
        let token_count = ctx.token_ids.len() + ctx.token_ids_pending.len();
        self.current = Some(Checkpoint {
            ctx: ctx.fork(),
            token_count_at_save: token_count,
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
}

impl AsyncLmCoordinator {
    fn new(registry: &CmlTokenRegistry, tokenizer: Tokenizer, trap_config: TrapConfig) -> Self {
        AsyncLmCoordinator {
            parser: CmlParser::new(registry),
            executor: MessageExecutor::new(),
            interrupt_mgr: InterruptManager::new(registry),
            checkpoint_mgr: CheckpointManager::new(),
            trap_handler: TrapHandler::new(trap_config),
            tokenizer,
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

            // Always fill the sampled token into the context — the parser buffers
            // tokens internally and emits them as events once classification is done.
            ctx.fill_token(token_id);

            let events = self.parser.feed(token_id, &self.tokenizer);

            for event in events {
                match event {
                    CmlEvent::PassthroughToken { token_id } => {
                        generated_token_ids.push(token_id);
                    }
                    CmlEvent::EnterCriticalSection => {
                        self.interrupt_mgr.set_critical_section(true);
                        // Save checkpoint before first call block if none exists
                        if !self.checkpoint_mgr.has_checkpoint() {
                            self.checkpoint_mgr
                                .save(ctx, generated_token_ids.len());
                        }
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
                // Wait for all pending, drain and inject
                let results = self.executor.wait_all().await;
                for result in results {
                    self.interrupt_mgr.enqueue(result);
                }
                // Critical section is already exited by TrapDetected event
                for intr_text in self.interrupt_mgr.drain() {
                    ctx.fill(&intr_text);
                }
                ctx.flush().await;
            }
            TrapStrategy::Swap => {
                let results = self.executor.wait_all().await;
                for result in results {
                    self.interrupt_mgr.enqueue(result);
                }

                if let Some(checkpoint) = self.checkpoint_mgr.take() {
                    *ctx = checkpoint.ctx;
                    generated_token_ids.truncate(checkpoint.generated_len_at_save);

                    for intr_text in self.interrupt_mgr.drain() {
                        ctx.fill(&intr_text);
                    }
                    ctx.flush().await;
                }
            }
            TrapStrategy::Recompute => {
                let results = self.executor.wait_all().await;
                for result in results {
                    self.interrupt_mgr.enqueue(result);
                }

                if let Some(checkpoint) = self.checkpoint_mgr.take() {
                    let tokens_to_remove =
                        context_len.saturating_sub(checkpoint.token_count_at_save);

                    if tokens_to_remove > 0 {
                        ctx.shrink_kv_pages(tokens_to_remove);
                        let new_len = ctx.token_ids.len().saturating_sub(tokens_to_remove);
                        ctx.token_ids.truncate(new_len);
                        ctx.position_ids.truncate(new_len);
                        ctx.token_mask_current
                            .remove_range(new_len, new_len + tokens_to_remove);
                    }

                    generated_token_ids.truncate(checkpoint.generated_len_at_save);

                    for intr_text in self.interrupt_mgr.drain() {
                        ctx.fill(&intr_text);
                    }
                    ctx.flush().await;
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
            "You are a helpful assistant with async function calling. \
             Use <|call|> id <|head|> code <|end|> blocks for functions \
             and <|trap|><|end|> to wait for all pending results."
                .to_string()
        });
    let temperature: f32 = args.value_from_str(["-t", "--temperature"]).unwrap_or(0.6);
    let top_p: f32 = args.value_from_str("--top-p").unwrap_or(0.95);
    let max_context: usize = args
        .value_from_str("--max-context")
        .unwrap_or(4096);

    let model = inferlet::get_auto_model();
    let tokenizer = model.get_tokenizer();

    // Build CML token registry
    let registry = CmlTokenRegistry::new(&tokenizer);

    // Create custom sampler that suppresses <|intr|>
    let async_sampler = AsyncLmSampler::new(registry.suppressed_ids.clone(), top_p);
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
    let mut coordinator = AsyncLmCoordinator::new(&registry, tokenizer, trap_config);

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
