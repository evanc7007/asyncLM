# Runtime API Overview (Rust / WASM)

This document describes the runtime-facing APIs exposed to inferlets, along with host-side runtime implementation modules.

---

## Runtime Interface Functions

### Functions
- `get_version() -> String`  
  Get the runtime version string.

- `get_instance_id() -> String`  
  Get the unique ID for the current runtime instance.

- `get_arguments() -> Vec<String>`  
  Retrieve CLI arguments passed to the inferlet.

- `set_return(value: String)`  
  Set the inferlet’s return value.

- `get_model(name: &str) -> Option<Model>`  
  Retrieve a model by name.

- `get_all_models() -> Vec<String>`  
  List all available model names.

- `get_all_models_with_traits(traits: &[&str]) -> Vec<String>`  
  List models matching the given traits.

- `debug_query(query: &str) -> DebugQueryResult`  
  Execute a runtime debug command.

---

## Common Interface

### Resources

#### Blob
Binary data container.

**Methods**
- `Blob::new(data: &[u8])`
- `read(offset: u64, n: u64) -> Vec<u8>`
- `size() -> u64`

---

#### Model
Represents a loaded model instance.

**Methods**
- `get_name()`
- `get_traits()`
- `get_description()`
- `get_prompt_template()`
- `get_stop_tokens()`
- `get_service_id()`
- `get_kv_page_size()`
- `create_queue() -> Queue`

---

#### Queue
Execution queue for model commands.

**Methods**
- `get_service_id()`
- `synchronize()`
- `set_priority(priority)`
- `debug_query(query: &str)`

---

### Resource Management Functions
- `allocate_resources()`
- `deallocate_resources()`
- `export_resources()`
- `import_resources()`
- `get_all_exported_resources()`
- `release_exported_resources()`

---

## Forward Pass Interface

### Functions
- `create_forward_pass(queue: &Queue) -> ForwardPass`

---

### ForwardPass

**Methods**
- `attention_mask(mask: &[Vec<u32>])`
- `kv_cache(kv_page_ptrs: &[u32], last_kv_page_len: u32)`
- `input_embeddings(emb_ptrs: &[u32], positions: &[u32])`
- `input_tokens(tokens: &[u32], positions: &[u32])`
- `output_embeddings(emb_ptrs: &[u32], indices: &[u32])`
- `output_distributions(indices: &[u32], temperature: f32, top_k: Option<u32>)`
- `output_tokens(indices: &[u32], temperature: f32)`
- `output_tokens_top_k(indices: &[u32], temperature: f32, top_k: u32)`
- `output_tokens_top_p(indices: &[u32], temperature: f32, top_p: f32)`
- `output_tokens_min_p(indices: &[u32], temperature: f32, min_p: f32)`
- `output_tokens_top_k_top_p(
    indices: &[u32],
    temperature: f32,
    top_k: u32,
    top_p: f32
  )`
- `execute() -> Option<ForwardPassResult>`

---

## Tokenization Interface

### Functions
- `get_tokenizer(model: &Model) -> Tokenizer`

---

### Tokenizer

**Methods**
- `tokenize(text: &str) -> Vec<u32>`
- `detokenize(tokens: &[u32]) -> String`
- `get_vocabs()`
- `get_split_regex()`
- `get_special_tokens()`

---

## Key-Value Store Interface

### Functions
- `store_get(key: &str) -> Option<String>`
- `store_set(key: &str, value: &str)`
- `store_delete(key: &str)`
- `store_exists(key: &str) -> bool`
- `store_list_keys() -> Vec<String>`

---

## Messaging Interface

### Functions
- `send(message: &str)`
- `receive() -> ReceiveResult`
- `send_blob(blob: Blob)`
- `receive_blob() -> BlobResult`
- `broadcast(topic: &str, message: &str)`
- `subscribe(topic: &str) -> Subscription`

---

## High-Level Rust Wrappers

These provide ergonomic Rust abstractions over the raw runtime APIs.

- **Context Management** (`context.rs`)
- **Forward Pass Wrapper** (`forward.rs`)
- **Chat API** (`chat.rs`)
- **Sampler** (`sampler.rs`)
- **Adapter Management** (`adapter.rs`)
- **ZO Evolution** (`zo.rs`)
- **Image Processing** (`image.rs`)

---

## Runtime Implementation APIs (Host Side)

Located under `src/`, these implement the host-side logic invoked by inferlets.

---

### Main Runtime (`runtime.rs`)
- `start_service(engine: Engine)` — Start the runtime daemon
- Instance management
- Component loading and linking

---

### API Implementation (`api.rs`)
- Contains the `bindgen!` macro
- Generates host implementations for all WIT interfaces

---

### Service Layer (`service.rs`)
- `Service` trait and implementations
- Command dispatching
- Resource management

---

### Model Management (`model.rs`)
- Model loading and caching
- Model trait checking
- Forward pass execution

---

### Instance Management (`instance.rs`)
- WASM instance lifecycle
- Resource allocation/deallocation
- Output streaming

---

### Authentication (`auth.rs`)
- User authentication
- Permission checking

---

### Messaging (`messaging.rs`)
- Inter-instance communication
- Topic-based pub/sub

---

### Telemetry (`telemetry.rs`)
- Performance monitoring
- Logging and metrics
