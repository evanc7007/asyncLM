//! Simple text completion inferlet.
//! This example demonstrates basic text generation with a system prompt and user message.

use inferlet::stop_condition::{StopCondition, ends_with_any, max_len};
use inferlet::{Args, Result, Sampler};

#[inferlet::main]
async fn main(mut args: Args) -> Result<String> {
    // Parse arguments
    let prompt: String = args.value_from_str(["-p", "--prompt"])?;
    let max_tokens: usize = args.value_from_str(["-n", "--max-tokens"]).unwrap_or(256);
    let system: String = args
        .value_from_str(["-s", "--system"])
        .unwrap_or_else(|_| "You are a helpful, respectful and honest assistant.".to_string());
    let temperature: f32 = args.value_from_str(["-t", "--temperature"]).unwrap_or(0.6);
    let top_p: f32 = args.value_from_str("--top-p").unwrap_or(0.95);

    // Get model and create context
    let model = inferlet::get_auto_model();
    let mut ctx = model.create_context();

    // Fill context with messages
    ctx.fill_system(&system);
    ctx.fill_user(&prompt);

    // Generate response
    let sampler = Sampler::top_p(temperature, top_p);
    let stop_cond = max_len(max_tokens).or(ends_with_any(model.eos_tokens()));

    let result = ctx.generate(sampler, stop_cond).await;

    Ok(result)
}
