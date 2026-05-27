//! Simple text completion inferlet.
//!
//! Demonstrates chat-style generation with the explicit per-step Generator
//! loop and `chat::Decoder`.

use inferlet::{Context, Result, chat, model::Model, runtime, sample::Sampler};
use serde::Deserialize;

#[derive(Deserialize)]
struct Input {
    /// The user prompt to complete.
    prompt: String,

    /// Maximum number of tokens to generate.
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,

    /// System message for the assistant.
    #[serde(default = "default_system")]
    system: String,

    /// Sampling temperature.
    #[serde(default = "default_temperature")]
    temperature: f32,

    /// Top-p (nucleus) sampling threshold.
    #[serde(default = "default_top_p")]
    top_p: f32,
}

fn default_max_tokens() -> usize {
    1024
}
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
     is 07:30, just starting the workday, so this is a good window.".into()
}
fn default_temperature() -> f32 {
    0.6
}
fn default_top_p() -> f32 {
    0.95
}

#[inferlet::main]
async fn main(input: Input) -> Result<String> {
    let models = runtime::models();
    let model_name = models.first().ok_or("No models available")?;
    let model = Model::load(model_name)?;

    let mut ctx = Context::new(&model)?;
    ctx.system(&input.system).user(&input.prompt).cue();

    let mut chat = chat::Decoder::new(&model);
    let mut text = String::new();

    let mut g = ctx
        .generate(Sampler::TopP {
            temperature: input.temperature,
            p: input.top_p,
        })
        .max_tokens(input.max_tokens)
        .stop(&chat::stop_tokens(&model));

    while let Some(step) = g.next()? {
        let out = step.execute().await?;
        if out.tokens.is_empty() {
            continue;
        }

        match chat.feed(&out.tokens)? {
            chat::Event::Delta(s) => {
                print!("{}", s);
                text.push_str(&s);
            }
            chat::Event::Done(s) => {
                text = s;
                break;
            }
            _ => {}
        }
    }

    Ok(text)
}
