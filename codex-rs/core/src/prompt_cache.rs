use codex_api::PromptCacheBreakpoint;
use codex_api::PromptCacheMode;
use codex_api::PromptCacheOptions;
use codex_api::PromptCacheTtl;
use codex_protocol::models::ContentItem;
use codex_protocol::models::ResponseItem;
use serde::Serialize;
use uuid::Uuid;

pub(crate) const STABLE_PREFIX_CACHE_PERCENT_ENV: &str = "CODEX_GPT56_STABLE_PREFIX_CACHE_PERCENT";
pub(crate) const CACHE_COHORT_METADATA_KEY: &str = "gen6598_prompt_cache_cohort";
pub(crate) const CACHE_KEY_VERSION_METADATA_KEY: &str = "gen6598_prompt_cache_key_version";
pub(crate) const CACHE_KEY_VERSION: &str = "v1";

const CACHE_KEY_NAMESPACE: Uuid = Uuid::from_u128(0x96e9507c_226b_4f77_a6df_f43b6f617bd6);
const CACHE_COHORT_NAMESPACE: Uuid = Uuid::from_u128(0xc8e1587f_e9bf_4d82_83a5_acef23bfe1ae);

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum StablePrefixCachePlan {
    Disabled,
    Control,
    Treatment {
        prompt_cache_key: String,
        prompt_cache_options: PromptCacheOptions,
        prompt_cache_breakpoints: Vec<PromptCacheBreakpoint>,
    },
}

#[derive(Serialize)]
struct StablePrefixKeyMaterial<'a> {
    version: &'static str,
    provider_name: &'a str,
    provider_base_url: Option<&'a str>,
    model: &'a str,
    instructions: &'a str,
    tools: &'a serde_json::Value,
    stable_input_prefix: &'a [ResponseItem],
    repository_scope: &'a str,
    repository_instructions: &'a str,
    installation_id: &'a str,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn stable_prefix_cache_plan(
    rollout_percent: u8,
    provider_name: &str,
    provider_base_url: Option<&str>,
    model: &str,
    instructions: &str,
    tools: &serde_json::Value,
    input: &[ResponseItem],
    repository_scope: &str,
    installation_id: &str,
    session_id: &str,
) -> StablePrefixCachePlan {
    if rollout_percent == 0
        || !is_supported_provider(provider_name, provider_base_url)
        || !is_gpt_5_6_model(model)
    {
        return StablePrefixCachePlan::Disabled;
    }

    let Some((prompt_cache_breakpoints, stable_input_prefix, repository_instructions)) =
        find_repository_instructions(input)
    else {
        return StablePrefixCachePlan::Disabled;
    };

    // Git enrichment is intentionally asynchronous and can be absent on the
    // first model request. The exact stable repository instructions remain in
    // the key below, so an empty remote scope cannot mix different prompt
    // prefixes; installation_id supplies the outer isolation boundary.

    if !is_treatment_session(session_id, rollout_percent) {
        return StablePrefixCachePlan::Control;
    }

    let key_material = StablePrefixKeyMaterial {
        version: CACHE_KEY_VERSION,
        provider_name,
        provider_base_url,
        model,
        instructions,
        tools,
        stable_input_prefix,
        repository_scope,
        repository_instructions,
        installation_id,
    };
    let Ok(encoded) = serde_json::to_vec(&key_material) else {
        return StablePrefixCachePlan::Disabled;
    };
    let digest = Uuid::new_v5(&CACHE_KEY_NAMESPACE, &encoded);

    StablePrefixCachePlan::Treatment {
        prompt_cache_key: format!("codex-gpt56-{}", digest.simple()),
        prompt_cache_options: PromptCacheOptions {
            mode: PromptCacheMode::Explicit,
            ttl: PromptCacheTtl::ThirtyMinutes,
        },
        prompt_cache_breakpoints,
    }
}

fn is_supported_provider(provider_name: &str, provider_base_url: Option<&str>) -> bool {
    let provider_name = provider_name.to_ascii_lowercase();
    let provider_base_url = provider_base_url.unwrap_or_default().to_ascii_lowercase();
    [provider_name.as_str(), provider_base_url.as_str()]
        .iter()
        .any(|value| value.contains("bedrock") || value.contains("mantle"))
}

fn is_gpt_5_6_model(model: &str) -> bool {
    model.to_ascii_lowercase().contains("gpt-5.6")
}

fn is_treatment_session(session_id: &str, rollout_percent: u8) -> bool {
    if rollout_percent >= 100 {
        return true;
    }
    let cohort = Uuid::new_v5(&CACHE_COHORT_NAMESPACE, session_id.as_bytes());
    cohort.as_u128() % 100 < u128::from(rollout_percent)
}

fn find_repository_instructions(
    input: &[ResponseItem],
) -> Option<(Vec<PromptCacheBreakpoint>, &[ResponseItem], &str)> {
    input.iter().enumerate().find_map(|(input_index, item)| {
        let ResponseItem::Message { content, .. } = item else {
            return None;
        };
        content
            .iter()
            .enumerate()
            .find_map(|(content_index, content)| {
                let ContentItem::InputText { text } = content else {
                    return None;
                };
                let (_, stable_instructions) = text.split_once('\n')?;
                if !text.starts_with("# AGENTS.md instructions") {
                    return None;
                }
                let mut breakpoints = input[..input_index]
                    .iter()
                    .enumerate()
                    .rev()
                    .find_map(|(prefix_input_index, prefix_item)| {
                        let ResponseItem::Message {
                            content: prefix_content,
                            ..
                        } = prefix_item
                        else {
                            return None;
                        };
                        prefix_content.last().map(|_| PromptCacheBreakpoint {
                            input_index: prefix_input_index,
                            content_index: prefix_content.len() - 1,
                        })
                    })
                    .into_iter()
                    .collect::<Vec<_>>();
                breakpoints.push(PromptCacheBreakpoint {
                    input_index,
                    content_index,
                });
                Some((breakpoints, &input[..input_index], stable_instructions))
            })
    })
}

#[cfg(test)]
#[path = "prompt_cache_tests.rs"]
mod tests;
