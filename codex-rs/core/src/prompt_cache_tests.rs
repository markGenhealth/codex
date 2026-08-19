use super::*;
use pretty_assertions::assert_eq;
use serde_json::json;

fn repository_input(text: &str) -> Vec<ResponseItem> {
    vec![ResponseItem::Message {
        id: None,
        role: "user".to_string(),
        content: vec![ContentItem::InputText {
            text: text.to_string(),
        }],
        phase: None,
        internal_chat_message_metadata_passthrough: None,
    }]
}

fn treatment_plan(
    model: &str,
    instructions: &str,
    tools: &serde_json::Value,
    repository_instructions: &str,
    installation_id: &str,
) -> StablePrefixCachePlan {
    stable_prefix_cache_plan(
        /*rollout_percent*/ 100,
        "Amazon Bedrock Mantle",
        Some("https://bedrock-mantle.example/v1"),
        model,
        instructions,
        tools,
        &repository_input(repository_instructions),
        "origin=https://github.com/genhealth/etl.git",
        installation_id,
        "session-a",
    )
}

fn key(plan: StablePrefixCachePlan) -> String {
    let StablePrefixCachePlan::Treatment {
        prompt_cache_key, ..
    } = plan
    else {
        panic!("expected treatment plan");
    };
    prompt_cache_key
}

#[test]
fn treatment_key_is_stable_across_sessions_and_scoped_to_stable_prefix() {
    let tools = json!([{"type": "function", "name": "exec"}]);
    let first = stable_prefix_cache_plan(
        /*rollout_percent*/ 100,
        "Amazon Bedrock Mantle",
        Some("https://bedrock-mantle.example/v1"),
        "openai.gpt-5.6-terra",
        "system instructions",
        &tools,
        &repository_input("# AGENTS.md instructions for /review-worktrees/run-a\nrepo rules"),
        "origin=https://github.com/genhealth/etl.git",
        "installation-a",
        "session-a",
    );
    let second = stable_prefix_cache_plan(
        /*rollout_percent*/ 100,
        "Amazon Bedrock Mantle",
        Some("https://bedrock-mantle.example/v1"),
        "openai.gpt-5.6-terra",
        "system instructions",
        &tools,
        &repository_input("# AGENTS.md instructions for /review-worktrees/run-b\nrepo rules"),
        "origin=https://github.com/genhealth/etl.git",
        "installation-a",
        "session-b",
    );

    assert_eq!(key(first), key(second));
}

#[test]
fn key_changes_for_each_security_or_prefix_boundary() {
    let tools = json!([{"type": "function", "name": "exec"}]);
    let baseline = key(treatment_plan(
        "openai.gpt-5.6-terra",
        "system instructions",
        &tools,
        "# AGENTS.md instructions\nrepo rules",
        "installation-a",
    ));

    let variants = [
        key(treatment_plan(
            "openai.gpt-5.6-sol",
            "system instructions",
            &tools,
            "# AGENTS.md instructions\nrepo rules",
            "installation-a",
        )),
        key(treatment_plan(
            "openai.gpt-5.6-terra",
            "changed system instructions",
            &tools,
            "# AGENTS.md instructions\nrepo rules",
            "installation-a",
        )),
        key(treatment_plan(
            "openai.gpt-5.6-terra",
            "system instructions",
            &json!([{"type": "function", "name": "different"}]),
            "# AGENTS.md instructions\nrepo rules",
            "installation-a",
        )),
        key(treatment_plan(
            "openai.gpt-5.6-terra",
            "system instructions",
            &tools,
            "# AGENTS.md instructions\ndifferent repo rules",
            "installation-a",
        )),
        key(treatment_plan(
            "openai.gpt-5.6-terra",
            "system instructions",
            &tools,
            "# AGENTS.md instructions\nrepo rules",
            "installation-b",
        )),
    ];

    assert!(variants.iter().all(|variant| variant != &baseline));

    let changed_provider = stable_prefix_cache_plan(
        /*rollout_percent*/ 100,
        "Different Bedrock provider",
        Some("https://other-bedrock.example/v1"),
        "openai.gpt-5.6-terra",
        "system instructions",
        &tools,
        &repository_input("# AGENTS.md instructions\nrepo rules"),
        "origin=https://github.com/genhealth/etl.git",
        "installation-a",
        "session-a",
    );
    assert_ne!(key(changed_provider), baseline);

    let changed_repository = stable_prefix_cache_plan(
        /*rollout_percent*/ 100,
        "Amazon Bedrock Mantle",
        Some("https://bedrock-mantle.example/v1"),
        "openai.gpt-5.6-terra",
        "system instructions",
        &tools,
        &repository_input("# AGENTS.md instructions\nrepo rules"),
        "origin=https://github.com/genhealth/other.git",
        "installation-a",
        "session-a",
    );
    assert_ne!(key(changed_repository), baseline);
}

#[test]
fn kill_switch_and_incompatible_requests_preserve_control_shape() {
    let input = repository_input("# AGENTS.md instructions\nrepo rules");
    let tools = json!([]);
    let plan = |percent, provider, model| {
        stable_prefix_cache_plan(
            percent,
            provider,
            /*provider_base_url*/ None,
            model,
            "system instructions",
            &tools,
            &input,
            "origin=https://github.com/genhealth/etl.git",
            "installation-a",
            "session-a",
        )
    };

    assert_eq!(
        plan(0, "Amazon Bedrock Mantle", "openai.gpt-5.6-terra"),
        StablePrefixCachePlan::Disabled
    );
    assert_eq!(
        plan(100, "OpenAI", "openai.gpt-5.6-terra"),
        StablePrefixCachePlan::Disabled
    );
    assert_eq!(
        plan(100, "Amazon Bedrock Mantle", "openai.gpt-5.5"),
        StablePrefixCachePlan::Disabled
    );
    assert!(matches!(
        stable_prefix_cache_plan(
            /*rollout_percent*/ 100,
            "Amazon Bedrock Mantle",
            /*provider_base_url*/ None,
            "openai.gpt-5.6-terra",
            "system instructions",
            &tools,
            &input,
            "",
            "installation-a",
            "session-a",
        ),
        StablePrefixCachePlan::Treatment { .. }
    ));
}

#[test]
fn treatment_marks_existing_repository_instruction_content() {
    let StablePrefixCachePlan::Treatment {
        prompt_cache_options,
        prompt_cache_breakpoints,
        ..
    } = treatment_plan(
        "openai.gpt-5.6-terra",
        "system instructions",
        &json!([]),
        "# AGENTS.md instructions\nrepo rules",
        "installation-a",
    )
    else {
        panic!("expected treatment plan");
    };

    assert_eq!(prompt_cache_options.mode, PromptCacheMode::Explicit);
    assert_eq!(prompt_cache_options.ttl, PromptCacheTtl::ThirtyMinutes);
    assert_eq!(
        prompt_cache_breakpoints,
        vec![PromptCacheBreakpoint {
            input_index: 0,
            content_index: 0,
        }]
    );
}

#[test]
fn treatment_marks_stable_prefix_and_repository_instruction_boundaries() {
    let mut input = repository_input("stable developer instructions");
    input.extend(repository_input(
        "# AGENTS.md instructions for /review-worktrees/run-a\nrepo rules",
    ));

    let StablePrefixCachePlan::Treatment {
        prompt_cache_breakpoints,
        ..
    } = stable_prefix_cache_plan(
        /*rollout_percent*/ 100,
        "Amazon Bedrock Mantle",
        Some("https://bedrock-mantle.example/v1"),
        "openai.gpt-5.6-terra",
        "system instructions",
        &json!([]),
        &input,
        "origin=https://github.com/genhealth/etl.git",
        "installation-a",
        "session-a",
    )
    else {
        panic!("expected treatment plan");
    };

    assert_eq!(
        prompt_cache_breakpoints,
        vec![
            PromptCacheBreakpoint {
                input_index: 0,
                content_index: 0,
            },
            PromptCacheBreakpoint {
                input_index: 1,
                content_index: 0,
            },
        ]
    );
}
