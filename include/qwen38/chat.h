// Qwen3.8 chat formatting (text-only subset of the model's chat_template.jinja, no tool calls).
#pragma once

#include <string>
#include <vector>

namespace qwen38 {

struct ChatMessage {
  std::string role;       // "system", "user" or "assistant"
  std::string content;
  std::string reasoning;  // assistant turns: previous thinking, if any
};

struct ChatOptions {
  bool enable_thinking = true;
  std::string reasoning_effort = "xhigh";  // "xhigh" (template default), "medium" or "low"
  bool preserve_thinking = true;           // template default: keep reasoning of earlier turns
};

// Renders the conversation exactly like the Jinja template with add_generation_prompt=true.
std::string render_chat(const std::vector<ChatMessage>& messages, const ChatOptions& options);

// Token ids used to split model output into reasoning and answer.
constexpr int kThinkOpenId = 248068;   // <think>
constexpr int kThinkCloseId = 248069;  // </think>

}  // namespace qwen38
