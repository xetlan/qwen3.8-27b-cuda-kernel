// Chat formatting against the model's own Jinja template. Expected strings were rendered with
// jinja2 from testdata/qwen3.8/chat_template.jinja (Qwen3.8-27B), add_generation_prompt=True.
#include <string>
#include <vector>

#include "qwen38/chat.h"
#include "test_util.h"

using qwen38::ChatMessage;
using qwen38::ChatOptions;

QWEN38_TEST(chat_user_only_default) {
  ChatOptions o;
  const std::vector<ChatMessage> msgs = {{"user", "Hello!", ""}};
  const std::string expected = "<|im_start|>system\nReasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.<|im_end|>\n<|im_start|>user\nHello!<|im_end|>\n<|im_start|>assistant\n<think>\n";
  CHECK(qwen38::render_chat(msgs, o) == expected);
}

QWEN38_TEST(chat_with_system) {
  ChatOptions o;
  const std::vector<ChatMessage> msgs = {{"system", "You are terse.", ""}, {"user", "  Hi  ", ""}};
  const std::string expected = "<|im_start|>system\nReasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.\n\nYou are terse.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n";
  CHECK(qwen38::render_chat(msgs, o) == expected);
}

QWEN38_TEST(chat_no_think) {
  ChatOptions o;
  o.enable_thinking = false;
  const std::vector<ChatMessage> msgs = {{"user", "2+2?", ""}};
  const std::string expected = "<|im_start|>user\n2+2?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
  CHECK(qwen38::render_chat(msgs, o) == expected);
}

QWEN38_TEST(chat_effort_low) {
  ChatOptions o;
  o.reasoning_effort = "low";
  const std::vector<ChatMessage> msgs = {{"user", "x", ""}};
  const std::string expected = "<|im_start|>system\nReasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration.<|im_end|>\n<|im_start|>user\nx<|im_end|>\n<|im_start|>assistant\n<think>\n";
  CHECK(qwen38::render_chat(msgs, o) == expected);
}

QWEN38_TEST(chat_effort_medium_system) {
  ChatOptions o;
  o.reasoning_effort = "medium";
  const std::vector<ChatMessage> msgs = {{"system", "S", ""}, {"user", "x", ""}};
  const std::string expected = "<|im_start|>system\nS<|im_end|>\n<|im_start|>user\nx<|im_end|>\n<|im_start|>assistant\n<think>\n";
  CHECK(qwen38::render_chat(msgs, o) == expected);
}

QWEN38_TEST(chat_multi_turn) {
  ChatOptions o;
  const std::vector<ChatMessage> msgs = {{"user", "a", ""}, {"assistant", "b", "think b"}, {"user", "c", ""}};
  const std::string expected = "<|im_start|>system\nReasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.<|im_end|>\n<|im_start|>user\na<|im_end|>\n<|im_start|>assistant\n<think>\nthink b\n</think>\n\nb<|im_end|>\n<|im_start|>user\nc<|im_end|>\n<|im_start|>assistant\n<think>\n";
  CHECK(qwen38::render_chat(msgs, o) == expected);
}

QWEN38_TEST(chat_multi_turn_no_preserve) {
  ChatOptions o;
  o.preserve_thinking = false;
  const std::vector<ChatMessage> msgs = {{"user", "a", ""}, {"assistant", "b", "think b"}, {"user", "c", ""}};
  const std::string expected = "<|im_start|>system\nReasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.<|im_end|>\n<|im_start|>user\na<|im_end|>\n<|im_start|>assistant\nb<|im_end|>\n<|im_start|>user\nc<|im_end|>\n<|im_start|>assistant\n<think>\n";
  CHECK(qwen38::render_chat(msgs, o) == expected);
}
