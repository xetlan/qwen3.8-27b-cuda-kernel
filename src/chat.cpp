#include "qwen38/chat.h"

#include "qwen38/common.h"

namespace qwen38 {

namespace {

std::string trim(const std::string& s) {
  const char* ws = " \t\n\r\f\v";
  const size_t b = s.find_first_not_of(ws);
  if (b == std::string::npos) return {};
  const size_t e = s.find_last_not_of(ws);
  return s.substr(b, e - b + 1);
}

std::string reasoning_instructions(const ChatOptions& o) {
  if (!o.enable_thinking) return {};
  if (o.reasoning_effort == "xhigh")
    return "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, "
           "consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final "
           "answer.";
  if (o.reasoning_effort == "low")
    return "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion "
           "without unnecessary elaboration.";
  QWEN38_CHECK(o.reasoning_effort == "medium", "reasoning effort must be xhigh, medium or low");
  return {};
}

}  // namespace

std::string render_chat(const std::vector<ChatMessage>& messages, const ChatOptions& o) {
  QWEN38_CHECK(!messages.empty(), "no messages");
  const std::string instr = reasoning_instructions(o);
  std::string out;
  size_t first = 0;
  if (messages[0].role == "system") {
    const std::string content = trim(messages[0].content);
    if (!content.empty()) {
      out += "<|im_start|>system\n" + (instr.empty() ? std::string() : instr + "\n\n") + content + "<|im_end|>\n";
    } else if (!instr.empty()) {
      out += "<|im_start|>system\n" + instr + "<|im_end|>\n";
    }
    first = 1;
  } else if (!instr.empty()) {
    out += "<|im_start|>system\n" + instr + "<|im_end|>\n";
  }
  // index of the last user turn (the template's last_query_index; no tool turns here)
  int last_query = -1;
  for (size_t i = 0; i < messages.size(); ++i)
    if (messages[i].role == "user") last_query = int(i);
  QWEN38_CHECK(last_query >= 0, "no user message");
  for (size_t i = first; i < messages.size(); ++i) {
    const ChatMessage& m = messages[i];
    const std::string content = trim(m.content);
    if (m.role == "user") {
      out += "<|im_start|>user\n" + content + "<|im_end|>\n";
    } else if (m.role == "assistant") {
      if (o.preserve_thinking || int(i) > last_query) {
        out += "<|im_start|>assistant\n<think>\n" + trim(m.reasoning) + "\n</think>\n\n" + content;
      } else {
        out += "<|im_start|>assistant\n" + content;
      }
      out += "<|im_end|>\n";
    } else {
      QWEN38_CHECK(m.role != "system", "system message must be at the beginning");
      fail("unsupported chat role '" + m.role + "'");
    }
  }
  out += "<|im_start|>assistant\n";
  out += o.enable_thinking ? "<think>\n" : "<think>\n\n</think>\n\n";
  return out;
}

}  // namespace qwen38
