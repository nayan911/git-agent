import { config } from "dotenv";
config();

import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { tool } from "@langchain/core/tools";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import { MessagesAnnotation, StateGraph } from "@langchain/langgraph";
import * as z from "zod";
import { execSync } from "child_process";

const llm = new ChatGoogleGenerativeAI({
  model: "gemini-2.5-flash",
  maxOutputTokens: 2048,
  apiKey: process.env.GOOGLE_API_KEY,
});

// 🧰 Tool: Generate commit message from git diff
const generateCommitMessage = tool(
  async () => {
    try {
      // 🟢 Get both staged and unstaged diffs
      let diff = "";
      try {
        diff += execSync("git diff --cached", { encoding: "utf-8" });
      } catch {}
      try {
        diff += execSync("git diff", { encoding: "utf-8" });
      } catch {}

      // 🟢 Also include untracked files (if any)
      let untrackedFiles = [];
      try {
        untrackedFiles = execSync("git ls-files --others --exclude-standard", {
          encoding: "utf-8",
        })
          .split("\n")
          .filter(Boolean);
      } catch {}

      for (const file of untrackedFiles) {
        try {
          const content = execSync(`cat "${file}"`, { encoding: "utf-8" });
          diff += `\n\n--- New file: ${file} ---\n${content}\n`;
        } catch {}
      }

      // 🛑 If there’s truly nothing changed
      if (!diff.trim()) return "No changes detected.";

      // 🧠 Ask Gemini to summarize it into a commit message
      const response = await llm.invoke([
        {
          role: "system",
          content: `
You are a professional software engineer writing concise Git commit messages.
Summarize the intent of these changes in one clear sentence.
Example: "Refactor index2.js to import necessary modules and configure dotenv".
          `,
        },
        { role: "user", content: diff },
      ]);

      // ✅ Return the generated message
      return response.content;
    } catch (err) {
      return `❌ Failed to generate commit message: ${err.message}`;
    }
  },
  {
    name: "generate_commit_message",
    description:
      "Generate a concise, meaningful commit message based on git diff and untracked files.",
    schema: z.object({}),
  }
);

// 🧰 Tool: Commit
const gitCommit = tool(
  ({ message }) => {
    try {
      execSync(`git add . && git commit -m "${message}"`, { stdio: "inherit" });
      return `✅ Commit successful with message: "${message}"`;
    } catch (err) {
      return `❌ Commit failed: ${err.message}`;
    }
  },
  {
    name: "git_commit",
    description: "Commit all staged changes with a given message.",
    schema: z.object({
      message: z.string().describe("The commit message to use."),
    }),
  }
);

// 🧰 Tool: Push
const gitPush = tool(
  () => {
    try {
      const branch = execSync("git branch --show-current", { encoding: "utf-8" });
      console.log('branch', branch);
      execSync("git push", { stdio: "inherit" });
      return "✅ Push successful";
    } catch (err) {
      return `❌ Push failed: ${err.message}`;
    }
  },
  {
    name: "git_push",
    description: "Push committed changes to the remote repository.",
    schema: z.object({}),
  }
);

const tools = [generateCommitMessage, gitCommit, gitPush];
const llmWithTools = llm.bindTools(tools);
const toolNode = new ToolNode(tools);

// 🧠 LLM Node
async function llmCall(state) {
  const result = await llmWithTools.invoke([
    {
      role: "system",
      content: `
You are an autonomous Git assistant.
Your job: 
1️⃣ If user asks to commit, but no message provided, call generate_commit_message first.
2️⃣ Then call git_commit with that message.
3️⃣ If user asks to push, call git_push.
Never run shell commands directly.
      `
    },
    ...state.messages
  ]);

  return { messages: [result] };
}

// 🔁 Conditional routing
function shouldContinue(state) {
  const lastMessage = state.messages.at(-1);
  if (lastMessage?.tool_calls?.length) return "toolNode";
  return "__end__";
}

// 🧩 Build the agent graph
const agentBuilder = new StateGraph(MessagesAnnotation)
  .addNode("llmCall", llmCall)
  .addNode("toolNode", toolNode)
  .addEdge("__start__", "llmCall")
  .addConditionalEdges("llmCall", shouldContinue, ["toolNode", "__end__"])
  .addEdge("toolNode", "llmCall")
  .compile();

// 🧪 Example usage
const messages = [
  {
    role: "user",
    content: "Please commit the recent changes and push them.",
  }
];

const result = await agentBuilder.invoke({ messages });
console.log("\nFinal Conversation:\n", result.messages);
