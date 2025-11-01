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

const gitPush = tool(
  () => {
    try {
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

const tools = [gitCommit, gitPush];
const llmWithTools = llm.bindTools(tools);
const toolNode = new ToolNode(tools);

async function llmCall(state) {
  const result = await llmWithTools.invoke([
    {
      role: "system",
      content: `
You are a helpful assistant that manages git commits and pushes. 
When a user asks you to commit or push code, use the provided tools.
Never write shell commands yourself — only use the tools.
If the commit is successful, you can then push the changes.
      `
    },
    ...state.messages
  ]);

  return { messages: [result] };
}

function shouldContinue(state) {
  const lastMessage = state.messages.at(-1);
  if (lastMessage?.tool_calls?.length) return "toolNode";
  return "__end__";
}

const agentBuilder = new StateGraph(MessagesAnnotation)
  .addNode("llmCall", llmCall)
  .addNode("toolNode", toolNode)
  .addEdge("__start__", "llmCall")
  .addConditionalEdges("llmCall", shouldContinue, ["toolNode", "__end__"])
  .addEdge("toolNode", "llmCall")
  .compile();

  const messages = [
  {
    role: "user",
    content: 'Please commit with message "testing git here" and push the changes.'
  }
];

const result = await agentBuilder.invoke({ messages });
console.log("\nFinal Conversation:\n", result.messages);
