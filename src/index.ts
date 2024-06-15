import {
  FunctionTool,
  OpenAI as LIOpenAI,
  OpenAIAgent,
  QueryEngineTool,
  Settings,
} from "llamaindex";
import * as path from "path";
import * as fs from "node:fs";
import OpenAI from "openai";
import { CodeRAG } from "./CodeRAG/index.js";
import prompts from "prompts";
import { DependencyRAG } from "./DependencyRAG.js";
import { writeToNestedFolder } from "./utils.js";

Settings.llm = new LIOpenAI({
  apiKey: process.env.OPENAI_API_KEY,
  model: "gpt-4o",
});

Settings.callbackManager.on("llm-tool-call", (event) => {
  const toolCall = event.detail.payload.toolCall;

  switch (toolCall.name) {
    case "readFileContent": {
      console.log("TOOL_CALL - readFileContent: " + toolCall.input.filepath);
      break;
    }
    case "writeFileContent": {
      console.log("TOOL_CALL - writeFileContent: " + toolCall.input.filepath);
      break;
    }
    default: {
      console.log("TOOL_CALL - RAG tool: " + toolCall.input.query);
    }
  }
});
Settings.callbackManager.on("llm-tool-result", (event) => {
  const toolResult = event.detail.payload.toolResult;

  if (toolResult.tool instanceof QueryEngineTool) {
    console.log(event.detail.payload.toolResult.output);
  }

  console.log("===");
});

const workspacePath = path.join(process.cwd(), "project");

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

const readFileTool = FunctionTool.from(
  ({ filepath }: { filepath: string }) =>
    fs.readFileSync(path.join(workspacePath, filepath)).toString(),
  {
    name: "readFileContent",
    description: "Use this function to read the contents of a file",
    parameters: {
      type: "object",
      properties: {
        filepath: {
          type: "string",
          description: "The path of the file",
        },
      },
      required: ["filepath"],
    },
  }
);

const writeFileTool = FunctionTool.from(
  ({ filepath, content }: { filepath: string; content: string }) => {
    writeToNestedFolder(path.join(workspacePath, filepath), content);
    return "OK";
  },
  {
    name: "writeFileContent",
    description: "Use this function to write the contents of a file",
    parameters: {
      type: "object",
      properties: {
        filepath: {
          type: "string",
          description: "The path of the file",
        },
        content: {
          type: "string",
          description: "The content of the file",
        },
      },
      required: ["filepath", "content"],
    },
  }
);

async function main() {
  const agentResponse = await superAgent();

  console.log("=======================================");
  console.log(agentResponse.response.message.content);
}

async function shittyAgent() {
  const agent = new OpenAIAgent({
    tools: [readFileTool, writeFileTool],
  });

  const { query } = await prompts({
    type: "text",
    name: "query",
    message: "What you wanna query?",
  });

  return agent.chat({
    message: `${query}

# Instructions
1. Write the suggested changes to the project
2. Respond with a short summary of the changes without code references`,
  });
}

async function simpleAgent() {
  const [codeRAG] = await Promise.all([CodeRAG.create(workspacePath, openai)]);

  // Split text and create embeddings. Store them in a VectorStoreIndex
  const [codeTool] = await Promise.all([codeRAG.createQueryEngineTool()]);

  const agent = new OpenAIAgent({
    tools: [codeTool, readFileTool, writeFileTool],
  });

  const { query } = await prompts({
    type: "text",
    name: "query",
    message: "What you wanna query?",
  });

  return agent.chat({
    message: `${query}

# Instructions
1. Query the code_rag_tool in the format of "Does this suggested change {CHANGE} already exist in the code?"
2. If the suggested change does not exist and it is a new implementation, query the code_rag_tool in the format of "List the files and directories that would be affected by the following change: {CHANGE}?"
3. Read any affected files and write the suggested changes to the codebase
4. Respond with a short summary of the changes without code references`,
  });
}

async function superAgent() {
  const [codeRAG, dependencyRAG] = await Promise.all([
    CodeRAG.create(workspacePath, openai),
    DependencyRAG.create(workspacePath, openai),
  ]);

  // Split text and create embeddings. Store them in a VectorStoreIndex
  const [codeTool, dependencyTool] = await Promise.all([
    codeRAG.createQueryEngineTool(),
    dependencyRAG.createQueryEngineTool(),
  ]);

  const agent = new OpenAIAgent({
    tools: [codeTool, dependencyTool, readFileTool, writeFileTool],
  });

  const { query } = await prompts({
    type: "text",
    name: "query",
    message: "What you wanna query?",
  });

  return agent.chat({
    message: `${query}

# Instructions
1. Query the code_rag_tool in the format of "Does this suggested change {CHANGE} already exist in the code?"
2. If the suggested change does not exist and it is a new implementation, query the dependency_rag_tool in the format of "Write a short general guide on how to implement: {CHANGE}" 
3. Query the code_rag_tool in the format of "List the files and directories that would be affected by the following change: {CHANGE}?"
4. Read any affected files and write the suggested changes to the codebase
5. Respond with a short summary of the changes without code references`,
  });
}

main().catch(console.error);
