import {
  FunctionTool,
  OpenAI as LIOpenAI,
  OpenAIAgent,
  Settings,
} from "llamaindex";
import * as path from "path";
import * as fs from "node:fs";
import OpenAI from "openai";
import { CodeRAG } from "./CodeRAG.js";
import prompts from "prompts";
import { DependencyRAG } from "./DependencyRAG.js";

Settings.llm = new LIOpenAI({
  apiKey: process.env.OPENAI_API_KEY,
  model: "gpt-4o",
});

Settings.callbackManager.on("llm-tool-call", (event) => {
  console.log(event.detail.payload);
});
Settings.callbackManager.on("llm-tool-result", (event) => {
  console.log(event.detail.payload);
});

const workspacePath = path.join(process.cwd(), "project");

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

async function main() {
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
    tools: [
      codeTool,
      dependencyTool,
      FunctionTool.from(
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
      ),
      FunctionTool.from(() => codeRAG.allFilePaths, {
        name: "getProjectFileStructure",
        description:
          "Use this function to read the file structure of the project",
        parameters: {
          type: "object",
          properties: {},
          required: [],
        },
      }),
      FunctionTool.from(() => dependencyRAG.getDependencies(workspacePath), {
        name: "getProjectDependencies",
        description:
          "Use this function to get a list of dependencies and their descriptions",
        parameters: {
          type: "object",
          properties: {},
          required: [],
        },
      }),
    ],
  });

  const { query } = await prompts({
    type: "text",
    name: "query",
    message: "What you wanna query?",
  });

  let response = await agent.chat({
    message: `${query}

# Instructions
- Always verify if the prompt is relevant for the project. If not, ask for more information
- Only use the dependencies of the project and the related documentation`,
  });

  console.log(response.sources);
  console.log(response);
  console.log("DONE!!!");
}

main().catch(console.error);
