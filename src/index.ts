import {
  FunctionTool,
  OpenAI as LIOpenAI,
  OpenAIAgent,
  Settings,
  SubQuestionQueryEngine,
} from "llamaindex";
import * as path from "path";
import * as fs from "node:fs";
import OpenAI from "openai";
import { CodeRAG } from "./CodeRAG/index.js";
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

  /*
  const queryEngine = SubQuestionQueryEngine.fromDefaults({
    queryEngineTools: [, dependencyTool],
  });
  */

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
      FunctionTool.from(
        ({ filepath, content }: { filepath: string; content: string }) => {
          fs.writeFileSync(path.join(workspacePath, filepath), content);
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
    ],
  });

  const { query } = await prompts({
    type: "text",
    name: "query",
    message: "What you wanna query?",
  });
  /*
  let response = await queryEngine.query({
    query: `With the following request, delimited by triple quotes:

"""    
${query}
"""

# Instructions
- Validate if the request makes sense for the current codebase
- Suggest a general implementation of the request without installing any new dependencies`,
  });

  console.log(response.sourceNodes);
  console.log(response.response);
  console.log("====================");
*/
  let agentResponse = await agent.chat({
    message: `${query}

# Instructions
1. Query the code_rag_tool in the format of "Does this suggested change {CHANGE} already exist in the code?"
2. If the suggested change does not exist and it is a new implementation, query the dependency_rag_tool in the format of "Write a short general guide on how to implement: {CHANGE}" 
3. Query the code_rag_tool in the format of "List the files and directories that would be affected by the following change: {CHANGE}?"
4. Read any affected files and write the suggested changes to the codebase
5. Respond with a short summary of the changes without code references`,
  });
  console.log(agentResponse.sources);
  console.log(agentResponse.response);
  console.log("DONE!!!");
}

main().catch(console.error);
