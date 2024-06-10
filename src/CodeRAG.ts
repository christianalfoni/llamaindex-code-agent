import * as fs from "node:fs/promises";
import {
  Document,
  VectorStoreIndex,
  storageContextFromDefaults,
  OpenAIEmbedding,
  Settings,
  TextNode,
  OpenAI as LLamaOpenAi,
  QueryEngineTool,
} from "llamaindex";
import * as path from "path";
import {
  WEB_ASSISTANT_FOLDER_NAME,
  codeExtensions,
  getAllFilePaths,
  getIgnoreGlobs,
} from "./utils.js";
import { SerialQueue } from "./SerialQueue.js";
import OpenAI from "openai";

const CODE_EMBEDDINGS_FOLDER_NAME = "code-embeddings";

function getIndexTypeFromFilepath(filepath: string) {
  const extension = path.extname(filepath);

  return codeExtensions.includes(extension) ? "code" : "doc";
}

export class CodeRAG {
  constructor(public index: VectorStoreIndex, public allFilePaths: string[]) {}
  async createQueryEngineTool() {
    const retriever = await this.index.asRetriever({
      similarityTopK: 10,
    });

    const queryEngine = await this.index.asQueryEngine({
      retriever,
    });

    return new QueryEngineTool({
      queryEngine,
      metadata: {
        name: "code_tool",
        description: `This tool can answer detailed questions about the code and file paths`,
      },
    });
  }
  static async create(workspacePath: string, openai: OpenAI) {
    console.log("Verifying code RAG...");

    const storageContext = await storageContextFromDefaults({
      persistDir: path.join(
        workspacePath,
        WEB_ASSISTANT_FOLDER_NAME,
        CODE_EMBEDDINGS_FOLDER_NAME
      ),
    });

    let index: VectorStoreIndex;
    // Should use VectorStoreIndex.init, though probably errors if no index, need to handle that

    const filesToEmbed = await getAllFilePaths(workspacePath);

    try {
      index = await VectorStoreIndex.init({
        storageContext,
      });
    } catch (error) {
      const evaluatedFiles = await Promise.all(
        filesToEmbed.map(async (filepath) => {
          const absolutePath = path.join(workspacePath, filepath);
          const stat = await fs.stat(absolutePath);

          return {
            filepath,
            mtime: stat.mtime.getTime(),
            shouldUpdate: true,
          };
        })
      );

      const queue = new SerialQueue();

      const documents = await Promise.all(
        evaluatedFiles
          .filter((evaluatedFile) => evaluatedFile.shouldUpdate)
          .map(
            (evaluatedFile) =>
              new Promise<Document>((resolve) => {
                queue.add(async () => {
                  const absolutePath = path.join(
                    workspacePath,
                    evaluatedFile.filepath
                  );
                  const content = (await fs.readFile(absolutePath)).toString();
                  const type = getIndexTypeFromFilepath(evaluatedFile.filepath);
                  const response = await openai.chat.completions.create({
                    model: "gpt-4o",
                    temperature: 0.2,
                    messages: [
                      {
                        role: "user",
                        content: `This is the code of a file:
      \`\`\`
      ${content}
      \`\`\`
    
      Please give me a summary of what this file does.`,
                      },
                    ],
                  });
                  const summary = response.choices[0].message.content;

                  const document = new Document({
                    text: summary!,
                    id_: evaluatedFile.filepath,
                    metadata: {
                      filepath: evaluatedFile.filepath,
                      type,
                      mtime: evaluatedFile.mtime,
                    },
                  });
                  console.log(
                    "## RESOLVED DOC",
                    evaluatedFile.mtime,
                    evaluatedFile.filepath,
                    summary
                  );

                  resolve(document);
                });
              })
          )
      );

      index = await VectorStoreIndex.fromDocuments(documents, {
        storageContext,
      });
    }

    return new CodeRAG(index, filesToEmbed);
  }
}
