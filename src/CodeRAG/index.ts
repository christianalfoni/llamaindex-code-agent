import * as fs from "node:fs/promises";
import {
  Document,
  VectorStoreIndex,
  storageContextFromDefaults,
  QueryEngineTool,
} from "llamaindex";
import * as path from "path";
import {
  WEB_ASSISTANT_FOLDER_NAME,
  codeExtensions,
  getAllFilePaths,
} from "../utils.js";
import { SerialQueue } from "../SerialQueue.js";
import { traverseTree } from "./traversePaths.js";
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
        name: "code_rag_tool",
        description: `This tool can answer detailed questions about the files in the codebase`,
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
      const queue = new SerialQueue();
      const documents: Document[] = [];

      async function getSummary(prompt: string) {
        const response = await openai.chat.completions.create({
          model: "gpt-3.5-turbo",
          temperature: 0.2,
          messages: [
            {
              role: "user",
              content: prompt,
            },
          ],
        });
        const summary = response.choices[0].message.content;

        if (!summary) {
          console.log("ERROR: No summary for\n" + prompt);
        }

        return summary || "Missing summary";
      }

      async function createFileDocument(filepath: string) {
        return queue.add(async () => {
          const absolutePath = path.join(workspacePath, filepath);
          const content = (await fs.readFile(absolutePath)).toString();
          const summary = await getSummary(`This is the code of a file:
\`\`\`
${content}
\`\`\`

Please give me a summary of what this file does.`);
          const document = new Document({
            text: `---
path: "${filepath}"
type: "file"
---
${summary}`,
            id_: filepath,
            metadata: {
              filepath: filepath,
              type: "file",
            },
          });

          console.log("## RESOLVED FILE", summary);

          documents.push(document);

          return summary;
        });
      }

      async function createDirectoryDocument(
        directorypath: string,
        summaries: string[]
      ) {
        return queue.add(async () => {
          const summary =
            await getSummary(`This is a list of file summaries in a directory:

${summaries.join("\n\n")}

Please give me a summary of all files in this directory.`);
          const document = new Document({
            text: `---
path: "${directorypath}"
type: "directory"
---
${summary}`,
            id_: directorypath,
            metadata: {
              filepath: directorypath,
              type: "directory",
            },
          });

          console.log("## RESOLVED DIRECTORY", summary);

          documents.push(document);

          return summary;
        });
      }

      await traverseTree(
        filesToEmbed,
        createFileDocument,
        async (directorypath, results) =>
          createDirectoryDocument(
            directorypath || "$ROOT",
            results.map((result) => result.summary)
          )
      );

      index = await VectorStoreIndex.fromDocuments(documents, {
        storageContext,
      });
    }

    return new CodeRAG(index, filesToEmbed);
  }
}
