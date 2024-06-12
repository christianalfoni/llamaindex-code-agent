import {
  Document,
  Metadata,
  QueryEngineTool,
  VectorStoreIndex,
  storageContextFromDefaults,
} from "llamaindex";
import * as path from "path";
import { WEB_ASSISTANT_FOLDER_NAME } from "./utils.js";
import { SerialQueue } from "./SerialQueue.js";
import OpenAI from "openai";
import * as fs from "node:fs/promises";

const DEPENDENCY_EMBEDDINGS_FOLDER_NAME = "dependency-embeddings";
// Should probably query the index to figure out if a dependency is missing
const DEPENDENCY_REF_FILE_NAME = "dependencies.json";

const GPT_4o_INPUT_PRICE = 5;
const GPT_4o_OUTPUT_PRICE = 15;

export class DependencyRAG {
  constructor(
    public workspacePath: string,
    public index: VectorStoreIndex,
    public allFilePaths: string[]
  ) {}
  async getDependencies() {
    const packageJson = await readFile(
      path.join(this.workspacePath, "package.json")
    );

    return Object.keys(JSON.parse(packageJson).dependencies);
  }
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
        name: "dependency_rag_tool",
        description: `This tool has documentation for all dependencies in the project`,
      },
    });
  }
  static async create(workspacePath: string, openai: OpenAI) {
    console.log("Verifying dependency RAG...");

    const storageContext = await storageContextFromDefaults({
      persistDir: path.join(
        workspacePath,
        WEB_ASSISTANT_FOLDER_NAME,
        DEPENDENCY_EMBEDDINGS_FOLDER_NAME
      ),
    });

    let index: VectorStoreIndex;

    const [dependencies, packageJson] = await Promise.all([
      fs
        .readFile(
          path.join(
            workspacePath,
            WEB_ASSISTANT_FOLDER_NAME,
            DEPENDENCY_REF_FILE_NAME
          )
        )
        .then((buffer) => JSON.parse(buffer.toString()))
        .catch(() => ({})),
      fs
        .readFile(path.join(workspacePath, "package.json"))
        .then((buffer) => JSON.parse(buffer.toString())),
    ]);

    const missingDependencies: string[] = [];

    for (const depdendency in packageJson.dependencies) {
      if (!dependencies[depdendency]) {
        missingDependencies.push(depdendency);
      }
    }

    let documents: Document<Metadata>[] = [];

    if (missingDependencies.length) {
      console.log(
        "Found the following missing dependencies",
        missingDependencies
      );

      documents = await learnMissingDependencies(
        openai,
        workspacePath,
        missingDependencies
      );
    } else {
      console.log("No new dependencies, initializing dependencies RAG");
    }

    try {
      index = await VectorStoreIndex.init({
        storageContext,
      });

      await Promise.all(documents.map((document) => index.insert(document)));
    } catch (error) {
      index = await VectorStoreIndex.fromDocuments(documents, {
        storageContext,
      });
    }

    await fs.writeFile(
      path.join(
        workspacePath,
        WEB_ASSISTANT_FOLDER_NAME,
        DEPENDENCY_REF_FILE_NAME
      ),
      // WE really would need to store the minor version here to know when an
      // update is needed
      JSON.stringify(packageJson.dependencies, null, 2)
    );

    return new DependencyRAG(workspacePath, index, dependencies);
  }
}

async function readFile(path: string) {
  return fs.readFile(path).then((buffer) => buffer.toString());
}

async function getDependencyPackageJson(
  workspacePath: string,
  dependency: string
) {
  const packageJson = await readFile(
    path.join(workspacePath, "node_modules", dependency, "package.json")
  );

  return JSON.parse(packageJson);
}

/**
 * Resolves type definitions, including "export" from other files. We assume
 * all exports reference directories with a "index.d.ts" file or an actual "d.ts" file,
 * which we resolve manually. We identify exports using a simple regexp for now
 */
async function getDependencyTypes(
  workspacePath: string,
  dependency: string,
  packageJson: Record<string, any>
): Promise<Array<{ types: string; name: string }>> {
  async function resolveNestedTypes(typesPath: string) {
    const types = await readFile(typesPath);
    const exports = Array.from(
      types.matchAll(/export.*from\s['"](.*)['"]/g)
    ).map((match) => match[1]);

    const nestedTypes: string[] = await Promise.all(
      exports.map(async (exportPathOrDependency) => {
        console.log(
          "Resolving nested types from ",
          typesPath,
          exportPathOrDependency
        );

        if (
          exportPathOrDependency.startsWith("./") ||
          exportPathOrDependency.startsWith("../")
        ) {
          // No exports are explicit to ".d.ts", so if the stat fails, it means it's a directory
          const isDirectory = await fs
            .stat(path.resolve(path.dirname(typesPath), exportPathOrDependency))
            .then(() => true)
            .catch(() => false);

          return resolveNestedTypes(
            path.resolve(
              path.dirname(typesPath),
              isDirectory
                ? path.join(exportPathOrDependency, "index.d.ts")
                : exportPathOrDependency + ".d.ts"
            )
          );
        }

        const nestedTypesFromDependency = await getDependencyTypes(
          workspacePath,
          exportPathOrDependency,
          await getDependencyPackageJson(workspacePath, exportPathOrDependency)
        );

        return nestedTypesFromDependency.map(({ types }) => types).join("\n\n");
      })
    );

    return `${types}\n\n${nestedTypes.join("\n\n")}`;
  }

  if (packageJson.types) {
    console.log("Resolving types from types field...");
    return [
      {
        types: await resolveNestedTypes(
          path.resolve(
            workspacePath,
            "node_modules",
            dependency,
            packageJson.types
          )
        ),
        name: dependency,
      },
    ];
  }

  const keysWithTypes = Object.keys(packageJson.exports || {}).filter((key) =>
    Boolean(packageJson.exports[key].types)
  );

  if (keysWithTypes.length) {
    console.log(
      "Resolving types from exports " + JSON.stringify(keysWithTypes) + "..."
    );
    return Promise.all(
      keysWithTypes.map(async (key) => {
        const typesPath = packageJson.exports[key].types;
        const name = path.join(dependency, key);

        return {
          types: await resolveNestedTypes(
            path.resolve(workspacePath, "node_modules", dependency, typesPath)
          ),
          name,
        };
      })
    );
  }

  const definitelyTypedDependency = `@types/${dependency}`;
  const hasDefinitelyTyped = await fs
    .stat(
      path.resolve(workspacePath, "node_modules", definitelyTypedDependency)
    )
    .then(() => true)
    .catch(() => false);

  if (hasDefinitelyTyped) {
    console.log("Resolving types from DefinitelyTyped...");
    const definitelyTypedPackageJson = await getDependencyPackageJson(
      workspacePath,
      definitelyTypedDependency
    );

    return getDependencyTypes(
      workspacePath,
      definitelyTypedDependency,
      definitelyTypedPackageJson
    );
  }

  console.log("WARNING: Dependency has no types", dependency);

  return [];
}

async function learnMissingDependencies(
  openai: OpenAI,
  workspacePath: string,
  missingDependencies: string[]
) {
  // Rather become a set amount of parallel items in queue
  const queue = new SerialQueue();

  const messages: Array<{ dependency: string; message: string }>[] =
    await Promise.all(
      missingDependencies.map(async (dependency) => {
        console.log("Finding package.json for " + dependency);
        const packageJson = await getDependencyPackageJson(
          workspacePath,
          dependency
        );
        console.log("Finding types for " + dependency);
        const [types, readme] = await Promise.all([
          getDependencyTypes(workspacePath, dependency, packageJson),
          readFile(
            path.join(workspacePath, "node_modules", dependency, "README.md")
          ),
        ]);

        return types.map(({ name, types }) => {
          if (name === dependency) {
            return {
              dependency,
              message: `Could you generate some usage examples from this README and typescript definitions of a dependency called "${dependency}"?

${readme}

\`\`\`ts
${types}
\`\`\``,
            };
          }

          return {
            dependency: name,
            message: `Could you generate some usage examples from these typescript definitions of a dependency called "${dependency}"?

\`\`\`ts
${types}
\`\`\``,
          };
        });
      })
    );

  return Promise.all(
    messages.flat().map(
      ({ message, dependency }) =>
        new Promise<Document>(async (resolve) => {
          queue.add(async () => {
            console.log("Learning " + dependency + "...");
            const response = await openai.chat.completions.create({
              model: "gpt-4o",
              temperature: 0.2,
              messages: [
                {
                  role: "user",
                  content: message,
                },
              ],
            });
            const text = response.choices[0].message.content;

            const document = new Document({
              text: text!,
              id_: dependency,
              metadata: {
                name: dependency,
              },
            });
            const price = response.usage
              ? {
                  input: response.usage.prompt_tokens,
                  output: response.usage.completion_tokens,
                  price:
                    response.usage.prompt_tokens *
                      (GPT_4o_INPUT_PRICE / 1_000_000) +
                    response.usage.completion_tokens *
                      (GPT_4o_OUTPUT_PRICE / 1_000_000),
                  model: "gpt-4o",
                }
              : "No usage reported";
            console.log("Usage examples created for " + dependency);
            console.log(text);
            console.log("==============", price);

            resolve(document);
          });
        })
    )
  );
}
