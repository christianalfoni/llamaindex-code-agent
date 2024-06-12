type TreeNode = TreeFileNode | TreeDirectoryNode;

type TreeFileNode = {
  type: "file";
  path: string;
};

type TreeDirectoryNode = {
  type: "directory";
  path: string;
  children: TreeNode[];
};

type Result = { path: string; summary: string };

type VisitFileCallback = (path: string) => Promise<string>;
type VisitDirectoryCallback = (
  path: string,
  results: Result[]
) => Promise<string>;

export async function traverseTree(
  paths: string[],
  fileCallback: VisitFileCallback,
  directoryCallback: VisitDirectoryCallback
) {
  const root: TreeDirectoryNode = { type: "directory", path: "", children: [] };

  for (const path of paths) {
    const parts = path.split("/");
    let currentNode: TreeDirectoryNode = root;
    for (let x = 0; x < parts.length; x++) {
      const part = parts[x];

      let childNode: TreeNode | undefined = currentNode.children.find(
        (node) => node.path === part
      );

      if (childNode?.type === "directory") {
        currentNode = childNode;
        continue;
      }

      if (x < parts.length - 1) {
        childNode = {
          type: "directory",
          path: parts.slice(0, x + 1).join("/"),
          children: [],
        };
        currentNode.children.push(childNode);
        currentNode = childNode;
      } else {
        currentNode.children.push({ type: "file", path });
      }
    }
  }

  async function traverse(node: TreeNode): Promise<{
    path: string;
    summary: string;
  }> {
    if (node.type === "file") {
      return {
        path: node.path,
        summary: await fileCallback(node.path),
      };
    }

    const summaries = await Promise.all(node.children.map(traverse));

    return {
      path: node.path,
      summary: await directoryCallback(node.path, summaries),
    };
  }

  return traverse(root);
}
