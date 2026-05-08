const fs = require('fs');
const path = require('path');

// Read app.js
const appJsPath = path.join(__dirname, 'app.js');
const appJsContent = fs.readFileSync(appJsPath, 'utf8');

// Extract extractSources function body using regex
const functionRegex = /function extractSources[\s\S]*?\n  \}/;
const match = appJsContent.match(functionRegex);
if (!match) {
  console.error("Error: Could not find extractSources function in app.js");
  process.exit(1);
}

const functionCode = match[0];

// Safely construct the function in the current environment
const extractSources = new Function(`return (${functionCode})`)();

console.log("Running frontend extractSources unit tests...\n");

let failures = 0;

function assertDeepEqual(actual, expected, testName) {
  const actualStr = JSON.stringify(actual);
  const expectedStr = JSON.stringify(expected);
  if (actualStr === expectedStr) {
    console.log(`[PASS] ${testName}`);
  } else {
    console.error(`[FAIL] ${testName}`);
    console.error(`       Expected: ${expectedStr}`);
    console.error(`       Got:      ${actualStr}`);
    failures++;
  }
}

// Scenario 1: Legacy /api/rag format (response.sources is present)
assertDeepEqual(
  extractSources({
    sources: ["local:file1.txt", "https://example.com/a"]
  }),
  ["local:file1.txt", "[https://example.com/a](https://example.com/a)"],
  "Legacy sources array extraction"
);

// Scenario 2: Derived retrieval.sources is present
assertDeepEqual(
  extractSources({
    retrieval: {
      sources: ["https://example.com/b", "local:file2.txt"]
    }
  }),
  ["[https://example.com/b](https://example.com/b)", "local:file2.txt"],
  "retrieval.sources array extraction"
);

// Scenario 3: retrieval.chunks format with source or url
assertDeepEqual(
  extractSources({
    retrieval: {
      chunks: [
        { source: "local:file3.txt" },
        { url: "https://example.com/c" },
        { source: "https://example.com/d" }
      ]
    }
  }),
  [
    "local:file3.txt",
    "[https://example.com/c](https://example.com/c)",
    "[https://example.com/d](https://example.com/d)"
  ],
  "retrieval.chunks source and url extraction"
);

// Scenario 4: Duplicate source removal
assertDeepEqual(
  extractSources({
    sources: ["local:dup.txt", "https://example.com/dup"],
    retrieval: {
      sources: ["local:dup.txt"],
      chunks: [
        { source: "https://example.com/dup" },
        { url: "local:dup.txt" }
      ]
    }
  }),
  ["local:dup.txt", "[https://example.com/dup](https://example.com/dup)"],
  "Duplicate sources are deduplicated"
);

// Scenario 5: Filter empty/null/unknown/undefined values
assertDeepEqual(
  extractSources({
    sources: [null, "unknown", "", "  ", "NULL", "local:valid.txt", undefined]
  }),
  ["local:valid.txt"],
  "Filter empty/null/unknown values"
);

// Scenario 6: Empty response handling
assertDeepEqual(extractSources(null), [], "Null response returns empty array");
assertDeepEqual(extractSources({}), [], "Empty object returns empty array");
assertDeepEqual(extractSources({ retrieval: {} }), [], "Empty retrieval object returns empty array");

console.log(`\nTests completed. Failures: ${failures}`);
process.exit(failures > 0 ? 1 : 0);
