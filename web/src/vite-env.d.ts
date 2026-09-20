/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_API_BASE?: string;
}

interface ImportMeta {
  // Optional: this module is also loaded outside Vite, by the node:test suite.
  readonly env?: ImportMetaEnv;
}
