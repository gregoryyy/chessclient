import resolve from "@rollup/plugin-node-resolve";
import commonjs from "@rollup/plugin-commonjs";
import typescript from "@rollup/plugin-typescript";
import serve from "rollup-plugin-serve";
import livereload from "rollup-plugin-livereload";

const dev = process.env.ROLLUP_WATCH === "true";

export default {
  input: "src/main.ts",
  output: {
    file: "dist/bundle.js",
    format: "esm",
    sourcemap: true
  },
  plugins: [
    resolve({ browser: true, preferBuiltins: false }),
    commonjs(),
    typescript({ tsconfig: "./tsconfig.json" }),
    // copy static files by serving from /public during dev and telling users to copy to dist for prod
    dev &&
      serve({
        contentBase: ["dist", "public"],
        host: "localhost",
        port: 5173,
        headers: {
          // Enable cross-origin isolation if you want to run threaded WASM later
          "Cross-Origin-Opener-Policy": "same-origin",
          "Cross-Origin-Embedder-Policy": "require-corp",
          // WASM MIME
          "Content-Type": "text/html; charset=UTF-8"
        }
      }),
    dev && livereload({ watch: ["dist", "public"] })
  ],
  watch: { clearScreen: false }
};
