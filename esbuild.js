/* eslint-disable import/no-extraneous-dependencies */
require("esbuild")
    .build({
        entryPoints: [
            "./client-src/pico/pico.css",
            "./client-src/base/base.css",
            "./client-src/index/index.css",
            "./client-src/create/index.js",
            "./client-src/create-results/index.js",
            "./client-src/digitize/index.js",
        ],
        entryNames: "[dir]", // will name the result files by their folder names
        outbase: "./client-src",
        bundle: true,
        minify: true,
        outdir: "./sketch_map_tool/static/bundles",
        format: "esm",
        external: [
            "/static/assets/*",
        ],
    }).catch(() => process.exit(1));
/* eslint-enable */