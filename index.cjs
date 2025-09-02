class NoopWarp {
  constructor(_dbname) {
    console.warn("*** constructing no-op Warp instance ***");
  }
  async search(query) {
    console.warn("no-op warp result for query", query);
    return [];
  }
}

try {
  module.exports = require('../../packages/warp/warp.node');
  console.log("Warp module successfully loaded", module.exports);
} catch {
  module.exports.Warp = NoopWarp;
}
