import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  /* config options here */
  reactCompiler: true,
  experimental: {
    turbo: {
      resolveAlias: {
        fs: false,
        path: false
      }
    }
  }
};

export default nextConfig;
