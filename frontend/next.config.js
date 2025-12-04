/** @type {import('next').NextConfig} */
const nextConfig = {
  // Enable static export for GitHub Pages
  output: 'export',
  
  // Base path for GitHub Pages (your repo name)
  basePath: process.env.NODE_ENV === 'production' ? '/InsurancePredictor' : '',
  
  // Required for static export
  images: {
    unoptimized: true,
  },
  
  // Trailing slash for GitHub Pages compatibility
  trailingSlash: true,
}

module.exports = nextConfig
