/**
 * Base path utility for GitHub Pages compatibility.
 * Vite sets import.meta.env.BASE_URL to '/portadoc/' in production.
 */

/**
 * Get the base path for the application.
 * Returns '/portadoc/' on GitHub Pages, './' or '/' locally.
 */
export function getBasePath(): string {
  // Vite injects BASE_URL at build time
  const base = import.meta.env.BASE_URL || '/';
  return base;
}

/**
 * Resolve a path relative to the base URL.
 * Handles both absolute paths (/foo) and relative paths (foo).
 */
export function resolveAssetPath(path: string): string {
  const base = getBasePath();

  // Remove leading slash from path if present
  const cleanPath = path.startsWith('/') ? path.slice(1) : path;

  // Ensure base ends with slash
  const cleanBase = base.endsWith('/') ? base : base + '/';

  return cleanBase + cleanPath;
}
