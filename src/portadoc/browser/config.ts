/**
 * Browser app configuration loader.
 * Loads config from /config.json at runtime with typed defaults.
 */

export interface OcrEngineConfig {
  /** Adjustment to add to confidence scores (can be negative) */
  confidenceAdjustment: number;
}

export interface HarmonizeConfig {
  /** Minimum IoU for bbox matching */
  iouThreshold: number;
  /** IoU threshold reduction when text matches */
  textMatchBonus: number;
  /** Max center distance for text-match fallback */
  centerDistanceMax: number;
  /** Y-axis tolerance for same-line detection */
  yBandTolerance: number;
}

export interface AppConfig {
  ocr: {
    doctr: OcrEngineConfig;
    tesseract: OcrEngineConfig;
  };
  harmonize: HarmonizeConfig;
}

// Default configuration
const DEFAULT_CONFIG: AppConfig = {
  ocr: {
    doctr: {
      confidenceAdjustment: -10,
    },
    tesseract: {
      confidenceAdjustment: 0,
    },
  },
  harmonize: {
    iouThreshold: 0.3,
    textMatchBonus: 0.15,
    centerDistanceMax: 12.0,
    yBandTolerance: 10.0,
  },
};

let loadedConfig: AppConfig | null = null;

/**
 * Load configuration from /config.json.
 * Falls back to defaults if loading fails.
 */
export async function loadConfig(): Promise<AppConfig> {
  if (loadedConfig) {
    return loadedConfig;
  }

  try {
    const response = await fetch('/config.json');
    if (!response.ok) {
      console.warn(`Failed to load config.json (${response.status}), using defaults`);
      loadedConfig = DEFAULT_CONFIG;
      return loadedConfig;
    }

    const json = await response.json();
    // Deep merge with defaults to ensure all fields exist
    loadedConfig = deepMerge(
      DEFAULT_CONFIG as unknown as Record<string, unknown>,
      json as Record<string, unknown>
    ) as unknown as AppConfig;
    console.log('Loaded config:', loadedConfig);
    return loadedConfig;
  } catch (error) {
    console.warn('Error loading config.json, using defaults:', error);
    loadedConfig = DEFAULT_CONFIG;
    return loadedConfig;
  }
}

/**
 * Get config synchronously (must call loadConfig first).
 * Returns defaults if not yet loaded.
 */
export function getConfig(): AppConfig {
  return loadedConfig ?? DEFAULT_CONFIG;
}

/**
 * Apply confidence adjustment for a specific engine.
 * Clamps result to 0-100 range.
 */
export function adjustConfidence(engine: 'doctr' | 'tesseract', rawConfidence: number): number {
  const config = getConfig();
  const adjustment = config.ocr[engine]?.confidenceAdjustment ?? 0;
  return Math.max(0, Math.min(100, rawConfidence + adjustment));
}

/**
 * Deep merge two objects, with source overriding target.
 */
function deepMerge(target: Record<string, unknown>, source: Record<string, unknown>): Record<string, unknown> {
  const result = { ...target };

  for (const key of Object.keys(source)) {
    if (
      source[key] !== null &&
      typeof source[key] === 'object' &&
      !Array.isArray(source[key]) &&
      target[key] !== null &&
      typeof target[key] === 'object' &&
      !Array.isArray(target[key])
    ) {
      result[key] = deepMerge(
        target[key] as Record<string, unknown>,
        source[key] as Record<string, unknown>
      );
    } else {
      result[key] = source[key];
    }
  }

  return result;
}
