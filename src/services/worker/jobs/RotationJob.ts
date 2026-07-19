
// Rotation job — idle-time maintenance pass that deduplicates near-identical
// observation records within a single project.
//
// Pipeline per project:
//   1. Fetch all non-superseded observations (superseded_by_id IS NULL).
//   2. For each unvisited observation, query Chroma for its nearest
//      neighbors scoped to the same project, and build a candidate cluster
//      from neighbors whose similarity clears CANDIDATE_THRESHOLD.
//   3. If every member of the cluster clears AUTOMERGE_THRESHOLD, merge
//      automatically. Otherwise ask the local LLM a strict yes/no + merged
//      content question; only merge on a valid "yes" (fail closed on any
//      parse/shape problem — never throws, just logs and skips).
//   4. On merge: write ONE new consolidated observation (+ Chroma sync),
//      then mark the originals' superseded_by_id to point at it. Rows are
//      NEVER hard-deleted — always reversible by clearing the column.
//
// Idempotency: every read filters on superseded_by_id IS NULL, so a row
// merged by a previous run can never be re-clustered by a later one.

import { logger } from '../../../utils/logger.js';
import { SettingsDefaultsManager } from '../../../shared/SettingsDefaultsManager.js';
import { getCredential } from '../../../shared/EnvManager.js';
import { resolveOpenRouterChatCompletionsUrl } from '../../../shared/openrouter-base-url.js';
import { USER_SETTINGS_PATH } from '../../../shared/paths.js';
import type { DatabaseManager } from '../DatabaseManager.js';
import type { ParsedObservation } from '../../../sdk/parser.js';

// Cap on merged-cluster size. Chroma similarity is transitive-ish (A~B~C
// doesn't imply A~C at the same strength), so without a cap a long chain of
// loosely-related-but-individually-similar observations could snowball into
// one giant merge that swallows content that isn't really the same thing.
// 6 is an arbitrary but generous ceiling — large enough for real duplicate
// clusters (e.g. "fixed the same bug 4 times while iterating"), small enough
// that a bad chain can't consume dozens of unrelated observations.
const MAX_CLUSTER_SIZE = 6;

// How many Chroma neighbors to pull per anchor observation before filtering
// by CANDIDATE_THRESHOLD. Generous headroom over MAX_CLUSTER_SIZE because
// most neighbors will be below threshold.
const NEIGHBOR_QUERY_LIMIT = 20;

// 30s was measured to be too tight against this project's actual local
// setup (LM Studio / Gemma running on-device): a real merge-judgement call
// with 3 observations took ~45s end to end, even with reasoning suppressed
// via the request body (see askLLMToMergeCluster) — some local models keep
// emitting hidden reasoning_content regardless of that hint on anything
// beyond a trivial prompt. 90s gives real local inference room without
// hanging the rotation job indefinitely on a truly stuck request.
const LLM_TIMEOUT_MS = 90_000;

export interface RotationSummary {
  project: string;
  clustersConsidered: number;
  autoMerged: number;
  llmMerged: number;
  llmRejected: number;
  skipped: number;
}

interface RotationObservationRow {
  id: number;
  memory_session_id: string;
  project: string;
  type: string;
  title: string | null;
  subtitle: string | null;
  facts: string | null;
  narrative: string | null;
  concepts: string | null;
  files_read: string | null;
  files_modified: string | null;
  prompt_number: number | null;
  created_at_epoch: number;
}

interface RotationLLMDecision {
  shouldMerge: boolean;
  title: string;
  narrative: string;
  facts: string[];
}

function safeParseJsonArray(value: string | null): string[] {
  if (!value) return [];
  try {
    const parsed = JSON.parse(value);
    return Array.isArray(parsed) ? parsed.filter((v): v is string => typeof v === 'string') : [];
  } catch {
    return [];
  }
}

function dedupeUnion(values: string[]): string[] {
  return Array.from(new Set(values.filter(v => typeof v === 'string' && v.length > 0)));
}

function buildObservationQueryText(obs: RotationObservationRow): string {
  const facts = safeParseJsonArray(obs.facts);
  return [obs.title ?? '', obs.narrative ?? '', facts.join('\n')]
    .filter(part => part.trim().length > 0)
    .join('\n\n');
}

/**
 * Convert a Chroma query distance to a (0, 1] similarity score.
 *
 * Deviation from spec: ChromaSync.ensureCollectionExists() creates the
 * collection with no explicit `hnsw:space` metadata, so we cannot assume
 * the collection is configured for cosine distance (chroma's default is
 * l2, not cosine, when unspecified) — verified by reading that method
 * rather than assuming. Because the true metric is unverified/likely not
 * cosine, "similarity = 1 - distance" is not safe here. This transform is a
 * monotonic decreasing function of distance that saturates to (0, 1]; it
 * preserves relative ranking (closer => higher score) which is all
 * threshold-based clustering actually needs, and the configured thresholds
 * (0.75 / 0.92) are calibrated operationally against this transform rather
 * than against a formally-verified cosine similarity.
 */
function distanceToSimilarity(distance: number): number {
  return 1 / (1 + Math.max(0, distance));
}

/**
 * Minimal single-call LLM helper. worker/OpenRouterProvider.ts is built for
 * multi-turn agent sessions (ActiveSession/SessionManager/withRetry/usage
 * telemetry) — not a generic "ask one question" utility — so rather than
 * dragging that whole session machinery in, this reuses only its config
 * resolution helpers (SettingsDefaultsManager, getCredential,
 * resolveOpenRouterChatCompletionsUrl) and issues one plain fetch call.
 */
async function askLLMToMergeCluster(cluster: RotationObservationRow[]): Promise<RotationLLMDecision | null> {
  const settings = SettingsDefaultsManager.loadFromFile(USER_SETTINGS_PATH);
  const apiKey = settings.CLAUDE_MEM_OPENROUTER_API_KEY || getCredential('OPENROUTER_API_KEY') || '';
  if (!apiKey) {
    logger.warn('DEDUP', 'No OpenRouter API key configured; cannot ask LLM to judge merge candidate, treating as no-merge');
    return null;
  }

  const model = settings.CLAUDE_MEM_OPENROUTER_MODEL || 'xiaomi/mimo-v2-flash:free';
  const baseUrl = settings.CLAUDE_MEM_OPENROUTER_BASE_URL || process.env.OPENROUTER_BASE_URL || '';
  const apiUrl = resolveOpenRouterChatCompletionsUrl(baseUrl);

  const clusterDescription = cluster.map((obs, i) => {
    const facts = safeParseJsonArray(obs.facts);
    return `Observation ${i + 1} (id=${obs.id}):\nTitle: ${obs.title ?? '(none)'}\nNarrative: ${obs.narrative ?? '(none)'}\nFacts: ${facts.length ? facts.join('; ') : '(none)'}`;
  }).join('\n\n');

  const prompt = `You are deduplicating near-duplicate memory observations captured from the same software project. Below are ${cluster.length} observations that a similarity search flagged as possibly describing the same thing.

Decide whether they truly describe the SAME underlying fact/event/decision (in which case they should be merged into one observation) or whether they are meaningfully distinct (in which case they must NOT be merged).

If they should merge, write a single consolidated title, narrative, and fact list that captures everything important from all of them without redundancy.

Respond with STRICT JSON only — no markdown code fences, no commentary before or after — matching exactly this shape:
{"shouldMerge": boolean, "title": string, "narrative": string, "facts": string[]}

If shouldMerge is false, title/narrative/facts may be empty strings/array.

Observations:
${clusterDescription}`;

  try {
    const response = await fetch(apiUrl, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${apiKey}`,
        'Content-Type': 'application/json',
        'HTTP-Referer': settings.CLAUDE_MEM_OPENROUTER_SITE_URL || 'https://github.com/thedotmack/claude-mem',
        'X-Title': settings.CLAUDE_MEM_OPENROUTER_APP_NAME || 'claude-mem',
      },
      body: JSON.stringify({
        model,
        messages: [{ role: 'user', content: prompt }],
        temperature: 0.1,
        max_tokens: 1536,
        // Best-effort: suppresses hidden reasoning tokens on some local
        // backends (measured effective on short prompts, not reliably on
        // longer ones — see LLM_TIMEOUT_MS comment). Harmless no-op on
        // providers that don't recognize the field.
        reasoning: { effort: 'none' },
      }),
      signal: AbortSignal.timeout(LLM_TIMEOUT_MS),
    });

    if (!response.ok) {
      const bodyText = await response.text().catch(() => '');
      logger.warn('DEDUP', 'LLM merge-judgement request failed', { status: response.status, body: bodyText.slice(0, 300) });
      return null;
    }

    const data = await response.json() as { choices?: Array<{ message?: { content?: string } }> };
    const content = data.choices?.[0]?.message?.content;
    if (!content) {
      logger.warn('DEDUP', 'LLM merge-judgement returned empty content');
      return null;
    }

    return parseLLMDecision(content);
  } catch (error) {
    logger.warn('DEDUP', 'LLM merge-judgement call threw', { error: error instanceof Error ? error.message : String(error) });
    return null;
  }
}

function parseLLMDecision(content: string): RotationLLMDecision | null {
  // Strip markdown code fences — some models wrap JSON in them despite instructions.
  const cleaned = content.trim().replace(/^```(?:json)?\s*/i, '').replace(/```\s*$/i, '').trim();

  let parsed: unknown;
  try {
    parsed = JSON.parse(cleaned);
  } catch {
    logger.warn('DEDUP', 'LLM merge-judgement response was not valid JSON; treating as no-merge', { content: cleaned.slice(0, 300) });
    return null;
  }

  if (
    typeof parsed !== 'object' || parsed === null ||
    typeof (parsed as Record<string, unknown>).shouldMerge !== 'boolean' ||
    typeof (parsed as Record<string, unknown>).title !== 'string' ||
    typeof (parsed as Record<string, unknown>).narrative !== 'string' ||
    !Array.isArray((parsed as Record<string, unknown>).facts) ||
    !((parsed as Record<string, unknown>).facts as unknown[]).every(f => typeof f === 'string')
  ) {
    logger.warn('DEDUP', 'LLM merge-judgement JSON failed shape validation; treating as no-merge', { content: cleaned.slice(0, 300) });
    return null;
  }

  return parsed as RotationLLMDecision;
}

/**
 * Merge heuristic for the AUTO-MERGE path (no LLM call): union facts /
 * files_read / files_modified / concepts by string-equality dedup. For the
 * narrative: prefer the LONGER of (newest observation, longest-narrative
 * observation) when they're within 20% length of each other — roughly the
 * same "class" of content, so more detail wins. Otherwise the length gap is
 * assumed to signal a stale/truncated draft vs. a full rewrite, so prefer
 * the NEWER one regardless of length.
 */
function buildAutoMergeContent(cluster: RotationObservationRow[]): { title: string; narrative: string } {
  const newest = cluster.reduce((a, b) => (b.created_at_epoch > a.created_at_epoch ? b : a));
  const longestNarrativeObs = cluster.reduce((a, b) =>
    (b.narrative?.length ?? 0) > (a.narrative?.length ?? 0) ? b : a
  );

  const newestLen = newest.narrative?.length ?? 0;
  const longestLen = longestNarrativeObs.narrative?.length ?? 0;
  const withinTwentyPercent = newestLen > 0 && Math.abs(longestLen - newestLen) / newestLen <= 0.2;

  const narrative = withinTwentyPercent
    ? (longestNarrativeObs.narrative ?? newest.narrative ?? '')
    : (newest.narrative ?? longestNarrativeObs.narrative ?? '');

  const title = newest.title ?? longestNarrativeObs.title ?? 'Merged observation';

  return { title, narrative };
}

async function mergeCluster(
  dbManager: DatabaseManager,
  project: string,
  cluster: RotationObservationRow[],
  llmDecision: RotationLLMDecision | null
): Promise<void> {
  const sessionStore = dbManager.getSessionStore();
  const chromaSync = dbManager.getChromaSync();

  const unionFacts = dedupeUnion(cluster.flatMap(o => safeParseJsonArray(o.facts)));
  const filesRead = dedupeUnion(cluster.flatMap(o => safeParseJsonArray(o.files_read)));
  const filesModified = dedupeUnion(cluster.flatMap(o => safeParseJsonArray(o.files_modified)));
  const concepts = dedupeUnion(cluster.flatMap(o => safeParseJsonArray(o.concepts)));

  let title: string;
  let narrative: string;
  let facts: string[];

  if (llmDecision) {
    title = llmDecision.title || buildAutoMergeContent(cluster).title;
    narrative = llmDecision.narrative || buildAutoMergeContent(cluster).narrative;
    facts = dedupeUnion([...unionFacts, ...llmDecision.facts]);
  } else {
    const auto = buildAutoMergeContent(cluster);
    title = auto.title;
    narrative = auto.narrative;
    facts = unionFacts;
  }

  const newest = cluster.reduce((a, b) => (b.created_at_epoch > a.created_at_epoch ? b : a));
  const promptNumber = cluster.reduce((max, o) => Math.max(max, o.prompt_number ?? 0), 0) || undefined;

  const observationPayload: ParsedObservation = {
    type: 'discovery',
    title,
    subtitle: 'Rotation merge (deduplicated)',
    facts,
    narrative,
    concepts,
    files_read: filesRead,
    files_modified: filesModified,
  };

  const result = sessionStore.storeObservation(
    newest.memory_session_id,
    project,
    { ...observationPayload, metadata: JSON.stringify({ rotation_merge_of: cluster.map(o => o.id) }) },
    promptNumber,
    0
  );

  for (const obs of cluster) {
    sessionStore.markObservationSuperseded(obs.id, result.id);
  }

  if (chromaSync) {
    try {
      await chromaSync.syncObservation(
        result.id,
        newest.memory_session_id,
        project,
        observationPayload,
        promptNumber ?? 0,
        result.createdAtEpoch
      );
    } catch (error) {
      logger.warn('DEDUP', 'Chroma sync failed for merged observation (SQLite write already committed)', {
        id: result.id,
        error: error instanceof Error ? error.message : String(error)
      });
    }
  }

  logger.info('DEDUP', 'Merged observation cluster', {
    project,
    newObservationId: result.id,
    mergedIds: cluster.map(o => o.id),
    viaLLM: llmDecision !== null
  });
}

/**
 * Run the rotation dedup pass for a single project. Never throws for
 * per-observation/per-cluster failures (Chroma query errors, LLM failures) —
 * those are logged and counted as skipped/rejected so one bad observation
 * can't abort the whole project's run.
 */
export async function runRotationJob(dbManager: DatabaseManager, project: string): Promise<RotationSummary> {
  const summary: RotationSummary = {
    project,
    clustersConsidered: 0,
    autoMerged: 0,
    llmMerged: 0,
    llmRejected: 0,
    skipped: 0,
  };

  const chromaSync = dbManager.getChromaSync();
  if (!chromaSync) {
    logger.warn('DEDUP', 'Chroma not available; rotation job requires embedding similarity, skipping project', { project });
    return summary;
  }

  const sessionStore = dbManager.getSessionStore();
  const settings = SettingsDefaultsManager.loadFromFile(USER_SETTINGS_PATH);
  const automergeThreshold = parseFloat(settings.CLAUDE_MEM_ROTATION_AUTOMERGE_THRESHOLD);
  const candidateThreshold = parseFloat(settings.CLAUDE_MEM_ROTATION_CANDIDATE_THRESHOLD);

  const observations = sessionStore.getActiveObservationsForProject(project) as unknown as RotationObservationRow[];
  if (observations.length < 2) {
    return summary;
  }

  const byId = new Map(observations.map(o => [o.id, o]));
  const visited = new Set<number>();

  for (const obs of observations) {
    if (visited.has(obs.id)) continue;
    visited.add(obs.id);

    const queryText = buildObservationQueryText(obs);
    if (!queryText) {
      summary.skipped++;
      continue;
    }

    let neighbors: { ids: number[]; distances: number[] };
    try {
      neighbors = await chromaSync.queryChroma(queryText, NEIGHBOR_QUERY_LIMIT, {
        $and: [{ doc_type: 'observation' }, { project }]
      });
    } catch (error) {
      logger.warn('DEDUP', 'Chroma query failed during clustering; skipping observation', {
        project,
        observationId: obs.id,
        error: error instanceof Error ? error.message : String(error)
      });
      summary.skipped++;
      continue;
    }

    const candidates: Array<{ obs: RotationObservationRow; similarity: number }> = [];
    for (let i = 0; i < neighbors.ids.length; i++) {
      const neighborId = neighbors.ids[i];
      if (neighborId === obs.id || visited.has(neighborId)) continue;

      const neighborObs = byId.get(neighborId);
      // Not in the active (non-superseded) set for this project — skip.
      if (!neighborObs) continue;

      const similarity = distanceToSimilarity(neighbors.distances[i]);
      if (similarity < candidateThreshold) continue;

      candidates.push({ obs: neighborObs, similarity });
    }

    // Cluster of 1 (no candidates above threshold) — nothing to merge.
    if (candidates.length === 0) {
      summary.skipped++;
      continue;
    }

    candidates.sort((a, b) => b.similarity - a.similarity);
    const clusterMembers = candidates.slice(0, MAX_CLUSTER_SIZE - 1);
    for (const c of clusterMembers) visited.add(c.obs.id);

    const cluster = [obs, ...clusterMembers.map(c => c.obs)];
    summary.clustersConsidered++;

    const allAboveAutomerge = clusterMembers.every(c => c.similarity >= automergeThreshold);

    if (allAboveAutomerge) {
      await mergeCluster(dbManager, project, cluster, null);
      summary.autoMerged++;
      continue;
    }

    const decision = await askLLMToMergeCluster(cluster);
    if (decision && decision.shouldMerge) {
      await mergeCluster(dbManager, project, cluster, decision);
      summary.llmMerged++;
    } else {
      summary.llmRejected++;
    }
  }

  return summary;
}
