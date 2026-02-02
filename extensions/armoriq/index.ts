import type { Api, Model } from "@mariozechner/pi-ai";
import type { ModelRegistry } from "@mariozechner/pi-coding-agent";
import type { OpenClawPluginApi } from "openclaw/plugin-sdk";
import { ArmorIQClient } from "@armoriq/sdk";
import { completeSimple } from "@mariozechner/pi-ai";
import { IAPVerificationService, type CsrgProofHeaders } from "./src/iap-verfication.service.js";

type ArmorIqConfig = {
  enabled: boolean;
  apiKey?: string;
  userId?: string;
  agentId?: string;
  contextId?: string;
  userIdSource?:
    | "senderE164"
    | "senderId"
    | "senderUsername"
    | "senderName"
    | "sessionKey"
    | "agentId";
  agentIdSource?: "agentId" | "sessionKey";
  contextIdSource?: "sessionKey" | "agentId" | "channel" | "accountId";
  policy?: Record<string, unknown>;
  validitySeconds: number;
  useProduction?: boolean;
  iapEndpoint?: string;
  proxyEndpoint?: string;
  backendEndpoint?: string;
  proxyEndpoints?: Record<string, string>;
  timeoutMs?: number;
  maxRetries?: number;
  verifySsl?: boolean;
  maxParamChars: number;
  maxParamDepth: number;
  maxParamKeys: number;
  maxParamItems: number;
};

type ToolContext = {
  agentId?: string;
  sessionKey?: string;
  messageChannel?: string;
  accountId?: string;
  senderId?: string;
  senderName?: string;
  senderUsername?: string;
  senderE164?: string;
  runId?: string;
  model?: Model<Api> | null;
  modelRegistry?: ModelRegistry | null;
  intentTokenRaw?: string;
  csrgPath?: string;
  csrgProofRaw?: string;
  csrgValueDigest?: string;
};

type IdentityBundle = {
  userId: string;
  agentId: string;
  contextId: string;
};

type PlanCacheEntry = {
  token: unknown;
  plan: Record<string, unknown>;
  allowedActions: Set<string>;
  createdAt: number;
  expiresAt?: number;
  error?: string;
};

const DEFAULT_VALIDITY_SECONDS = 60;
const DEFAULT_MAX_PARAM_CHARS = 2000;
const DEFAULT_MAX_PARAM_DEPTH = 4;
const DEFAULT_MAX_PARAM_KEYS = 50;
const DEFAULT_MAX_PARAM_ITEMS = 50;

const clientCache = new Map<string, ArmorIQClient>();
const planCache = new Map<string, PlanCacheEntry>();

function readString(value: unknown): string | undefined {
  return typeof value === "string" ? value.trim() || undefined : undefined;
}

function readNumber(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function readBoolean(value: unknown): boolean | undefined {
  return typeof value === "boolean" ? value : undefined;
}

function readRecord(value: unknown): Record<string, unknown> | undefined {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return undefined;
  }
  return value as Record<string, unknown>;
}

function resolveConfig(api: OpenClawPluginApi): ArmorIqConfig {
  const raw = readRecord(api.pluginConfig) ?? {};
  const enabled = readBoolean(raw.enabled) ?? false;
  return {
    enabled,
    apiKey: readString(raw.apiKey) ?? readString(process.env.ARMORIQ_API_KEY),
    userId: readString(raw.userId) ?? readString(process.env.USER_ID),
    agentId: readString(raw.agentId) ?? readString(process.env.AGENT_ID),
    contextId: readString(raw.contextId) ?? readString(process.env.CONTEXT_ID),
    userIdSource: readString(raw.userIdSource) as ArmorIqConfig["userIdSource"],
    agentIdSource: readString(raw.agentIdSource) as ArmorIqConfig["agentIdSource"],
    contextIdSource: readString(raw.contextIdSource) as ArmorIqConfig["contextIdSource"],
    policy: readRecord(raw.policy),
    validitySeconds: readNumber(raw.validitySeconds) ?? DEFAULT_VALIDITY_SECONDS,
    useProduction: readBoolean(raw.useProduction),
    iapEndpoint: readString(raw.iapEndpoint) ?? readString(process.env.IAP_ENDPOINT),
    proxyEndpoint: readString(raw.proxyEndpoint) ?? readString(process.env.PROXY_ENDPOINT),
    backendEndpoint: readString(raw.backendEndpoint) ?? readString(process.env.BACKEND_ENDPOINT),
    proxyEndpoints: readRecord(raw.proxyEndpoints) as Record<string, string> | undefined,
    timeoutMs: readNumber(raw.timeoutMs),
    maxRetries: readNumber(raw.maxRetries),
    verifySsl: readBoolean(raw.verifySsl),
    maxParamChars: readNumber(raw.maxParamChars) ?? DEFAULT_MAX_PARAM_CHARS,
    maxParamDepth: readNumber(raw.maxParamDepth) ?? DEFAULT_MAX_PARAM_DEPTH,
    maxParamKeys: readNumber(raw.maxParamKeys) ?? DEFAULT_MAX_PARAM_KEYS,
    maxParamItems: readNumber(raw.maxParamItems) ?? DEFAULT_MAX_PARAM_ITEMS,
  };
}

function resolveUserId(cfg: ArmorIqConfig, ctx: ToolContext): string | undefined {
  if (cfg.userId) {
    return cfg.userId;
  }
  const source = cfg.userIdSource;
  if (source === "senderE164") {
    return ctx.senderE164?.trim();
  }
  if (source === "senderId") {
    return ctx.senderId?.trim();
  }
  if (source === "senderUsername") {
    return ctx.senderUsername?.trim();
  }
  if (source === "senderName") {
    return ctx.senderName?.trim();
  }
  if (source === "sessionKey") {
    return ctx.sessionKey?.trim();
  }
  if (source === "agentId") {
    return ctx.agentId?.trim();
  }

  return (
    ctx.senderE164?.trim() ||
    ctx.senderId?.trim() ||
    ctx.senderUsername?.trim() ||
    ctx.senderName?.trim() ||
    ctx.sessionKey?.trim() ||
    ctx.agentId?.trim()
  );
}

function resolveAgentId(cfg: ArmorIqConfig, ctx: ToolContext): string | undefined {
  if (cfg.agentId) {
    return cfg.agentId;
  }
  const source = cfg.agentIdSource;
  if (source === "sessionKey") {
    return ctx.sessionKey?.trim();
  }
  return ctx.agentId?.trim();
}

function resolveContextId(cfg: ArmorIqConfig, ctx: ToolContext): string | undefined {
  if (cfg.contextId) {
    return cfg.contextId;
  }
  const source = cfg.contextIdSource;
  if (source === "agentId") {
    return ctx.agentId?.trim();
  }
  if (source === "channel") {
    return ctx.messageChannel?.trim();
  }
  if (source === "accountId") {
    return ctx.accountId?.trim();
  }
  return ctx.sessionKey?.trim();
}

function resolveIdentities(cfg: ArmorIqConfig, ctx: ToolContext): IdentityBundle | null {
  const userId = resolveUserId(cfg, ctx);
  const agentId = resolveAgentId(cfg, ctx);
  const contextId = resolveContextId(cfg, ctx) ?? "default";
  if (!userId || !agentId) {
    return null;
  }
  return { userId, agentId, contextId };
}

function resolveRunKey(ctx: ToolContext): string | null {
  const runId = ctx.runId?.trim();
  if (runId) {
    const sessionKey = ctx.sessionKey?.trim();
    return sessionKey ? `${sessionKey}::${runId}` : runId;
  }
  return null;
}

function normalizeToolName(value: string): string {
  return value.trim().toLowerCase();
}

function parseCsrgProofHeaders(ctx: ToolContext): {
  proofs?: CsrgProofHeaders;
  error?: string;
} {
  const path = readString(ctx.csrgPath);
  const valueDigest = readString(ctx.csrgValueDigest);
  const proofRaw = readString(ctx.csrgProofRaw);
  if (!path && !valueDigest && !proofRaw) {
    return {};
  }

  let proof: unknown = undefined;
  if (proofRaw) {
    try {
      proof = JSON.parse(proofRaw);
    } catch {
      return { error: "ArmorIQ CSRG proof header invalid JSON" };
    }
    if (!Array.isArray(proof)) {
      return { error: "ArmorIQ CSRG proof header must be a JSON array" };
    }
  }

  return { proofs: { path, valueDigest, proof } };
}

function validateCsrgProofHeaders(
  proofs: CsrgProofHeaders | undefined,
  required: boolean,
): string | null {
  if (!required) {
    return null;
  }
  if (!proofs) {
    return "ArmorIQ CSRG proof headers missing";
  }
  if (!proofs.path) {
    return "ArmorIQ CSRG path header missing";
  }
  if (!proofs.valueDigest) {
    return "ArmorIQ CSRG value digest header missing";
  }
  if (!proofs.proof || !Array.isArray(proofs.proof)) {
    return "ArmorIQ CSRG proof header missing";
  }
  return null;
}

function extractAllowedActions(plan: Record<string, unknown>): Set<string> {
  const allowed = new Set<string>();
  const steps = Array.isArray(plan.steps) ? plan.steps : [];
  for (const step of steps) {
    if (!step || typeof step !== "object") {
      continue;
    }
    const action =
      typeof (step as { action?: unknown }).action === "string"
        ? String((step as { action?: unknown }).action)
        : typeof (step as { tool?: unknown }).tool === "string"
          ? String((step as { tool?: unknown }).tool)
          : "";
    if (action.trim()) {
      allowed.add(normalizeToolName(action));
    }
  }
  return allowed;
}

function extractPlanFromIntentToken(raw: string): {
  plan: Record<string, unknown>;
  expiresAt?: number;
} | null {
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch {
    // TODO(armoriq): Support base64-encoded token payloads.
    return null;
  }
  if (!parsed || typeof parsed !== "object") {
    return null;
  }
  const tokenObj = parsed as Record<string, unknown>;
  const rawToken = tokenObj.rawToken as Record<string, unknown> | undefined;
  const planCandidate =
    (rawToken && typeof rawToken.plan === "object"
      ? (rawToken.plan as Record<string, unknown>)
      : undefined) ||
    (typeof tokenObj.plan === "object" ? (tokenObj.plan as Record<string, unknown>) : undefined) ||
    (typeof (tokenObj.token as Record<string, unknown> | undefined)?.plan === "object"
      ? ((tokenObj.token as Record<string, unknown>).plan as Record<string, unknown>)
      : undefined);
  if (!planCandidate) {
    return null;
  }
  const expiresAt =
    typeof tokenObj.expiresAt === "number"
      ? tokenObj.expiresAt
      : typeof (tokenObj.token as Record<string, unknown> | undefined)?.expires_at === "number"
        ? ((tokenObj.token as Record<string, unknown>).expires_at as number)
        : undefined;
  return { plan: planCandidate, expiresAt };
}

function buildToolList(tools?: Array<{ name: string; description?: string }>): string {
  if (!tools || tools.length === 0) {
    return "- (no tools available)";
  }
  const lines: string[] = [];
  for (const tool of tools) {
    const name = tool.name?.trim();
    if (!name) {
      continue;
    }
    const description = tool.description?.trim();
    lines.push(description ? `- ${name}: ${description}` : `- ${name}`);
  }
  return lines.length > 0 ? lines.join("\n") : "- (no tools available)";
}

function findPlanStep(
  plan: Record<string, unknown>,
  toolName: string,
): Record<string, unknown> | null {
  const steps = Array.isArray(plan.steps) ? plan.steps : [];
  const normalizedTool = normalizeToolName(toolName);
  for (const step of steps) {
    if (!step || typeof step !== "object") {
      continue;
    }
    const action =
      typeof (step as { action?: unknown }).action === "string"
        ? String((step as { action?: unknown }).action)
        : typeof (step as { tool?: unknown }).tool === "string"
          ? String((step as { tool?: unknown }).tool)
          : "";
    if (normalizeToolName(action) === normalizedTool) {
      return step as Record<string, unknown>;
    }
  }
  return null;
}

function isParamsAllowedByPlan(
  _step: Record<string, unknown>,
  _params: Record<string, unknown>,
): boolean {
  // TODO(armoriq): Enforce parameter-level intent by comparing call params against step metadata inputs.
  // This should support placeholders or allowlists of fields to avoid blocking dynamic results.
  return true;
}

async function buildPlanFromPrompt(params: {
  prompt: string;
  tools?: Array<{ name: string; description?: string }>;
  model?: Model<Api> | null;
  modelRegistry?: ModelRegistry | null;
  log: (message: string) => void;
}): Promise<Record<string, unknown>> {
  const toolDescriptions = new Map<string, string>();
  for (const tool of params.tools ?? []) {
    const name = tool.name?.trim();
    const description = tool.description?.trim();
    if (name && description) {
      toolDescriptions.set(normalizeToolName(name), description);
    }
  }
  if (!params.model || !params.modelRegistry) {
    throw new Error("Missing model context for planning");
  }
  const apiKey = await params.modelRegistry.getApiKey(params.model);
  if (!apiKey) {
    throw new Error("No API key available for planning model");
  }

  const toolList = buildToolList(params.tools);
  const planningPrompt =
    `You are a planning assistant. Produce a JSON plan for the user's request.\n` +
    `Rules:\n` +
    `- Output ONLY valid JSON.\n` +
    `- Use the tool names exactly as given.\n` +
    `- Create a sequence of tool calls needed to satisfy the request.\n` +
    `- If no tools are needed, return an empty steps array.\n` +
    `- Every step MUST include: { action, mcp }.\n` +
    `- Use mcp="openclaw" for all steps.\n\n` +
    `Available tools:\n${toolList}\n\n` +
    `User request:\n${params.prompt}\n\n` +
    `Return JSON with shape:\n` +
    `{\n  "steps": [ { "action": "tool_name", "mcp": "openclaw", "description": "...", "metadata": { } } ],\n  "metadata": { "goal": "..." }\n}\n`;
  // TODO(armoriq): Include tool parameter schemas in the planning prompt when size allows.

  params.log(`armoriq: planning with model ${params.model.provider}/${params.model.id}`);

  const response = await completeSimple(
    params.model as never,
    {
      messages: [
        {
          role: "user",
          content: planningPrompt,
          timestamp: Date.now(),
        },
      ],
    },
    {
      apiKey,
      maxTokens: 512,
      temperature: 0.2,
    },
  );

  const text =
    typeof response.content === "string"
      ? response.content.trim()
      : Array.isArray(response.content)
        ? response.content
            .filter((block) => block && (block as { type?: string }).type === "text")
            .map((block) => (block as { text?: string }).text ?? "")
            .join(" ")
            .trim()
        : "";

  if (!text) {
    throw new Error("Planner returned empty response");
  }

  try {
    const parsed = JSON.parse(text) as Record<string, unknown>;
    if (!parsed.steps || !Array.isArray(parsed.steps)) {
      parsed.steps = [];
    }
    if (!parsed.metadata || typeof parsed.metadata !== "object" || Array.isArray(parsed.metadata)) {
      parsed.metadata = { goal: params.prompt };
    }
    for (const step of parsed.steps) {
      if (!step || typeof step !== "object") {
        continue;
      }
      const stepObj = step as Record<string, unknown>;
      if (!stepObj.action && typeof stepObj.tool === "string") {
        stepObj.action = stepObj.tool;
      }
      if (!stepObj.mcp || typeof stepObj.mcp !== "string") {
        stepObj.mcp = "openclaw";
      }
      if (!stepObj.description && typeof stepObj.action === "string") {
        const description = toolDescriptions.get(normalizeToolName(stepObj.action));
        if (description) {
          stepObj.description = description;
        }
      }
    }
    return parsed;
  } catch (err) {
    throw new Error(`Planner returned invalid JSON: ${err instanceof Error ? err.message : err}`, {
      cause: err,
    });
  }
}

function buildClientKey(cfg: ArmorIqConfig, ids: IdentityBundle): string {
  return [
    cfg.apiKey,
    ids.userId,
    ids.agentId,
    ids.contextId,
    cfg.iapEndpoint,
    cfg.proxyEndpoint,
    cfg.backendEndpoint,
    cfg.useProduction ? "prod" : "dev",
  ]
    .filter(Boolean)
    .join("|");
}

function getClient(cfg: ArmorIqConfig, ids: IdentityBundle): ArmorIQClient {
  const key = buildClientKey(cfg, ids);
  const cached = clientCache.get(key);
  if (cached) {
    return cached;
  }

  const client = new ArmorIQClient({
    apiKey: cfg.apiKey,
    userId: ids.userId,
    agentId: ids.agentId,
    contextId: ids.contextId,
    useProduction: cfg.useProduction,
    iapEndpoint: cfg.iapEndpoint,
    proxyEndpoint: cfg.proxyEndpoint,
    backendEndpoint: cfg.backendEndpoint,
    proxyEndpoints: cfg.proxyEndpoints,
    timeout: cfg.timeoutMs,
    maxRetries: cfg.maxRetries,
    verifySsl: cfg.verifySsl,
  });
  clientCache.set(key, client);
  return client;
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function sanitizeValue(
  value: unknown,
  opts: {
    maxChars: number;
    maxDepth: number;
    maxKeys: number;
    maxItems: number;
  },
  depth: number,
): unknown {
  if (depth > opts.maxDepth) {
    return "<max-depth>";
  }
  if (value == null) {
    return value;
  }
  if (typeof value === "string") {
    if (value.length <= opts.maxChars) {
      return value;
    }
    return `${value.slice(0, opts.maxChars)}...`;
  }
  if (typeof value === "number" || typeof value === "boolean") {
    return value;
  }
  if (typeof value === "bigint") {
    return value.toString();
  }
  if (typeof value === "symbol") {
    return value.toString();
  }
  if (typeof value === "function") {
    return "<function>";
  }
  if (value instanceof Uint8Array) {
    return `<binary:${value.length}>`;
  }
  if (Array.isArray(value)) {
    return value.slice(0, opts.maxItems).map((entry) => sanitizeValue(entry, opts, depth + 1));
  }
  if (isPlainObject(value)) {
    const entries = Object.entries(value).slice(0, opts.maxKeys);
    const next: Record<string, unknown> = {};
    for (const [key, entry] of entries) {
      next[key] = sanitizeValue(entry, opts, depth + 1);
    }
    return next;
  }
  try {
    return JSON.parse(JSON.stringify(value));
  } catch {
    return "<unserializable>";
  }
}

function sanitizeParams(
  params: Record<string, unknown>,
  cfg: ArmorIqConfig,
): Record<string, unknown> {
  const sanitized = sanitizeValue(
    params,
    {
      maxChars: cfg.maxParamChars,
      maxDepth: cfg.maxParamDepth,
      maxKeys: cfg.maxParamKeys,
      maxItems: cfg.maxParamItems,
    },
    0,
  );
  return isPlainObject(sanitized) ? sanitized : {};
}

export default function register(api: OpenClawPluginApi) {
  const cfg = resolveConfig(api);

  if (!cfg.enabled) {
    api.logger.info("armoriq: plugin disabled (set plugins.entries.armoriq.enabled=true)");
    return;
  }

  const verificationService = new IAPVerificationService({
    iapBaseUrl: cfg.backendEndpoint ?? cfg.iapEndpoint,
    timeoutMs: cfg.timeoutMs,
    logger: api.logger,
  });

  api.on("before_agent_start", async (event, ctx) => {
    const runKey = resolveRunKey(ctx as ToolContext);
    if (!runKey) {
      return;
    }
    if (planCache.has(runKey)) {
      return;
    }

    const identity = resolveIdentities(cfg, ctx as ToolContext);
    if (!identity) {
      planCache.set(runKey, {
        token: null,
        plan: { steps: [], metadata: { goal: "invalid" } },
        allowedActions: new Set(),
        createdAt: Date.now(),
        error: "ArmorIQ identity missing (userId/agentId)",
      });
      return;
    }

    try {
      const plan = await buildPlanFromPrompt({
        prompt: event.prompt,
        tools: event.tools,
        model: ctx.model ?? null,
        modelRegistry: ctx.modelRegistry ?? null,
        log: (message) => api.logger.info(message),
      });

      const client = getClient(cfg, identity);
      const planCapture = client.capturePlan("openclaw", event.prompt, plan, {
        sessionKey: ctx.sessionKey,
        messageChannel: ctx.messageChannel,
        accountId: ctx.accountId,
        senderId: ctx.senderId,
        senderName: ctx.senderName,
        senderUsername: ctx.senderUsername,
        senderE164: ctx.senderE164,
        runId: ctx.runId,
      });
      const token = await client.getIntentToken(planCapture, cfg.policy, cfg.validitySeconds);
      planCache.set(runKey, {
        token,
        plan,
        allowedActions: extractAllowedActions(plan),
        createdAt: Date.now(),
        expiresAt: typeof token.expiresAt === "number" ? token.expiresAt : undefined,
      });
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      planCache.set(runKey, {
        token: null,
        plan: { steps: [], metadata: { goal: "invalid" } },
        allowedActions: new Set(),
        createdAt: Date.now(),
        error: `ArmorIQ planning failed: ${message}`,
      });
    }
  });

  api.on("agent_end", async (_event, ctx) => {
    const runKey = resolveRunKey(ctx as ToolContext);
    if (runKey) {
      planCache.delete(runKey);
    }
  });

  api.on("before_tool_call", async (event, ctx) => {
    const normalizedTool = normalizeToolName(event.toolName);
    const toolCtx = ctx as ToolContext;
    const intentTokenRaw = readString(toolCtx.intentTokenRaw);

    if (intentTokenRaw) {
      const parsed = extractPlanFromIntentToken(intentTokenRaw);
      if (parsed) {
        if (parsed.expiresAt && Date.now() / 1000 > parsed.expiresAt) {
          return {
            block: true,
            blockReason: "ArmorIQ intent token expired",
          };
        }
        const allowedActions = extractAllowedActions(parsed.plan);
        if (!allowedActions.has(normalizedTool)) {
          return {
            block: true,
            blockReason: `ArmorIQ intent drift: tool not in plan (${event.toolName})`,
          };
        }
        const step = findPlanStep(parsed.plan, event.toolName);
        if (step) {
          const params = isPlainObject(event.params) ? event.params : {};
          if (!isParamsAllowedByPlan(step, params)) {
            return {
              block: true,
              blockReason: `ArmorIQ intent mismatch: parameters not allowed for ${event.toolName}`,
            };
          }
        }
        return { params: event.params };
      }

      const proofParse = parseCsrgProofHeaders(toolCtx);
      if (proofParse.error) {
        return { block: true, blockReason: proofParse.error };
      }
      const proofError = validateCsrgProofHeaders(
        proofParse.proofs,
        verificationService.csrgProofsRequired(),
      );
      if (proofError) {
        return { block: true, blockReason: proofError };
      }

      try {
        const verifyResult = await verificationService.verifyStep(
          intentTokenRaw,
          proofParse.proofs,
          event.toolName,
        );
        if (!verifyResult.allowed) {
          return {
            block: true,
            blockReason: verifyResult.reason || "ArmorIQ intent verification denied",
          };
        }
        return { params: event.params };
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        return {
          block: true,
          blockReason: `ArmorIQ intent verification failed: ${message}`,
        };
      }
    }

    if (!cfg.apiKey) {
      return { block: true, blockReason: "ArmorIQ API key missing" };
    }

    const identity = resolveIdentities(cfg, toolCtx);
    if (!identity) {
      return {
        block: true,
        blockReason: "ArmorIQ identity missing (userId/agentId)",
      };
    }

    const runKey = resolveRunKey(toolCtx);
    if (!runKey) {
      return {
        block: true,
        blockReason: "ArmorIQ run id missing",
      };
    }

    let cached = planCache.get(runKey);
    if (!cached && toolCtx.runId?.startsWith("http-")) {
      const client = getClient(cfg, identity);
      const sanitizedParams = sanitizeParams(event.params ?? {}, cfg);
      const plan = {
        steps: [
          {
            action: event.toolName,
            mcp: "openclaw",
            description: `Authorize tool ${event.toolName}`,
            metadata: { inputs: sanitizedParams },
          },
        ],
        metadata: { goal: `Authorize tool ${event.toolName}` },
      };
      try {
        const planCapture = client.capturePlan(
          "openclaw",
          `Authorize tool ${event.toolName}`,
          plan,
          {
            sessionKey: toolCtx.sessionKey,
            messageChannel: toolCtx.messageChannel,
            accountId: toolCtx.accountId,
            senderId: toolCtx.senderId,
            senderName: toolCtx.senderName,
            senderUsername: toolCtx.senderUsername,
            senderE164: toolCtx.senderE164,
            runId: toolCtx.runId,
          },
        );
        const token = await client.getIntentToken(planCapture, cfg.policy, cfg.validitySeconds);
        cached = {
          token,
          plan,
          allowedActions: extractAllowedActions(plan),
          createdAt: Date.now(),
          expiresAt: typeof token.expiresAt === "number" ? token.expiresAt : undefined,
        };
        planCache.set(runKey, cached);
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        return {
          block: true,
          blockReason: `ArmorIQ authorization failed: ${message}`,
        };
      }
    }

    if (!cached) {
      return {
        block: true,
        blockReason: "ArmorIQ intent plan missing for this run",
      };
    }

    if (cached.error) {
      return {
        block: true,
        blockReason: cached.error,
      };
    }

    if (cached.expiresAt && Date.now() / 1000 > cached.expiresAt) {
      return {
        block: true,
        blockReason: "ArmorIQ intent token expired",
      };
    }

    if (!cached.allowedActions.has(normalizedTool)) {
      return {
        block: true,
        blockReason: `ArmorIQ intent drift: tool not in plan (${event.toolName})`,
      };
    }

    const step = findPlanStep(cached.plan, event.toolName);
    if (step) {
      const params = isPlainObject(event.params) ? event.params : {};
      if (!isParamsAllowedByPlan(step, params)) {
        return {
          block: true,
          blockReason: `ArmorIQ intent mismatch: parameters not allowed for ${event.toolName}`,
        };
      }
    }

    return { params: event.params };
  });
}
