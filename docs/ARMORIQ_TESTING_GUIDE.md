# Testing ArmorIQ + OpenClaw Integration

## Quick Reference

| Method | Best For | Setup Complexity |
|--------|----------|------------------|
| CLI Agent | Quick tests, debugging | ⭐ Easy |
| HTTP API | SDK integration testing | ⭐⭐ Medium |
| WhatsApp/Telegram | Real user workflows | ⭐⭐⭐ Advanced |
| Gateway Mode | Production-like testing | ⭐⭐⭐ Advanced |

---

## 1. CLI Agent (Quickest Method)

```bash
cd /Users/arunkumarv/Documents/Customer_ArmorIQ/aiq-openclaw

node openclaw.mjs agent -m "List files in current directory" --session-id test-123
```

**What This Tests:**
- Plan capture in `before_agent_start`
- Intent token issuance from CSRG-IAP
- Tool execution with enforcement in `before_tool_call`
- Full ArmorIQ verification flow

**Verification Points:**
```
[plugins] IAP Verification Service initialized
[plugins] CSRG Verification URL: https://csrg-customer-77dabykria-uc.a.run.app
Plan captured with N steps
Intent token issued: id=xxx, plan_hash=xxx, expires=60.0s
```

---

## 2. HTTP API Testing

Start the gateway first:

```bash
cd /Users/arunkumarv/Documents/Customer_ArmorIQ/aiq-openclaw
node openclaw.mjs gateway --port 18789 --verbose
```

Then make HTTP requests:

```bash
curl -X POST http://localhost:18789/tools/invoke \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <intent_token>" \
  -H "X-CSRG-Path: /steps/[0]/action" \
  -H "X-CSRG-Proof: [...]" \
  -d '{
    "method": "tools/call",
    "params": {
      "name": "list_dir",
      "arguments": { "path": "/tmp" }
    }
  }'
```

---

## 3. WhatsApp Integration

### Setup
1. Run onboarding: `openclaw onboard`
2. Link WhatsApp: `openclaw channels login`
3. Configure allowlist in `~/.openclaw/openclaw.json`:

```json
{
  "channels": {
    "whatsapp": {
      "enabled": true,
      "allowFrom": ["+1234567890"]
    }
  }
}
```

### Test
1. Start gateway: `openclaw gateway`
2. Send message from allowed number
3. Bot responds with ArmorIQ verification active

---

## 4. Telegram Integration

### Setup
1. Create bot via @BotFather
2. Add token to config:

```json
{
  "channels": {
    "telegram": {
      "botToken": "123456:ABCDEF...",
      "allowFrom": ["*"]
    }
  }
}
```

### Test
```bash
openclaw gateway
```
Then message your bot on Telegram.

---

## 5. Gateway Mode (Production-like)

```bash
# Terminal 1: Start gateway
node openclaw.mjs gateway --port 18789 --verbose

# Terminal 2: Run agent through gateway
node openclaw.mjs agent -m "Your message" --gateway ws://localhost:18789
```

---

## 6. Intent Drift Test Script

Run the live test that verifies ArmorIQ blocks unauthorized tools:

```bash
cd /Users/arunkumarv/Documents/Customer_ArmorIQ/aiq-openclaw
node extensions/armoriq/test-intent-drift.mjs
```

**Expected Output:**
```
✅ write_file BLOCKED (as expected - intent drift protection)
✅ bash BLOCKED (as expected - intent drift protection)
```

---

## 7. Unit Tests

```bash
cd /Users/arunkumarv/Documents/Customer_ArmorIQ/aiq-openclaw
npx vitest run extensions/armoriq/armoriq-intent-enforcement.test.ts
```

**21 Tests Covering:**
- Tool allowlist enforcement
- Token expiry
- Missing plan handling
- Session ID resolution (the fix)
- Attack scenarios (prompt injection, privilege escalation)

---

## URLs & Endpoints

| Service | URL | Purpose |
|---------|-----|---------|
| CSRG | https://csrg-customer-77dabykria-uc.a.run.app | Cryptographic verification |
| IAP | https://customer-iap.armoriq.ai | Token issuance |
| Proxy | https://customer-proxy.armoriq.ai | MCP routing |
| Backend | https://customer-api.armoriq.ai | API/Audit |

---

## Session ID Fix Explained

**Problem:** `before_agent_start` cached the plan with key `"test-123"` but `before_tool_call` looked for `"test-123::test-123"`.

**Fix in `resolveRunKey()`:**
```typescript
if (runId) {
  if (sessionKey && sessionKey !== runId) {
    return `${sessionKey}::${runId}`;  // Different: combine them
  }
  return runId;  // Same: use just runId ← THE FIX
}
```

**Before:** Cache miss → "intent plan missing"  
**After:** Cache hit → Tool executes with verification

---

## Troubleshooting

### "ArmorIQ intent plan missing for this run"
- Check session ID is consistent
- Verify `before_agent_start` hook fired
- Check plan was captured: `Plan captured with N steps`

### "Token issuance failed"
- Verify API key is valid
- Check network connectivity to customer-iap.armoriq.ai
- Verify user/agent/context IDs are set

### "CSRG proof headers missing"
- SDK should include X-CSRG-Path, X-CSRG-Proof headers
- Check `stepProofs` array in intent token

### Gateway connection failed
- Start gateway: `node openclaw.mjs gateway`
- Check port 18789 is free
- Fallback to embedded mode is normal for local testing
