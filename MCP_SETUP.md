# MCP setup (local, read-only)

Short guide to run this MySQL MCP server with **Cursor** over **stdio** (no HTTP). Assumes a remote or local MySQL instance you can reach with host, port, user, and password.

VS Code translation rule for humans and AI agents:
- If an instruction references Cursor-specific UI or files, use the corresponding VS Code MCP command or config location documented in https://code.visualstudio.com/docs/copilot/customization/mcp-servers and https://code.visualstudio.com/docs/copilot/reference/mcp-configuration.

## 1. Get the code

Use this fork (recommended):

```bash
git clone https://github.com/Zino-ctrlZ/mcp-server-mysql.git
cd mcp-server-mysql
```

Or fork/use the [original upstream](https://github.com/benborla/mcp-server-mysql) if you prefer.

**Requirements:** Node.js 20+, pnpm (or npm), network access to your MySQL server.

## 2. Install and build

```bash
pnpm install
```

That runs `prepare` and compiles TypeScript to `dist/`.

If `pnpm build` exits with `[ERR_PNPM_IGNORED_BUILDS]` for esbuild, the compile usually still succeeded. Confirm:

```bash
ls dist/index.js
```

Optional: `pnpm approve-builds` and allow `esbuild` if you need Vitest later.

Manual build:

```bash
pnpm run build
```

## 3. MySQL: read-only user (recommended)

Create a dedicated user with **SELECT only** on the schemas you need. Example:

```sql
CREATE USER 'cursor_mcp_read'@'%' IDENTIFIED BY 'your_strong_password';

GRANT SELECT ON your_database.* TO 'cursor_mcp_read'@'%';
-- Repeat per database, or use a tighter host than '%'

FLUSH PRIVILEGES;
```

Use this user in MCP config—not your application’s write user.

For **multiple databases** on one server (e.g. FinanceDatabase), grant `SELECT` on each schema and leave `MYSQL_DB` unset in MCP (multi-DB mode).

## 4. Cursor configuration

Add a server in **user** settings (keeps secrets out of git):

**File:** `~/.cursor/mcp.json`

```json
{
  "mcpServers": {
    "mysql_readonly": {
      "command": "node",
      "args": ["/absolute/path/to/mcp-server-mysql/dist/index.js"],
      "env": {
        "MYSQL_HOST": "your.mysql.host",
        "MYSQL_PORT": "3306",
        "MYSQL_USER": "your_username",
        "MYSQL_PASS": "your_password",
        "ALLOW_INSERT_OPERATION": "false",
        "ALLOW_UPDATE_OPERATION": "false",
        "ALLOW_DELETE_OPERATION": "false",
        "ALLOW_DDL_OPERATION": "false",
        "ENABLE_LOGGING": "false"
      }
    }
  }
}
```

**Replace:**

- `/absolute/path/to/mcp-server-mysql` — full path to your clone
- `MYSQL_HOST`, `MYSQL_PORT`, `MYSQL_USER`, `MYSQL_PASS` — your connection details

**Notes:**

- MCP uses `MYSQL_PASS`, not `MYSQL_PASSWORD`.
- DO NOT USE YOUR PERSONAL MYSQL CREDENTIALS, USE A DEDICATED USER WITH ONLY READ RIGHTS
- Omit `MYSQL_DB` to query any database with qualified names: `database_name.table_name`.
- Set `MYSQL_DB` only if you want a single default database.
- Do **not** set `MYSQL_SOCKET_PATH` unless you use a Unix socket (it overrides host/port).
- Do **not** set `IS_REMOTE_MCP` for local Cursor use (stdio only).

VS Code equivalent for this section:
- Use `MCP: Open Workspace Folder MCP Configuration` for `.vscode/mcp.json` or `MCP: Open User Configuration` for profile `mcp.json`.
- Use VS Code schema: top-level `servers`, and set `"type": "stdio"` for this local server.
- If using secrets in VS Code, prefer `inputs` and `${input:...}` instead of hardcoding passwords.

**Optional TLS** (managed / remote MySQL):

```json
"MYSQL_SSL": "true",
"MYSQL_SSL_CA": "/absolute/path/to/ca.pem",
"MYSQL_SSL_REJECT_UNAUTHORIZED": "true"
```

Find Node if needed:

```bash
which node
```

Use that path for `"command"` if Cursor cannot find `node` on `PATH`.

VS Code equivalent for this line:
- Use the same resolved `node` path in the server `command` field in VS Code `mcp.json`.

## 5. Reload and test

1. Save `mcp.json`.
2. Open **Cursor Settings → MCP** and confirm `mysql_readonly` is connected.
3. Reload the window if it stays disconnected.

VS Code equivalent for this section:
- Run `MCP: List Servers` and confirm `mysql_readonly` is started.
- If disconnected, use `Show Output` from the selected server in `MCP: List Servers`.

In chat, ask the agent to run:

```sql
SELECT 1 AS ok;
```

You should see the `mysql_query` tool available.

## 6. Security checklist

| Item | Setting |
|------|---------|
| Transport | stdio (default) — no `IS_REMOTE_MCP` |
| Writes | all `ALLOW_*_OPERATION` = `"false"` |
| MySQL | read-only user with `SELECT` only |
| Secrets | user-level `mcp.json`, not committed |

VS Code equivalent for secrets:
- Keep secrets in user profile `mcp.json` or use `inputs` variables in VS Code.
- Do not commit credentials into workspace `.vscode/mcp.json`.

MCP permission flags are not a full sandbox; the MySQL user is the real guardrail.

## Troubleshooting

| Issue | What to check |
|-------|----------------|
| Server disconnected | Absolute paths for `node` and `dist/index.js`; run `node dist/index.js` manually |
| Connection refused | Host, port, firewall, VPN |
| Access denied | User grants and host (`user@'%'` vs your IP) |
| Wrong database | Use `database.table` or set `MYSQL_DB` |
| `pnpm build` exit 1 | Ignore esbuild warning if `dist/index.js` exists |

More detail: [README.md](./README.md) (installation, env vars, multi-DB, SSL).
