"""
MySQL connection audit and cleanup for the FinanceDatabase stack.

Audits SQLHelpers caches in this process, local TCP sockets to MYSQL_HOST,
and server-side sessions via information_schema.processlist when reachable.

Core Functions:
    cleanup_mysql_connections: Audit and optionally close local/server sessions.
    audit_mysql_connections: Report-only audit.
    dispose_local_mysql_connections: Dispose this process's engine/pymysql caches.

Processing Flow:
    1. Snapshot SQLHelpers engine/pymysql caches for this PID.
    2. lsof local ESTABLISHED TCP to MYSQL_HOST:3306 (includes other PIDs).
    3. Query processlist when MySQL is reachable; optional KILL.

Usage:
    >>> from dbase.database.connection_cleanup import cleanup_mysql_connections
    >>> cleanup_mysql_connections(dispose_local=True)
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class CachedEngineInfo:
    """SQLAlchemy engine entry held in this process's SQLHelpers cache."""

    pid: int
    database: str
    pool_status: str


@dataclass
class CachedPymysqlInfo:
    """pymysql connection entry held in this process's SQLHelpers cache."""

    pid: int
    database: str
    is_open: Optional[bool]


@dataclass
class LocalTcpConnection:
    """Established TCP session from a local process to the MySQL host."""

    pid: int
    command: str
    is_this_process: bool
    detail: str


@dataclass
class ServerMysqlSession:
    """Row from information_schema.processlist."""

    session_id: int
    user: str
    host: str
    database: Optional[str]
    command: str
    time_seconds: int
    state: Optional[str]
    info_preview: str


@dataclass
class MysqlConnectionAudit:
    """Full audit snapshot for local process, OS sockets, and server sessions."""

    this_pid: int
    cached_engines: List[CachedEngineInfo] = field(default_factory=list)
    cached_pymysql: List[CachedPymysqlInfo] = field(default_factory=list)
    local_tcp: List[LocalTcpConnection] = field(default_factory=list)
    server_sessions: List[ServerMysqlSession] = field(default_factory=list)
    server_error: Optional[str] = None
    disposed_local: bool = False
    killed_server_count: int = 0


def _mysql_connection_settings() -> Dict[str, Any]:
    """
    Load MySQL connection settings from SQLHelpers module-level env vars.

    Returns:
        Dict with host, port, user, and password keys.

    Raises:
        ValueError: If required MYSQL_* environment variables are missing.
    """
    from .SQLHelpers import sql_host, sql_port, sql_pw, sql_user

    host = sql_host or os.environ.get("MYSQL_HOST")
    user = sql_user or os.environ.get("MYSQL_USER")
    password = sql_pw or os.environ.get("MYSQL_PASSWORD")
    port_raw = sql_port or os.environ.get("MYSQL_PORT", "3306")

    missing = [
        name
        for name, value in (
            ("MYSQL_HOST", host),
            ("MYSQL_USER", user),
            ("MYSQL_PASSWORD", password),
        )
        if not value
    ]
    if missing:
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

    return {
        "host": str(host),
        "port": int(port_raw),
        "user": str(user),
        "password": str(password),
    }


def audit_local_process_connections() -> tuple[List[CachedEngineInfo], List[CachedPymysqlInfo]]:
    """
    List SQLAlchemy engines and pymysql connections cached in this Python process.

    Returns:
        Tuple of (cached engines, cached pymysql connections). Other processes
        maintain separate caches; use ``audit_local_tcp_connections`` for those.
    """
    from .SQLHelpers import _PROCESS_ENGINE_CACHE, _PYMYSQL_CONNECTION_CACHE

    engines = [
        CachedEngineInfo(
            pid=int(pid),
            database=str(db),
            pool_status=str(engine.pool.status()),
        )
        for (pid, db), engine in _PROCESS_ENGINE_CACHE.items()
    ]
    pymysql_rows = [
        CachedPymysqlInfo(
            pid=int(pid),
            database=str(db),
            is_open=getattr(conn, "open", None),
        )
        for (pid, db), conn in _PYMYSQL_CONNECTION_CACHE.items()
    ]
    return engines, pymysql_rows


def dispose_local_mysql_connections() -> None:
    """
    Dispose SQLAlchemy engines and close pymysql connections in this process.

    Calls SQLHelpers ``_dispose_all_engines``. Does not affect other PIDs or
    server-side sessions until those clients reconnect.
    """
    from .SQLHelpers import _dispose_all_engines

    _dispose_all_engines()


def audit_local_tcp_connections(
    host: Optional[str] = None,
    port: Optional[int] = None,
) -> List[LocalTcpConnection]:
    """
    List local processes with established TCP to the MySQL host.

    Uses ``lsof`` on macOS/Linux. Cannot inspect other processes' in-memory
    SQLAlchemy engines — only OS-level sockets.

    Args:
        host: MySQL host override; defaults to MYSQL_HOST.
        port: MySQL port override; defaults to MYSQL_PORT or 3306.

    Returns:
        Local TCP connection rows for all PIDs on this machine.
    """
    settings = _mysql_connection_settings()
    mysql_host = host or settings["host"]
    mysql_port = port or settings["port"]
    my_pid = os.getpid()

    try:
        output = subprocess.check_output(
            ["lsof", "-nP", f"-iTCP@{mysql_host}:{mysql_port}", "-sTCP:ESTABLISHED"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []

    rows: List[LocalTcpConnection] = []
    lines = output.strip().splitlines()
    for line in lines[1:]:
        parts = line.split()
        if len(parts) < 2:
            continue
        command, pid_text = parts[0], parts[1]
        try:
            pid = int(pid_text)
        except ValueError:
            continue
        rows.append(
            LocalTcpConnection(
                pid=pid,
                command=command,
                is_this_process=pid == my_pid,
                detail=line,
            )
        )
    return rows


def audit_server_mysql_sessions(
    user_filter: Optional[str] = None,
    host: Optional[str] = None,
    port: Optional[int] = None,
) -> tuple[List[ServerMysqlSession], Optional[str]]:
    """
    Query MySQL ``information_schema.processlist`` for client sessions.

    Args:
        user_filter: If set, only rows for this MySQL user are returned.
        host: MySQL host override; defaults to MYSQL_HOST.
        port: MySQL port override; defaults to MYSQL_PORT or 3306.

    Returns:
        Tuple of (session rows, error message). ``error`` is set when the
        server cannot be reached (e.g. firewall / VPN).
    """
    settings = _mysql_connection_settings()
    mysql_host = host or settings["host"]
    mysql_port = port or settings["port"]
    mysql_user = settings["user"]
    mysql_password = settings["password"]
    filter_user = user_filter or mysql_user

    try:
        import mysql.connector
    except ImportError as exc:
        return [], f"mysql.connector not available: {exc}"

    try:
        connection = mysql.connector.connect(
            host=mysql_host,
            port=mysql_port,
            user=mysql_user,
            password=mysql_password,
        )
    except Exception as exc:
        return [], str(exc)

    sessions: List[ServerMysqlSession] = []
    try:
        cursor = connection.cursor(dictionary=True)
        cursor.execute(
            """
            SELECT id, user, host, db, command, time, state, info
            FROM information_schema.processlist
            WHERE user != 'system user'
            ORDER BY time DESC
            """
        )
        for row in cursor.fetchall():
            if filter_user and row.get("user") != filter_user:
                continue
            info = row.get("info") or ""
            sessions.append(
                ServerMysqlSession(
                    session_id=int(row["id"]),
                    user=str(row.get("user") or ""),
                    host=str(row.get("host") or ""),
                    database=row.get("db"),
                    command=str(row.get("command") or ""),
                    time_seconds=int(row.get("time") or 0),
                    state=row.get("state"),
                    info_preview=str(info)[:80],
                )
            )
    finally:
        connection.close()

    return sessions, None


def kill_server_mysql_sessions(
    user_filter: Optional[str] = None,
    exclude_self: bool = True,
    host: Optional[str] = None,
    port: Optional[int] = None,
) -> tuple[int, Optional[str]]:
    """
    Issue ``KILL`` for MySQL sessions belonging to a user.

    Args:
        user_filter: MySQL user whose sessions are killed; defaults to MYSQL_USER.
        exclude_self: Skip the connection used by this cleanup call.
        host: MySQL host override; defaults to MYSQL_HOST.
        port: MySQL port override; defaults to MYSQL_PORT or 3306.

    Returns:
        Tuple of (killed count, error message if connect/kill failed).
    """
    settings = _mysql_connection_settings()
    mysql_host = host or settings["host"]
    mysql_port = port or settings["port"]
    mysql_user = settings["user"]
    mysql_password = settings["password"]
    target_user = user_filter or mysql_user

    try:
        import mysql.connector
    except ImportError as exc:
        return 0, f"mysql.connector not available: {exc}"

    try:
        connection = mysql.connector.connect(
            host=mysql_host,
            port=mysql_port,
            user=mysql_user,
            password=mysql_password,
        )
    except Exception as exc:
        return 0, str(exc)

    killed = 0
    try:
        cursor = connection.cursor(dictionary=True)
        cursor.execute(
            """
            SELECT id, user, host, db, command, time
            FROM information_schema.processlist
            WHERE user = %s AND command != 'Daemon'
            """,
            (target_user,),
        )
        my_connection_id = connection.connection_id if exclude_self else None
        for row in cursor.fetchall():
            session_id = int(row["id"])
            if my_connection_id is not None and session_id == my_connection_id:
                continue
            cursor.execute(f"KILL {session_id}")
            killed += 1
        connection.commit()
    except Exception as exc:
        return killed, str(exc)
    finally:
        connection.close()

    return killed, None


def audit_mysql_connections(
    user_filter: Optional[str] = None,
    host: Optional[str] = None,
    port: Optional[int] = None,
) -> MysqlConnectionAudit:
    """
    Audit MySQL-related connections without closing anything.

    Args:
        user_filter: Optional MySQL user filter for server processlist.
        host: MySQL host override; defaults to MYSQL_HOST.
        port: MySQL port override; defaults to MYSQL_PORT or 3306.

    Returns:
        ``MysqlConnectionAudit`` with this-process caches, local TCP, and server rows.
    """
    engines, pymysql_rows = audit_local_process_connections()
    local_tcp = audit_local_tcp_connections(host=host, port=port)
    server_sessions, server_error = audit_server_mysql_sessions(
        user_filter=user_filter,
        host=host,
        port=port,
    )
    return MysqlConnectionAudit(
        this_pid=os.getpid(),
        cached_engines=engines,
        cached_pymysql=pymysql_rows,
        local_tcp=local_tcp,
        server_sessions=server_sessions,
        server_error=server_error,
    )


def print_mysql_connection_audit(audit: MysqlConnectionAudit) -> None:
    """
    Print a human-readable summary of ``MysqlConnectionAudit``.

    Args:
        audit: Audit result from ``audit_mysql_connections`` or ``cleanup_mysql_connections``.
    """
    settings = _mysql_connection_settings()
    print(f"=== This process (PID {audit.this_pid}) ===")
    print(f"  SQLAlchemy engines cached: {len(audit.cached_engines)}")
    for row in audit.cached_engines:
        print(f"    PID {row.pid} db={row.database} pool={row.pool_status}")
    print(f"  pymysql connections cached: {len(audit.cached_pymysql)}")
    for row in audit.cached_pymysql:
        print(f"    PID {row.pid} db={row.database} open={row.is_open}")

    print(f"\n=== Local TCP to {settings['host']}:{settings['port']} ===")
    if not audit.local_tcp:
        print("  (no established connections)")
    for row in audit.local_tcp:
        tag = "THIS PROCESS" if row.is_this_process else "other process"
        print(f"  [{tag}] {row.detail}")

    print("\n=== MySQL server processlist ===")
    if audit.server_error:
        print(f"  Cannot reach server: {audit.server_error}")
    elif not audit.server_sessions:
        print("  (no matching sessions)")
    else:
        for row in audit.server_sessions:
            print(
                f"  id={row.session_id} user={row.user} host={row.host} "
                f"db={row.database} cmd={row.command} time={row.time_seconds}s "
                f"{row.info_preview!r}"
            )

    if audit.disposed_local:
        print("\nDisposed local SQLHelpers caches in this process.")
    if audit.killed_server_count:
        print(f"\nKilled {audit.killed_server_count} server session(s).")


def cleanup_mysql_connections(
    dispose_local: bool = False,
    kill_server: bool = False,
    user_filter: Optional[str] = None,
    host: Optional[str] = None,
    port: Optional[int] = None,
    print_report: bool = True,
) -> MysqlConnectionAudit:
    """
    Audit and optionally clean up MySQL connections.

    Simple API for notebooks and scripts::

        from dbase.database.connection_cleanup import cleanup_mysql_connections
        cleanup_mysql_connections(dispose_local=True)

    Args:
        dispose_local: Dispose SQLAlchemy engines / close pymysql in this process.
        kill_server: ``KILL`` server sessions for ``user_filter`` (or MYSQL_USER).
        user_filter: MySQL user for server audit/kill; defaults to MYSQL_USER.
        host: MySQL host override; defaults to MYSQL_HOST.
        port: MySQL port override; defaults to MYSQL_PORT or 3306.
        print_report: Print summary to stdout when True.

    Returns:
        ``MysqlConnectionAudit`` including any cleanup actions taken.
    """
    audit = audit_mysql_connections(user_filter=user_filter, host=host, port=port)

    if dispose_local:
        dispose_local_mysql_connections()
        audit.disposed_local = True
        audit.cached_engines, audit.cached_pymysql = audit_local_process_connections()

    if kill_server:
        killed, kill_error = kill_server_mysql_sessions(
            user_filter=user_filter,
            host=host,
            port=port,
        )
        audit.killed_server_count = killed
        if kill_error:
            audit.server_error = kill_error
        else:
            audit.server_sessions, audit.server_error = audit_server_mysql_sessions(
                user_filter=user_filter,
                host=host,
                port=port,
            )

    if print_report:
        print_mysql_connection_audit(audit)

    return audit


def _build_cleanup_cli_parser() -> argparse.ArgumentParser:
    """Build argparse parser for ``python -m dbase.database.connection_cleanup``."""
    parser = argparse.ArgumentParser(
        description="Audit and optionally clean up MySQL connections.",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Dispose SQLAlchemy/pymysql caches in this Python process.",
    )
    parser.add_argument(
        "--kill-server",
        action="store_true",
        help="KILL MySQL server sessions for MYSQL_USER (or --user-filter).",
    )
    parser.add_argument(
        "--user-filter",
        default=None,
        help="MySQL user for server audit/kill (default: MYSQL_USER).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Return audit only; do not print the report.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """
    CLI entrypoint for MySQL connection cleanup.

    Args:
        argv: Optional argument list; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code (0 on success).
    """
    parser = _build_cleanup_cli_parser()
    args = parser.parse_args(argv)
    cleanup_mysql_connections(
        dispose_local=args.local,
        kill_server=args.kill_server,
        user_filter=args.user_filter,
        print_report=not args.quiet,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
