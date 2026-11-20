"""
SQLite audit history database for the Car Audit Tool.

Stores audits and users in a local SQLite file.
- `heatmap_files` is stored as JSON (same shape as the API response).
- Lightweight column migrations run on startup.
- All queries are parameterised.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


logger = logging.getLogger("car_audit_tool.database")


class AuditDatabase:
    """SQLite-backed audit history store."""

    SCHEMA_VERSION = 4

    def __init__(self, db_path: str = "database/audit_history.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self.init_database()

    # --------------------------------------------------------------------------
    # Setup
    # --------------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=15)
        conn.row_factory = sqlite3.Row
        return conn

    def init_database(self) -> None:
        conn = self._connect()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS audits (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    audit_id TEXT UNIQUE NOT NULL,
                    vehicle_id TEXT,
                    customer_name TEXT,
                    rental_start_date TEXT,
                    rental_end_date TEXT,
                    trust_score REAL,
                    condition_assessment TEXT,
                    severity_level INTEGER,
                    recommendation TEXT,
                    semantic_similarity REAL,
                    ssim REAL,
                    mse REAL,
                    psnr REAL,
                    mad REAL,
                    heatmap_files TEXT,
                    golden_image_path TEXT,
                    returned_image_path TEXT,
                    processing_time_sec REAL,
                    weight_source TEXT,
                    notes TEXT,
                    changed_area_pct REAL,
                    region_count INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    processed_by TEXT
                )
                """
            )
            # Light-weight migration: add columns if upgrading from v1 schema
            existing = {row[1] for row in cursor.execute("PRAGMA table_info(audits)").fetchall()}
            for col, decl in [
                ("processing_time_sec", "REAL"),
                ("weight_source", "TEXT"),
                ("notes", "TEXT"),
                ("changed_area_pct", "REAL"),
                ("region_count", "INTEGER"),
                ("processed_by", "TEXT"),
            ]:
                if col not in existing:
                    cursor.execute(f"ALTER TABLE audits ADD COLUMN {col} {decl}")

            cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_id ON audits(audit_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_vehicle_id ON audits(vehicle_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON audits(created_at)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_condition ON audits(condition_assessment)")

            # ---- Users table (v2.1) ----
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL DEFAULT 'user',
                    display_name TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP,
                    token_version INTEGER NOT NULL DEFAULT 1
                )
                """
            )
            # v3 migration: add token_version if upgrading from an older schema.
            # token_version is included in every JWT, so bumping it (on password
            # change, role change, delete or logout) invalidates every previously
            # issued token before its natural expiry.
            user_cols = {row[1] for row in cursor.execute("PRAGMA table_info(users)").fetchall()}
            if "token_version" not in user_cols:
                cursor.execute("ALTER TABLE users ADD COLUMN token_version INTEGER NOT NULL DEFAULT 1")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)")
            conn.commit()
            logger.info("Database ready at %s (schema v%d)", self.db_path, self.SCHEMA_VERSION)
        finally:
            conn.close()

    # --------------------------------------------------------------------------
    # CRUD
    # --------------------------------------------------------------------------

    def save_audit(self, data: Dict[str, Any]) -> bool:
        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT INTO audits (
                    audit_id, vehicle_id, customer_name,
                    rental_start_date, rental_end_date,
                    trust_score, condition_assessment, severity_level,
                    recommendation, semantic_similarity,
                    ssim, mse, psnr, mad,
                    heatmap_files, golden_image_path, returned_image_path,
                    processing_time_sec, weight_source, processed_by,
                    changed_area_pct, region_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    data.get("audit_id"),
                    data.get("vehicle_id"),
                    data.get("customer_name"),
                    data.get("rental_start_date"),
                    data.get("rental_end_date"),
                    data.get("trust_score"),
                    data.get("condition_assessment"),
                    data.get("severity_level"),
                    data.get("recommendation"),
                    (data.get("semantic_analysis") or {}).get("cosine_similarity"),
                    (data.get("pixel_level_metrics") or {}).get("ssim"),
                    (data.get("pixel_level_metrics") or {}).get("mse"),
                    (data.get("pixel_level_metrics") or {}).get("psnr"),
                    (data.get("pixel_level_metrics") or {}).get("mad"),
                    json.dumps(data.get("heatmap_files") or {}),
                    data.get("golden_image_path"),
                    data.get("returned_image_path"),
                    (data.get("metadata") or {}).get("processing_time_sec"),
                    (data.get("metadata") or {}).get("weight_source"),
                    data.get("processed_by", "system"),
                    (data.get("change_analysis") or {}).get("changed_area_pct"),
                    (data.get("change_analysis") or {}).get("region_count"),
                ),
            )
            conn.commit()
            logger.info("Saved audit %s", data.get("audit_id"))
            return True
        except Exception:
            logger.exception("Failed to save audit %s", data.get("audit_id"))
            conn.rollback()
            return False
        finally:
            conn.close()

    def get_audit(self, audit_id: str) -> Optional[Dict[str, Any]]:
        conn = self._connect()
        try:
            row = conn.execute("SELECT * FROM audits WHERE audit_id = ?", (audit_id,)).fetchone()
            return self._row_to_dict(row)
        finally:
            conn.close()

    def get_all_audits(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM audits ORDER BY created_at DESC, id DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
            return [self._row_to_dict(r) for r in rows]
        finally:
            conn.close()

    def get_audits_by_vehicle(self, vehicle_id: str, limit: int = 500, offset: int = 0) -> List[Dict[str, Any]]:
        """Return audits for one vehicle, newest first. Limit/offset capped to
        keep the response bounded (the previous endpoint returned every row,
        which grows forever as audits accumulate)."""
        limit = max(1, min(limit, 500))
        offset = max(0, offset)
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM audits WHERE vehicle_id = ? ORDER BY created_at DESC, id DESC LIMIT ? OFFSET ?",
                (vehicle_id, limit, offset),
            ).fetchall()
            return [self._row_to_dict(r) for r in rows]
        finally:
            conn.close()

    def delete_audit(self, audit_id: str) -> bool:
        conn = self._connect()
        try:
            cur = conn.execute("DELETE FROM audits WHERE audit_id = ?", (audit_id,))
            conn.commit()
            return cur.rowcount > 0
        finally:
            conn.close()

    def update_notes(self, audit_id: str, notes: str) -> bool:
        conn = self._connect()
        try:
            cur = conn.execute("UPDATE audits SET notes = ? WHERE audit_id = ?", (notes, audit_id))
            conn.commit()
            return cur.rowcount > 0
        finally:
            conn.close()

    # --------------------------------------------------------------------------
    # Statistics
    # --------------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        conn = self._connect()
        try:
            stats: Dict[str, Any] = {}
            stats["total_audits"] = conn.execute("SELECT COUNT(*) FROM audits").fetchone()[0]

            avg_row = conn.execute("SELECT AVG(trust_score) FROM audits").fetchone()
            stats["avg_trust_score"] = round(avg_row[0] or 0.0, 4)

            stats["condition_distribution"] = {
                row[0] or "UNKNOWN": row[1]
                for row in conn.execute(
                    "SELECT condition_assessment, COUNT(*) FROM audits GROUP BY condition_assessment"
                ).fetchall()
            }

            stats["recent_audits_7days"] = conn.execute(
                "SELECT COUNT(*) FROM audits WHERE created_at >= datetime('now', '-7 days')"
            ).fetchone()[0]

            # Top 5 worst-performing vehicles (lowest average trust score)
            stats["worst_vehicles"] = [
                {
                    "vehicle_id": row[0],
                    "audits": row[1],
                    "avg_trust_score": round(row[2] or 0.0, 4),
                }
                for row in conn.execute(
                    """
                    SELECT vehicle_id, COUNT(*), AVG(trust_score)
                    FROM audits
                    WHERE vehicle_id IS NOT NULL AND vehicle_id != ''
                    GROUP BY vehicle_id
                    ORDER BY AVG(trust_score) ASC
                    LIMIT 5
                    """
                ).fetchall()
            ]

            # Score trend (last 14 audits) - for the inline sparkline in the UI
            trend_rows = conn.execute(
                "SELECT trust_score, created_at FROM audits ORDER BY created_at DESC, id DESC LIMIT 14"
            ).fetchall()
            stats["trend"] = [
                {"trust_score": row[0], "created_at": row[1]} for row in reversed(trend_rows)
            ]
            return stats
        finally:
            conn.close()

    # --------------------------------------------------------------------------
    # User management (auth)
    # --------------------------------------------------------------------------

    def count_users(self) -> int:
        conn = self._connect()
        try:
            return conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        finally:
            conn.close()

    def create_user(self, username: str, password_hash: str, role: str = "user",
                    display_name: Optional[str] = None) -> int:
        conn = self._connect()
        try:
            cur = conn.execute(
                "INSERT INTO users (username, password_hash, role, display_name) VALUES (?, ?, ?, ?)",
                (username, password_hash, role, display_name),
            )
            conn.commit()
            return cur.lastrowid
        finally:
            conn.close()

    def get_user_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM users WHERE username = ? COLLATE NOCASE", (username,)
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def get_user_by_id(self, user_id: int) -> Optional[Dict[str, Any]]:
        conn = self._connect()
        try:
            row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def update_last_login(self, user_id: int) -> None:
        conn = self._connect()
        try:
            conn.execute(
                "UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?", (user_id,)
            )
            conn.commit()
        finally:
            conn.close()

    def update_password(self, user_id: int, password_hash: str) -> bool:
        """Set a new password hash AND bump token_version so every previously
        issued JWT (access and refresh) is immediately invalid - i.e. a password
        change logs the user out everywhere."""
        conn = self._connect()
        try:
            cur = conn.execute(
                "UPDATE users SET password_hash = ?, token_version = token_version + 1 WHERE id = ?",
                (password_hash, user_id),
            )
            conn.commit()
            return cur.rowcount > 0
        finally:
            conn.close()

    def bump_token_version(self, user_id: int) -> bool:
        """Invalidate every JWT issued for this user by bumping token_version.
        Used by the logout endpoint and by role changes."""
        conn = self._connect()
        try:
            cur = conn.execute(
                "UPDATE users SET token_version = token_version + 1 WHERE id = ?",
                (user_id,),
            )
            conn.commit()
            return cur.rowcount > 0
        finally:
            conn.close()

    def update_user_role(self, user_id: int, role: str) -> bool:
        """Change a user's role and bump token_version so the change takes
        effect immediately instead of after the 8h access-token expiry."""
        if role not in ("admin", "user"):
            return False
        conn = self._connect()
        try:
            cur = conn.execute(
                "UPDATE users SET role = ?, token_version = token_version + 1 WHERE id = ?",
                (role, user_id),
            )
            conn.commit()
            return cur.rowcount > 0
        finally:
            conn.close()

    def list_users(self) -> List[Dict[str, Any]]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT id, username, role, display_name, created_at, last_login, token_version "
                "FROM users ORDER BY id"
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def delete_user(self, user_id: int) -> bool:
        # Never allow deleting the last admin. Bump token_version first so any
        # token the deleted user still holds fails immediately on the next
        # request (the user row is then gone, so get_user_by_id returns None).
        conn = self._connect()
        try:
            admin_count = conn.execute(
                "SELECT COUNT(*) FROM users WHERE role = 'admin'"
            ).fetchone()[0]
            row = conn.execute("SELECT role FROM users WHERE id = ?", (user_id,)).fetchone()
            if row and row["role"] == "admin" and admin_count <= 1:
                return False
            conn.execute(
                "UPDATE users SET token_version = token_version + 1 WHERE id = ?", (user_id,)
            )
            cur = conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
            conn.commit()
            return cur.rowcount > 0
        finally:
            conn.close()

    # --------------------------------------------------------------------------
    # Helpers
    # --------------------------------------------------------------------------

    @staticmethod
    def _row_to_dict(row: Optional[sqlite3.Row]) -> Optional[Dict[str, Any]]:
        if row is None:
            return None
        d = dict(row)
        # Parse JSON heatmap_files back into a dict
        if d.get("heatmap_files"):
            try:
                d["heatmap_files"] = json.loads(d["heatmap_files"])
            except (ValueError, TypeError):
                pass
        # SQLite's CURRENT_TIMESTAMP is UTC but carries no zone marker, and JS
        # parses a zone-less ISO string as *local* time (so every audit showed
        # the wrong time unless the browser was in UTC). Mark it as UTC.
        if d.get("created_at"):
            try:
                ts = datetime.fromisoformat(str(d["created_at"]).replace("Z", "+00:00"))
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                d["created_at"] = ts.isoformat()
            except ValueError:
                pass
        return d
