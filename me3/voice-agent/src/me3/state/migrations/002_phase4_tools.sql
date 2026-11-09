-- Phase 4: idempotent tool ledger + notes idempotency; notification source link.

CREATE TABLE IF NOT EXISTS tool_idempotency (
    key TEXT PRIMARY KEY,
    tool TEXT NOT NULL,
    result_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

ALTER TABLE notes ADD COLUMN idempotency_key TEXT;
CREATE UNIQUE INDEX IF NOT EXISTS idx_notes_idempotency
    ON notes(idempotency_key) WHERE idempotency_key IS NOT NULL;

ALTER TABLE notifications ADD COLUMN source_kind TEXT;
ALTER TABLE notifications ADD COLUMN source_id TEXT;
CREATE INDEX IF NOT EXISTS idx_notifications_status ON notifications(status);
