from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


NOTES_HEADER = "## Notes"
OWNER_RULES_HEADER = "## Owner Rules"


def _append_to_section(content: str, header: str, line: str, default_after_header_blank: bool = True) -> str:
    """Insert `line` into the section identified by `header`.

    Behavior:
    - If header exists, append the bullet at the end of that section (before the
      next `## ` heading or the end of file).
    - If header does not exist, append the section to the end of the file with
      the bullet under it.
    """
    lines = content.rstrip().splitlines() if content.strip() else []
    header_index: int | None = None
    for index, candidate in enumerate(lines):
        if candidate.strip() == header:
            header_index = index
            break

    if header_index is None:
        rebuilt = list(lines)
        if rebuilt and rebuilt[-1] != "":
            rebuilt.append("")
        rebuilt.append(header)
        if default_after_header_blank:
            rebuilt.append("")
        rebuilt.append(line)
        return "\n".join(rebuilt) + "\n"

    section_end = len(lines)
    for index in range(header_index + 1, len(lines)):
        if lines[index].startswith("## "):
            section_end = index
            break

    insert_at = section_end
    while insert_at > header_index + 1 and lines[insert_at - 1].strip() == "":
        insert_at -= 1

    rebuilt = lines[:insert_at] + [line] + lines[insert_at:]
    return "\n".join(rebuilt) + "\n"


def _section_lines(content: str, header: str) -> list[str]:
    """Return the bullet lines (raw, including '- ' prefix) under `header`."""
    if not content.strip():
        return []
    lines = content.splitlines()
    start = None
    for index, candidate in enumerate(lines):
        if candidate.strip() == header:
            start = index + 1
            break
    if start is None:
        return []
    out: list[str] = []
    for line in lines[start:]:
        if line.startswith("## "):
            break
        if line.strip().startswith("- "):
            out.append(line.strip())
    return out


def _strip_section(content: str, header: str) -> str:
    if not content.strip():
        return content
    lines = content.splitlines()
    output: list[str] = []
    in_section = False
    for line in lines:
        if line.strip() == header:
            in_section = True
            continue
        if in_section and line.startswith("## "):
            in_section = False
        if in_section:
            continue
        output.append(line)
    cleaned = "\n".join(output)
    while "\n\n\n" in cleaned:
        cleaned = cleaned.replace("\n\n\n", "\n\n")
    return cleaned.strip() + ("\n" if cleaned.strip() else "")


@dataclass(slots=True)
class UserMemoryStore:
    base_dir: Path
    max_chars_in_prompt: int = 6000

    def __post_init__(self) -> None:
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def path_for_user(self, user_id: int) -> Path:
        return self.base_dir / f"{user_id}.md"

    def read_for_user(self, user_id: int) -> str:
        path = self.path_for_user(user_id)
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8").strip()

    def read_for_prompt(self, user_id: int) -> str:
        content = self.read_for_user(user_id)
        if not content:
            return "User memory:\n(none)\n"

        if len(content) > self.max_chars_in_prompt:
            content = content[-self.max_chars_in_prompt :]
            first_newline = content.find("\n")
            if first_newline != -1:
                content = content[first_newline + 1 :]

        return f"User memory:\n{content}\n"

    def append_for_user(
        self,
        *,
        user_id: int,
        note: str,
        author_display_name: str | None = None,
        source_channel_id: int | None = None,  # kept for back-compat; not used in line
    ) -> Path:
        del source_channel_id
        path = self.path_for_user(user_id)
        safe_note = note.strip()
        if not safe_note:
            raise ValueError("Memory note cannot be empty.")

        if path.exists():
            existing = path.read_text(encoding="utf-8")
        else:
            existing = self._new_file_header(
                user_id=user_id,
                author_display_name=author_display_name,
            )

        new_content = _append_to_section(existing, NOTES_HEADER, f"- {safe_note}")
        path.write_text(new_content, encoding="utf-8")
        return path

    def replace_for_user(
        self,
        *,
        user_id: int,
        old_text: str,
        new_text: str,
    ) -> tuple[Path, bool]:
        path = self.path_for_user(user_id)
        if not path.exists():
            return path, False

        old_value = old_text.strip()
        new_value = new_text.strip()
        if not old_value:
            raise ValueError("old_text is required.")

        content = path.read_text(encoding="utf-8")
        if old_value not in content:
            return path, False

        updated = content.replace(old_value, new_value, 1)
        path.write_text(updated, encoding="utf-8")
        return path, True

    @staticmethod
    def _new_file_header(*, user_id: int, author_display_name: str | None) -> str:
        display_name = author_display_name or "unknown"
        return (
            f"# User Memory: {user_id}\n\n"
            f"## Identity\n"
            f"- user_id: {user_id}\n"
            f"- display_name: {display_name}\n\n"
            f"## Notes\n"
        )


@dataclass(slots=True)
class GuildMemoryStore:
    base_dir: Path
    max_chars_in_prompt: int = 6000
    owner_rules_header: str = OWNER_RULES_HEADER

    def __post_init__(self) -> None:
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def path_for_guild(self, guild_id: int) -> Path:
        return self.base_dir / f"{guild_id}.md"

    def read_for_guild(self, guild_id: int) -> str:
        path = self.path_for_guild(guild_id)
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8").strip()

    def read_for_prompt(self, guild_id: int) -> str:
        content = self.read_for_guild(guild_id)
        if not content:
            return "Server memory:\n(none)\n"

        if len(content) > self.max_chars_in_prompt:
            content = content[-self.max_chars_in_prompt :]
            first_newline = content.find("\n")
            if first_newline != -1:
                content = content[first_newline + 1 :]

        return (
            "Server memory:\n"
            "Treat lines prefixed with [OWNER] as highest-priority instructions.\n"
            f"{content}\n"
        )

    def append_for_guild(
        self,
        *,
        guild_id: int,
        note: str,
        guild_name: str | None = None,
        author_display_name: str | None = None,  # kept for back-compat; not used in line
        source_channel_id: int | None = None,    # kept for back-compat; not used in line
        owner_priority: bool = False,
    ) -> Path:
        del author_display_name, source_channel_id
        path = self.path_for_guild(guild_id)
        safe_note = note.strip()
        if not safe_note:
            raise ValueError("Server memory note cannot be empty.")

        if path.exists():
            existing = path.read_text(encoding="utf-8")
        else:
            existing = self._new_file_header(
                guild_id=guild_id,
                guild_name=guild_name,
            )

        if owner_priority:
            normalized_rule = self._normalize_owner_rule_text(safe_note)
            new_content = self._upsert_owner_rules(existing, [normalized_rule])
        else:
            new_content = _append_to_section(existing, NOTES_HEADER, f"- {safe_note}")
        path.write_text(new_content, encoding="utf-8")
        return path

    def replace_for_guild(
        self,
        *,
        guild_id: int,
        old_text: str,
        new_text: str,
    ) -> tuple[Path, bool]:
        path = self.path_for_guild(guild_id)
        if not path.exists():
            return path, False

        old_value = old_text.strip()
        new_value = new_text.strip()
        if not old_value:
            raise ValueError("old_text is required.")

        content = path.read_text(encoding="utf-8")
        if old_value not in content:
            return path, False

        updated = content.replace(old_value, new_value, 1)
        path.write_text(updated, encoding="utf-8")
        return path, True

    def write_for_guild(
        self,
        *,
        guild_id: int,
        content: str,
    ) -> Path:
        path = self.path_for_guild(guild_id)
        existing = path.read_text(encoding="utf-8") if path.exists() else ""
        owner_rules = self.extract_owner_rules(existing)
        rendered = content.rstrip() + "\n"
        if owner_rules:
            rendered = self._upsert_owner_rules(rendered, owner_rules)
        path.write_text(rendered, encoding="utf-8")
        return path

    def clear_for_guild(
        self,
        *,
        guild_id: int,
        guild_name: str | None = None,
    ) -> Path:
        path = self.path_for_guild(guild_id)
        path.write_text(self._new_file_header(guild_id=guild_id, guild_name=guild_name), encoding="utf-8")
        return path

    @staticmethod
    def _new_file_header(*, guild_id: int, guild_name: str | None) -> str:
        safe_guild_name = guild_name or "unknown"
        return (
            f"# Server Memory: {guild_id}\n\n"
            f"## Server Info\n"
            f"- guild_id: {guild_id}\n"
            f"- name: {safe_guild_name}\n\n"
            f"## Owner Rules\n\n"
            f"## Notes\n"
        )

    def extract_owner_rules(self, content: str) -> list[str]:
        rules: list[str] = []
        for raw in _section_lines(content, self.owner_rules_header):
            normalized = self._normalize_owner_rule_text(raw[2:] if raw.startswith("- ") else raw)
            if normalized:
                rules.append(normalized)
        return self._dedupe_owner_rules(rules)

    def strip_owner_rules_section(self, content: str) -> str:
        return _strip_section(content, self.owner_rules_header)

    def _upsert_owner_rules(self, content: str, new_rules: list[str]) -> str:
        merged_rules = self._dedupe_owner_rules([*self.extract_owner_rules(content), *new_rules])
        base_content = self.strip_owner_rules_section(content).rstrip()
        if not merged_rules:
            return base_content + "\n"

        owner_lines = [self.owner_rules_header, *[f"- [OWNER] {rule}" for rule in merged_rules]]
        lines = base_content.splitlines()

        insert_at = len(lines)
        for index, line in enumerate(lines):
            if line.startswith("## "):
                insert_at = index
                break

        before = lines[:insert_at]
        after = lines[insert_at:]
        rebuilt: list[str] = [*before]
        if rebuilt and rebuilt[-1] != "":
            rebuilt.append("")
        rebuilt.extend(owner_lines)
        if after:
            rebuilt.append("")
            rebuilt.extend(after)
        return "\n".join(rebuilt).rstrip() + "\n"

    @staticmethod
    def _normalize_owner_rule_text(value: str) -> str:
        text = value.strip()
        text = text.removeprefix("- ").strip()
        while text.startswith("[OWNER]"):
            text = text[len("[OWNER]") :].strip()
        return text

    @staticmethod
    def _dedupe_owner_rules(rules: list[str]) -> list[str]:
        output: list[str] = []
        seen: set[str] = set()
        for rule in rules:
            normalized = rule.strip()
            if not normalized:
                continue
            key = normalized.casefold()
            if key in seen:
                continue
            seen.add(key)
            output.append(normalized)
        return output
