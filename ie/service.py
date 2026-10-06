"""信息抽取（IE）的模板版本管理与抽取结果存取。

模板：``ie_template`` 分片存储
    每次新建模板产生一条版本记录（version 从 1 自增）；
    调整字段 = 新建一版，**老版本永不修改**，老结果永远绑定抽取当时的版本。

结果：``ie_result`` 分片存储
    每条结果记录 = 「某段文本 × 某模板版本」的一次抽取，并快照模板字段定义，
    因此同一段文本用不同模板（或同一模板不同版本）抽取互不影响、互不覆盖；
    同一 (text_hash, template_id, template_version) 重复抽取默认走 upsert，
    保持一个文档对应一条当前记录，历史版本记录各自保留。
"""

from __future__ import annotations

import csv
import io
import time
from typing import Any, Optional

from nlp.extractor import (BUILTIN_TEMPLATES, EXTRACTOR_VERSION,
                           FieldExtractor, Template, TemplateError,
                           validate_template_spec, text_hash)
from storage import StoreRegistry


def _sanitize_tpl_id(name: str) -> str:
    import re
    slug = re.sub(r"[^0-9A-Za-z_]", "_", name).strip("_").lower()
    return slug[:40] or "template"


class InfoExtractionService:
    def __init__(self, registry: StoreRegistry,
                 extractor: Optional[FieldExtractor] = None):
        self.registry = registry
        self.extractor = extractor or FieldExtractor()
        self._seeded = False

    # -- 模板存储 ---------------------------------------------------------
    @property
    def tpl_store(self):
        return self.registry.task("ie_template")

    @property
    def result_store(self):
        return self.registry.task("ie_result")

    def _all_version_records(self, tpl_id: str) -> list[dict]:
        return self.tpl_store.query(
            where=[("template_id", "eq", tpl_id)],
            order_by="version", order="asc")

    def seed_builtins(self, force: bool = False) -> int:
        """写入预置模板（简历 / 合同 / 通知），每个一版。幂等。"""
        count = 0
        for spec in BUILTIN_TEMPLATES:
            tpl_id = spec["key"]
            existing = self._all_version_records(tpl_id)
            if existing and not force:
                continue
            self._insert_version(
                tpl_id, spec, builtin=True,
                note=spec.get("description", ""))
            count += 1
        return count

    def _insert_version(self, tpl_id: str, spec: dict,
                        builtin: bool = False, note: str = "") -> dict:
        validate_template_spec(spec)  # 不合法直接抛 TemplateError
        records = self._all_version_records(tpl_id)
        version = (records[-1]["version"] + 1) if records else 1
        record = {
            "template_id": tpl_id,
            "version": version,
            "name": spec["name"],
            "description": spec.get("description", ""),
            "fields": spec["fields"],
            "builtin": builtin,
            "status": "active",
            "note": note,
            "created_at": time.time(),
        }
        rid = self.tpl_store.insert(record)
        record["id"] = rid
        return record

    def list_templates(self, include_versions: bool = False) -> list[dict]:
        """返回每个模板的最新版本；include_versions 时附带全部历史版本。"""
        latest: dict[str, dict] = {}
        for rec in self.tpl_store.all():
            if rec.get("_deleted"):
                continue
            tid = rec["template_id"]
            old = latest.get(tid)
            if old is None or rec["version"] > old["version"]:
                latest[tid] = rec
        items = []
        for rec in sorted(latest.values(),
                          key=lambda r: (not r.get("builtin", False),
                                         r["created_at"])):
            item = self._template_summary(rec)
            if include_versions:
                item["versions"] = [self._template_summary(v)
                                    for v in self._all_version_records(
                                        rec["template_id"])]
            items.append(item)
        return items

    @staticmethod
    def _template_summary(rec: dict) -> dict:
        return {
            "id": rec.get("id"),
            "template_id": rec["template_id"],
            "version": rec["version"],
            "name": rec["name"],
            "description": rec.get("description", ""),
            "builtin": rec.get("builtin", False),
            "fields": rec["fields"],
            "field_count": len(rec["fields"]),
            "created_at": rec.get("created_at"),
            "note": rec.get("note", ""),
        }

    def get_template(self, tpl_id: str,
                     version: Optional[int] = None) -> Optional[dict]:
        records = self._all_version_records(tpl_id)
        if not records:
            return None
        if version is None:
            return records[-1]
        for rec in records:
            if rec["version"] == version:
                return rec
        return None

    def create_template(self, spec: dict,
                        tpl_id: Optional[str] = None) -> dict:
        existing_id = tpl_id or self._find_id_by_name(spec["name"])
        if existing_id and self._all_version_records(existing_id):
            raise TemplateError(
                f"模板「{spec['name']}」已存在（id={existing_id}），"
                "如要修改请用「另存为新版本」")
        tpl_id = tpl_id or f"tpl_{_sanitize_tpl_id(spec['name'])}"
        # 同名前缀冲突时追加短后缀
        if self._all_version_records(tpl_id):
            tpl_id = f"{tpl_id}_{int(time.time()) % 100000}"
        return self._insert_version(tpl_id, spec)

    def new_version(self, tpl_id: str, spec: dict,
                    note: str = "") -> dict:
        if not self._all_version_records(tpl_id):
            raise TemplateError(f"模板不存在: {tpl_id}")
        if not spec.get("name"):
            spec["name"] = self.get_template(tpl_id)["name"]
        return self._insert_version(tpl_id, spec, note=note)

    def delete_template(self, tpl_id: str) -> bool:
        records = self._all_version_records(tpl_id)
        if not records:
            return False
        # 墓碑删除全部版本；已抽取结果仍保留，可回查
        for rec in records:
            self.tpl_store.delete(rec["id"])
        return True

    def _find_id_by_name(self, name: str) -> Optional[str]:
        for rec in self.tpl_store.all():
            if not rec.get("_deleted") and rec.get("name") == name:
                return rec["template_id"]
        return None

    # -- 抽取 -------------------------------------------------------------
    def extract(self, text: str, template_id: str,
                version: Optional[int] = None,
                corpus_id: Optional[str] = None,
                doc_name: Optional[str] = None,
                upsert: bool = True) -> dict:
        tpl = self.get_template(template_id, version)
        if not tpl:
            raise TemplateError(f"模板不存在: {template_id}@{version}")

        result = self.extractor.extract_template(text, Template(tpl))

        thash = text_hash(text)
        record = {
            "text": text,
            "text_hash": thash,
            "corpus_id": corpus_id,
            "doc_name": doc_name,
            "template_id": template_id,
            "template_version": tpl["version"],
            "template_name": tpl["name"],
            # 快照：即便模板以后改版/删除，本结果仍可完整回查
            "template_snapshot": {"name": tpl["name"], "fields": tpl["fields"]},
            "result": result,
            "missing": result["missing"],
            "missing_required": result["missing_required"],
            "conflicts": result["conflicts"],
            "extractor_version": EXTRACTOR_VERSION,
            "created_at": time.time(),
        }

        reused = None
        if upsert:
            old = self._find_records(thash, template_id, tpl["version"],
                                     corpus_id)
            if old:
                # 同一文本 × 同一模板版本：替换为新记录，保持一条当前结果
                self.result_store.delete(old[0]["id"])
                reused = old[0]["id"]
        rid = self.result_store.insert(record)
        record["id"] = rid
        return {"id": rid, "replaced": reused, "record": record}

    def extract_batch(self, docs: list[dict], template_id: str,
                      version: Optional[int] = None,
                      upsert: bool = True) -> dict:
        """对多篇文档批量抽取。docs: [{text, corpus_id?, doc_name?}]"""
        items = []
        missing_docs = 0
        for doc in docs:
            text = doc.get("text") or ""
            if not text.strip():
                continue
            out = self.extract(
                text, template_id, version,
                corpus_id=doc.get("corpus_id"),
                doc_name=doc.get("doc_name"), upsert=upsert)
            rec = out["record"]
            if rec["missing"]:
                missing_docs += 1
            items.append({"id": out["id"], "doc_name": rec["doc_name"],
                          "corpus_id": rec["corpus_id"],
                          "missing": rec["missing"],
                          "missing_required": rec["missing_required"],
                          "conflicts": rec["conflicts"]})
        return {"template_id": template_id, "total": len(items),
                "docs_with_missing": missing_docs, "items": items}

    # -- 结果查询 / 导出 --------------------------------------------------
    def _find_records(self, thash: str, template_id: str,
                      version: int, corpus_id: Optional[str]) -> list[dict]:
        where = [("text_hash", "eq", thash),
                 ("template_id", "eq", template_id),
                 ("template_version", "eq", version)]
        if corpus_id is not None:
            where.append(("corpus_id", "eq", corpus_id))
        return self.result_store.query(where=where)

    def query_results(self, template_id: Optional[str] = None,
                      version: Optional[int] = None,
                      corpus_id: Optional[str] = None,
                      text_hash_eq: Optional[str] = None) -> list[dict]:
        where = []
        if template_id:
            where.append(("template_id", "eq", template_id))
        if version is not None:
            where.append(("template_version", "eq", version))
        if corpus_id:
            where.append(("corpus_id", "eq", corpus_id))
        if text_hash_eq:
            where.append(("text_hash", "eq", text_hash_eq))
        records = self.result_store.query(where=where or None,
                                          order_by="created_at", order="desc")
        return [r for r in records if not r.get("_deleted")]

    def get_result(self, result_id: str) -> Optional[dict]:
        rec = self.result_store.get(result_id)
        if rec and not rec.get("_deleted"):
            return rec
        return None

    def flat_record(self, record: dict) -> dict:
        """把抽取结果拍平成一行规整记录（供表格 / 导出）。"""
        row = {
            "result_id": record["id"],
            "doc_name": record.get("doc_name") or "",
            "corpus_id": record.get("corpus_id") or "",
            "template": record["template_name"],
            "template_version": record["template_version"],
        }
        for key, fr in record["result"]["fields"].items():
            label = fr.get("label") or key
            if fr["status"] == "missing":
                row[label] = ""
                row[f"{label}__状态"] = "缺失"
            elif fr["status"] == "conflict":
                cands = " | ".join(c["raw"] for c in fr.get("candidates", []))
                row[label] = cands
                row[f"{label}__状态"] = "冲突待确认"
            else:
                val = fr["value"]
                if fr.get("multi"):
                    val = "\n".join(val) if isinstance(val, list) else val
                row[label] = val
                row[f"{label}__状态"] = "已抽取"
        return row

    def export_csv(self, records: list[dict]) -> str:
        """把多条结果导成 CSV（utf-8-sig，Excel 直接打开不乱码）。"""
        if not records:
            return ""
        rows = [self.flat_record(r) for r in records]
        # 以最新一版模板的字段顺序为列序
        columns: list[str] = []
        for row in rows:
            for col in row:
                if col not in columns:
                    columns.append(col)
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=columns,
                                extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
        return buf.getvalue()
