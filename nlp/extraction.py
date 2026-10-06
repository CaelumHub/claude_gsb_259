"""模板化字段抽取。

针对简历、合同、通知等半结构化文本，按用户定义的「字段模板」把关键信息
抽成规整字段。核心设计：

1. **标签锚点优先**：半结构化文本里字段通常写成 ``姓名：张三``、
   ``甲方（发包方）：xx公司`` 的形式。抽取器先在全文定位所有字段的标签
   锚点，取锚点之后、到下一个锚点 / 换行 / 结束标点之前的片段作为候选，
   因此同一段文本用不同模板互不干扰。
2. **类型兜底**：锚点没命中时，按字段类型（电话 / 邮箱 / 金额 / 日期 /
   人名 / 机构 / …）用正则或 NER 在全文扫描，结果标为 ``inferred``
   （推断），与锚点命中的 ``found`` 区分可信度。
3. **缺失显式标记**：抽不到就给 ``status = missing``，``value`` 为 ``None``，
   绝不填错值；单值字段抽到多个不同写法时给 ``ambiguous``，全部候选
   保留待人工裁决，而不是静默选一个。
4. **逐处可回查**：每个候选都带 ``start/end`` 字符偏移与 ``source``
   来源标记（anchor / type / ner / regex），结果可以与原文逐处对上。
5. **多写法归一**：同一字段的多种写法（如 ``138-1234-5678`` 与
   ``13812345678``）经类型归一化后去重；归并前的原文保留在 raw 中。

字段状态：``found``（锚点命中）/ ``inferred``（全文推断）/
``ambiguous``（多个冲突值）/ ``missing``（缺失）。
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from .ner import NERExtractor


# ---------------------------------------------------------------------------
# 字段类型定义
# ---------------------------------------------------------------------------

FIELD_TYPE_NAMES = {
    "text": "文本",
    "person": "人名",
    "org": "机构",
    "phone": "电话",
    "email": "邮箱",
    "money": "金额",
    "date": "日期",
    "daterange": "期限",
    "address": "地址",
    "list": "列表",
    "section": "段落区块",
}

# 类型正则（不依赖锚点，用于兜底全文扫描与锚点片段类型校验）
_PHONE_RE = re.compile(
    r"(?:(?:\+?86[-\s]?)?1[3-9]\d(?:[-\s]?\d){8}"      # 138-1234-5678 式分段
    r"|(?:\+?86[-\s]?)?1[3-9]\d{9}"                    # 11 位连续
    r"|0\d{2,3}[-\s]?\d{7,8})"
    r"(?:[-\s]?转\d{2,5})?"
)
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
_MONEY_CN_RE = re.compile(
    r"(?:人民币|美元|港元|港币|欧元|日元|英镑)?"
    r"[零〇一二两三四五六七八九十百千万亿壹贰叁肆伍陆柒捌玖拾佰仟0-9]+"
    r"(?:[.,][0-9]+)?"
    r"(?:万元|亿元|万人民币|亿人民币|元整|圆整|元|圆|美元|港元|港币|欧元|日元|英镑)"
)
_MONEY_RE = re.compile(
    r"(?:人民币|美元|港元|港币|欧元|日元|英镑)?\s?"
    r"\d[\d,]*(?:\.\d+)?\s?"
    r"(?:万元|亿元|万人民币|亿人民币|元整|圆整|元|圆|美元|港元|港币|欧元|日元|英镑)"
)
_DATE_RE = re.compile(
    r"\d{4}\s*[年/\-.]\s*\d{1,2}\s*(?:月|\-|/|\.)\s*\d{1,2}\s*日?"
    r"|\d{4}\s*年\s*\d{1,2}\s*月"
    r"|\d{1,2}\s*月\s*\d{1,2}\s*日"
)
_DATERANGE_RE = re.compile(
    r"\d{4}\s*[年/\-.]\s*\d{1,2}(?:\s*(?:月|\-|/|\.)\s*\d{1,2}\s*日?)?"
    r"\s*(?:\s*(?:至|到|—|–|~|～|-)\s*|起至\s*)"
    r"\d{4}\s*[年/\-.]\s*\d{1,2}(?:\s*(?:月|\-|/|\.)\s*\d{1,2}\s*日?)?"
)
# 标签里允许出现的括号补充说明，如「甲方（发包方）」
_LABEL_TAIL_RE = r"(?:[（(][^）)\n]{0,20}[）)])?"
# 默认分隔：标签与值之间用冒号
_SEP_RE = r"\s*[:：]\s*"
# 金额/日期/期限等字段允许「金额为 / 期限从…起」之类的非冒号写法
_VERB_SEP_RE = r"\s*(?:[:：]|为|是|达|共计|合计)\s*"
# 语义过于泛化的标签：动词分隔下要求词边界，避免
# 「放假时间为…」「发布时间…」等被误命中
_GENERIC_LABELS = {"时间", "日期", "标题", "金额", "数量"}
# 锚点片段遇到这些标点即截断（句号、分号等；逗号用于行内多字段）
_STOP_PUNCT = "。；;！!？?，,、"

# 列表/区块字段按行切分时使用
_LINE_SPLIT_RE = re.compile(r"[\n\r]+")
# 列表项前常见的项目符号
_BULLET_RE = re.compile(
    r"^\s*(?:[-*•·▪◦]|\d+[.、)]|[（(]\d+[）)]|[一二三四五六七八九十]+[、.])\s*"
)


def _strip_bullet(line: str) -> str:
    return _BULLET_RE.sub("", line).strip()


def _is_word_char(ch: str) -> bool:
    """标签前导字符判断：中日韩字符或字母数字视为词内（标签需词边界）。"""
    return bool(re.match(r"[一-鿿A-Za-z0-9]", ch))


# ---------------------------------------------------------------------------
# 模板
# ---------------------------------------------------------------------------

class FieldSpec:
    """模板中的单个字段定义。

    :param key: 字段键（英文标识，模板内唯一）
    :param name: 字段显示名，同时作为标签锚点；可给 ``labels`` 补充别名
    :param type: 字段类型，见 :data:`FIELD_TYPE_NAMES`；
                 自定义正则用 ``text`` + ``pattern``
    :param labels: 标签别名（如姓名/名字），默认含 ``name``
    :param required: 是否必填（仅用于完整度统计）
    :param multi: 是否多值（电话多个、经历多条）
    :param pattern: 自定义正则（type=text 时生效）
    :param description: 说明
    """

    def __init__(self, key: str, name: str, type: str = "text",
                 labels: Optional[list[str]] = None,
                 required: bool = False, multi: bool = False,
                 pattern: Optional[str] = None,
                 description: str = ""):
        if not key or not re.fullmatch(r"[A-Za-z0-9_\-]+", key):
            raise ValueError(f"字段键不合法: {key!r}")
        if type not in FIELD_TYPE_NAMES:
            raise ValueError(f"未知字段类型: {type}")
        if pattern:
            try:
                re.compile(pattern)
            except re.error as exc:
                raise ValueError(f"自定义正则不合法: {exc}") from exc
        self.key = key
        self.name = name
        self.type = type
        self.labels = list(dict.fromkeys([name, *(labels or [])]))
        self.required = bool(required)
        self.multi = bool(multi) or type in ("list", "section")
        self.pattern = pattern
        self.description = description

    def to_dict(self) -> dict:
        return {
            "key": self.key, "name": self.name, "type": self.type,
            "labels": self.labels, "required": self.required,
            "multi": self.multi, "pattern": self.pattern,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FieldSpec":
        missing = {"key", "name"} - set(data)
        if missing:
            raise ValueError(f"字段定义缺少: {sorted(missing)}")
        return cls(
            key=data["key"], name=data["name"],
            type=data.get("type", "text"),
            labels=data.get("labels"), required=data.get("required", False),
            multi=data.get("multi", False), pattern=data.get("pattern"),
            description=data.get("description", ""),
        )


class FieldTemplate:
    """字段模板：一组字段定义，如「简历模板」「合同模板」。"""

    def __init__(self, name: str, fields: list[FieldSpec],
                 key: Optional[str] = None, version: int = 1,
                 description: str = ""):
        if not fields:
            raise ValueError("模板至少需要一个字段")
        keys = [f.key for f in fields]
        if len(set(keys)) != len(keys):
            raise ValueError("模板内字段 key 重复")
        self.key = key
        self.name = name
        self.fields = fields
        self.version = version
        self.description = description

    def to_dict(self) -> dict:
        return {
            "key": self.key, "name": self.name, "version": self.version,
            "description": self.description,
            "fields": [f.to_dict() for f in self.fields],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FieldTemplate":
        fields = [FieldSpec.from_dict(f) for f in data.get("fields", [])]
        return cls(
            name=data.get("name", "未命名模板"), fields=fields,
            key=data.get("key"), version=int(data.get("version", 1)),
            description=data.get("description", ""),
        )


# ---------------------------------------------------------------------------
# 抽取器
# ---------------------------------------------------------------------------

class TemplateExtractor:
    """按 :class:`FieldTemplate` 从文本抽取规整字段。"""

    def __init__(self, ner: Optional[NERExtractor] = None):
        self.ner = ner or NERExtractor()
        # 单次抽取内缓存 NER 结果，避免逐字段重复全文识别
        self._ner_cache: dict[int, list[dict]] = {}

    def _ner_entities(self, text: str) -> list[dict]:
        cached = self._ner_cache.get("ents")
        if cached is None:
            cached = self.ner.recognize(text, with_source=True)
            self._ner_cache["ents"] = cached
        return cached

    # -- 对外接口 ---------------------------------------------------------
    def extract(self, text: str, template: FieldTemplate | dict) -> dict:
        """对一段文本执行模板抽取，返回逐字段可回查的结构化结果。"""
        if isinstance(template, dict):
            template = FieldTemplate.from_dict(template)

        self._ner_cache = {"ents": self.ner.recognize(text, with_source=True)}
        anchors = self._find_anchors(text, template.fields)
        fields_out: dict[str, dict] = {}
        for spec in template.fields:
            fields_out[spec.key] = self._extract_field(text, spec, anchors)
        self._ner_cache.clear()

        missing = [k for k, v in fields_out.items() if v["status"] == "missing"]
        ambiguous = [k for k, v in fields_out.items() if v["status"] == "ambiguous"]
        required_missing = [s.key for s in template.fields
                            if s.required and fields_out[s.key]["status"] == "missing"]
        return {
            "template_key": template.key,
            "template_version": template.version,
            "template_snapshot": template.to_dict(),
            "fields": fields_out,
            "missing": missing,
            "ambiguous": ambiguous,
            "required_missing": required_missing,
            "complete": not required_missing and not ambiguous,
        }

    # -- 标签锚点 ---------------------------------------------------------
    def _find_anchors(self, text: str, fields: list[FieldSpec]) -> list[dict]:
        """全文扫描所有字段的标签锚点，按位置排序。

        锚点 = 标签名（允许括号补充说明）+ 分隔符。默认要求冒号；
        金额/日期/期限字段也接受「金额为 12 万」这类动词分隔。
        边界约束：标签前必须是行首或非中文/字母数字字符，避免
        「放假时间为」里的标签「时间」、「专业技能」里的「技能」误命中。
        同一位置多个标签命中时只保留最长者。
        """
        anchors: list[dict] = []
        # 标签前缀表：短标签若只是更长词的后缀（如「金额」前有「合同」），
        # 让位给长标签，避免同位置重复命中
        label_set = {label for spec in fields for label in spec.labels if label}
        for spec in fields:
            verb_sep = spec.type in ("money", "date", "daterange")
            sep = _VERB_SEP_RE if verb_sep else _SEP_RE
            for label in spec.labels:
                if not label:
                    continue
                pattern = re.compile(re.escape(label) + _LABEL_TAIL_RE + sep)
                for m in pattern.finditer(text):
                    if m.start() > 0 and _is_word_char(text[m.start() - 1]):
                        # 前字与标签拼成更长标签（合同金额 中的 金额）→ 跳过
                        if text[m.start() - 1] + label in label_set:
                            continue
                        # 泛化词（时间/日期/标题…）动词分隔时要求词边界，
                        # 避免「放假时间为」被「时间」误命中
                        if verb_sep and label in _GENERIC_LABELS:
                            continue
                        # 冒号分隔下标签一律要求词边界
                        if not verb_sep:
                            continue
                    anchors.append({
                        "field_key": spec.key,
                        "label": label,
                        "start": m.start(),
                        "colon_end": m.end(),
                    })
        anchors.sort(key=lambda a: (a["start"], -len(a["label"])))
        deduped: list[dict] = []
        for anchor in anchors:
            if deduped and deduped[-1]["start"] == anchor["start"]:
                continue
            deduped.append(anchor)
        deduped.sort(key=lambda a: a["start"])
        return deduped

    @staticmethod
    def _anchor_window(text: str, anchor: dict,
                       all_anchors: list[dict]) -> tuple[int, int]:
        """锚点值窗口：从冒号之后到换行 / 下一个任意字段锚点 / 终止标点。"""
        start = anchor["colon_end"]
        line_end = text.find("\n", start)
        if line_end < 0:
            line_end = len(text)
        next_anchor = len(text)
        for other in all_anchors:
            if other["start"] >= start:
                next_anchor = other["start"]
                break
        end = min(line_end, next_anchor)
        # 截断到窗口内第一个终止标点
        window = text[start:end]
        stop = len(window)
        for i, ch in enumerate(window):
            if ch in _STOP_PUNCT:
                stop = i
                break
        return start, start + stop

    # -- 单字段抽取 -------------------------------------------------------
    def _extract_field(self, text: str, spec: FieldSpec,
                       anchors: list[dict]) -> dict:
        base = {"name": spec.name, "type": spec.type,
                "required": spec.required, "multi": spec.multi}
        mine = [a for a in anchors if a["field_key"] == spec.key]

        if spec.type in ("list", "section"):
            return {**base, **self._extract_block(text, spec, mine, anchors)}

        # 1) 锚点候选：对每个属于本字段的锚点取窗口并按类型提值
        candidates: list[dict] = []
        for anchor in mine:
            ws, we = self._anchor_window(text, anchor, anchors)
            candidates.extend(
                self._values_in_window(text, spec, ws, we, "anchor"))

        # 2) 全文扫描：
        #    - 锚点没提出来 -> 扫描结果作为推断（inferred），不冒充标签命中；
        #    - 多值字段（如电话）即使有锚点也合并全文同类型命中，
        #      保证「电话：x，备用手机 y」这种换标签写法也收得全。
        had_anchor_value = bool(candidates)
        if not had_anchor_value or spec.multi:
            candidates.extend(self._scan_fallback(text, spec))

        return self._resolve(text, spec, candidates,
                             had_anchor=had_anchor_value, base=base)

    def _extract_block(self, text: str, spec: FieldSpec,
                       mine: list[dict],
                       all_anchors: list[dict]) -> dict:
        """列表 / 段落区块字段：取标签后整块文本，按行拆成多条。"""
        if not mine:
            # 区块字段不做全文兜底：没有明确标签无法知道块的边界
            return {"value": None, "status": "missing", "candidates": []}

        anchor = mine[0]
        block_start = anchor["colon_end"]
        # 区块边界：下一个任意字段锚点（可跨行），或文末
        end = len(text)
        for other in all_anchors:
            if other["start"] > anchor["start"]:
                end = other["start"]
                break
        end = min(end, block_start + 4000)
        block = text[block_start:end].strip()
        lead = block_start + (len(text[block_start:end])
                              - len(text[block_start:end].lstrip()))
        block = text[lead:end].rstrip()

        items: list[dict] = []
        seen: set[str] = set()
        pos = lead
        for line in _LINE_SPLIT_RE.split(block):
            stripped = line.strip()
            if not stripped:
                pos += len(line) + 1
                continue
            col = line.find(stripped)
            s = pos + col
            raw = _strip_bullet(stripped)
            bullet_col = stripped.find(raw)
            s += bullet_col
            e = s + len(raw)
            pos += len(line) + 1
            key = _norm_key(raw)
            if key in seen:
                continue
            seen.add(key)
            items.append({"text": raw, "start": s, "end": e})
            if spec.type == "list" and len(items) >= 50:
                break
        if not items:
            return {"value": None, "status": "missing", "candidates": []}
        value = [it["text"] for it in items] if spec.type == "list" else block
        return {
            "value": value,
            "status": "found",
            "raw_text": block if spec.type == "list" else None,
            "candidates": [{
                "text": it["text"], "value": it["text"],
                "source": "anchor",
                "evidence": [{"text": it["text"], "start": it["start"],
                              "end": it["end"], "source": "anchor"}],
                "raw_variants": [it["text"]],
            } for it in items],
        }

    # -- 窗口提值 ---------------------------------------------------------
    def _values_in_window(self, text: str, spec: FieldSpec,
                          start: int, end: int, source: str) -> list[dict]:
        """在锚点值窗口 [start, end) 内按字段类型提取候选。"""
        window = text[start:end]
        if not window.strip():
            return []

        if spec.pattern:
            return self._regex_hits(text, spec.pattern, start, end, "regex")

        extractor = self._typed_extractors.get(spec.type)
        if extractor:
            return extractor(self, text, start, end, source, anchor=True)

        # text：整个窗口作为一个值，清理首尾空白与省略尾巴
        value = window.strip(" \t：:，,。；;、")
        if not value:
            return []
        vs = start + (len(window) - len(window.lstrip(" \t")))
        ve = vs + len(value)
        return [self._candidate(text, value, vs, ve, source, value)]

    def _hits_phone(self, text: str, start: int, end: int,
                    source: str = "type", anchor: bool = False) -> list[dict]:
        return [self._candidate(text, m.group(), m.start(), m.end(), source,
                                _canonical_phone(m.group()))
                for m in _PHONE_RE.finditer(text, start, end)]

    def _hits_email(self, text: str, start: int, end: int,
                    source: str = "type", anchor: bool = False) -> list[dict]:
        return [self._candidate(text, m.group(), m.start(), m.end(), source,
                                m.group().strip().lower())
                for m in _EMAIL_RE.finditer(text, start, end)]

    def _hits_money(self, text: str, start: int, end: int,
                    source: str = "type", anchor: bool = False) -> list[dict]:
        out: list[dict] = []
        pattern = _MONEY_CN_RE if anchor else _MONEY_RE
        for m in pattern.finditer(text, start, end):
            raw = m.group().strip()
            amount, unit = _normalize_money(raw)
            if amount is None:
                continue
            disp = _canonical_amount(amount, unit)
            out.append(self._candidate(
                text, raw, m.start(), m.end(), source,
                {"amount": amount, "unit": unit, "text": disp}))
        return out

    def _hits_date(self, text: str, start: int, end: int,
                   source: str = "type", anchor: bool = False) -> list[dict]:
        return [self._candidate(text, m.group().strip(), m.start(), m.end(),
                                source, _normalize_date(m.group()))
                for m in _DATE_RE.finditer(text, start, end)]

    def _hits_daterange(self, text: str, start: int, end: int,
                        source: str = "type", anchor: bool = False) -> list[dict]:
        return [self._candidate(text, m.group().strip(), m.start(), m.end(),
                                source, _normalize_daterange(m.group()))
                for m in _DATERANGE_RE.finditer(text, start, end)]

    def _hits_person(self, text: str, start: int, end: int,
                     source: str = "ner", anchor: bool = False) -> list[dict]:
        if anchor:
            # 标签已声明这是人名：在窗口里按「姓氏 + 名字」直接提值，
            # 比结构规则 NER 更可靠（NER 可能把「简历」误判成人名）。
            direct = self._person_in_window(text, start, end)
            if direct:
                return direct
        hits = self._ner_hits(text, start, end, {"PERSON"})
        for h in hits:
            h["source"] = source
        return hits

    def _person_in_window(self, text: str, start: int, end: int) -> list[dict]:
        from .lexicon import SURNAMES, GIVEN_NAME_CHARS
        window = text[start:end].strip(" \t：:，,。；;、")
        pad = start + (len(text[start:end]) - len(text[start:end].lstrip(" \t")))
        # 取窗口开头连续的中日韩字符（2~4 字，姓氏开头）
        m = re.match(r"[一-鿿]{2,4}", window)
        if m and m.group()[0] in SURNAMES:
            name = m.group()
            if len(name) == 2 or name[1] in GIVEN_NAME_CHARS:
                return [self._candidate(text, name, pad, pad + len(name),
                                        "anchor", name)]
        # 退一步：窗口内任意位置的姓氏开头 2~3 字词
        for m in re.finditer(r"[一-鿿]{2,3}", window):
            word = m.group()
            if word[0] in SURNAMES and (len(word) == 2
                                        or word[1] in GIVEN_NAME_CHARS):
                s = pad + m.start()
                return [self._candidate(text, word, s, s + len(word),
                                        "anchor", word)]
        return []

    def _hits_org(self, text: str, start: int, end: int,
                  source: str = "ner", anchor: bool = False) -> list[dict]:
        hits = self._ner_hits(text, start, end, {"ORGANIZATION"})
        if not hits and anchor:
            # 标签后直接跟机构名但 NER 词典没覆盖：取窗口开头到第一个标点/空白
            window = text[start:end].strip(" \t：:，,。；;、")
            if window:
                m = re.match(r"[一-鿿A-Za-z0-9（）()]{2,30}", window)
                if m:
                    word = m.group()
                    pad = start + (len(text[start:end])
                                   - len(text[start:end].lstrip(" \t")))
                    return [self._candidate(text, word, pad, pad + len(word),
                                            "anchor", word)]
        for h in hits:
            h["source"] = source
        return hits

    def _hits_address(self, text: str, start: int, end: int,
                      source: str = "type", anchor: bool = False) -> list[dict]:
        window = text[start:end].strip(" \t：:，,。；;、")
        if not window:
            return []
        pad = start + (len(text[start:end]) - len(text[start:end].lstrip(" \t")))
        return [self._candidate(text, window, pad, pad + len(window),
                                source, window)]

    def _ner_hits(self, text: str, start: int, end: int,
                  types: set[str]) -> list[dict]:
        out = []
        for ent in self._ner_entities(text):
            if ent["type"] in types and ent["start"] >= start and ent["end"] <= end:
                out.append(self._candidate(
                    text, ent["text"], ent["start"], ent["end"], "ner",
                    ent["text"].strip()))
        return out

    def _regex_hits(self, text: str, pattern: str, start: int, end: int,
                    source: str) -> list[dict]:
        compiled = re.compile(pattern)
        out = []
        for m in compiled.finditer(text, start, end):
            value = m.group(1) if m.groups() else m.group()
            out.append(self._candidate(text, m.group(), m.start(), m.end(),
                                       source, value.strip()))
        return out

    _typed_extractors = {
        "phone": _hits_phone,
        "email": _hits_email,
        "money": _hits_money,
        "date": _hits_date,
        "daterange": _hits_daterange,
        "person": _hits_person,
        "org": _hits_org,
        "address": _hits_address,
    }

    # -- 全文兜底扫描 ------------------------------------------------------
    def _scan_fallback(self, text: str, spec: FieldSpec) -> list[dict]:
        if spec.pattern:
            return self._regex_hits(text, spec.pattern, 0, len(text), "regex")
        extractor = self._typed_extractors.get(spec.type)
        if not extractor:
            # 纯文本字段无法在无标签时推断
            return []
        cands = extractor(self, text, 0, len(text),
                          "ner" if spec.type in ("person", "org") else "type",
                          anchor=False)
        if spec.type == "person":
            # 全文推断只采信词典人名，结构规则（姓氏猜词）误报率高，
            # 没有标签佐证时不拿来填字段
            cands = [c for c in cands
                     if self._ner_source_at(c["start"], c["end"]) == "dict"]
        return cands

    def _ner_source_at(self, start: int, end: int) -> str:
        for ent in self._ner_cache.get("ents", []):
            if ent["start"] == start and ent["end"] == end:
                return ent.get("source", "rule")
        return "rule"

    # -- 候选归并与状态裁决 ------------------------------------------------
    @staticmethod
    def _candidate(text: str, raw: str, start: int, end: int,
                   source: str, value: Any) -> dict:
        return {"raw": raw.strip(), "text": text[start:end],
                "start": start, "end": end, "source": source,
                "value": value}

    def _resolve(self, text: str, spec: FieldSpec,
                 candidates: list[dict], had_anchor: bool,
                 base: dict) -> dict:
        # 按归一化值去重；同一值保留全部出处（多种写法/多处出现）
        merged: dict[str, dict] = {}
        order: list[str] = []
        for cand in candidates:
            key = _norm_key(cand["value"])
            if not key:
                continue
            evidence = self._evidence(cand)
            if key not in merged:
                merged[key] = {
                    "text": cand["raw"], "value": cand["value"],
                    "source": cand["source"],
                    "evidence": [evidence],
                    "raw_variants": [cand["raw"]],
                }
                order.append(key)
            else:
                item = merged[key]
                # 同一处偏移只记一条证据（锚点窗口与全文补扫可能重叠）
                if not any(e["start"] == evidence["start"]
                           and e["end"] == evidence["end"]
                           for e in item["evidence"]):
                    item["evidence"].append(evidence)
                if cand["raw"] not in item["raw_variants"]:
                    item["raw_variants"].append(cand["raw"])
                # 锚点证据可信度高于推断
                if cand["source"] == "anchor" and item["source"] != "anchor":
                    item["source"] = "anchor"
                    item["text"] = cand["raw"]

        if not merged:
            return {**base, "value": None, "status": "missing",
                    "candidates": []}

        distinct = [merged[k] for k in order]

        if spec.multi:
            # 多值字段：多个不同值全部收录，不算歧义
            return {
                **base,
                "value": [d["value"] for d in distinct],
                "status": "found" if had_anchor else "inferred",
                "candidates": distinct,
            }

        if len(distinct) == 1:
            only = distinct[0]
            status = "found" if had_anchor or only["source"] == "anchor" else "inferred"
            return {**base, "value": only["value"], "status": status,
                    "raw_text": only["raw_variants"][0],
                    "candidates": distinct}

        # 多个相互冲突的值：不替用户选一个填上，value 留空并标记 ambiguous
        return {
            **base,
            "value": None,
            "status": "ambiguous",
            "candidates": distinct,
            "message": f"出现 {len(distinct)} 种写法，请人工确认",
        }

    @staticmethod
    def _evidence(cand: dict) -> dict:
        return {"text": cand["raw"], "start": cand["start"],
                "end": cand["end"], "source": cand["source"]}


# ---------------------------------------------------------------------------
# 归一化
# ---------------------------------------------------------------------------

def _norm_key(value: Any) -> str:
    """归并去重用的键（不用于展示）。金额按「币种+元值」归并。"""
    if isinstance(value, dict):
        if "amount" in value:
            return f"{value.get('unit', '')}:{_to_yuan(value['amount'], value.get('unit', '')):.4f}"
        value = value.get("text") or json.dumps(value, ensure_ascii=False)
    return re.sub(r"[\s\-()（）]", "", str(value)).lower()


def _to_yuan(amount: float, unit: str) -> float:
    """把不同量级的金额折算成元，用于跨写法归并。"""
    if unit == "万元":
        return amount * 10000
    if unit == "亿元":
        return amount * 100000000
    return amount


def _canonical_amount(amount: float, unit: str) -> str:
    if unit in ("万元", "亿元"):
        return f"{amount:g}{unit}"
    return f"{amount:g}{unit or '元'}"


_CN_DIGITS = {c: i for i, c in enumerate(
    "零〇一二两三四五六七八九")}
_CN_DIGITS.update({c: i for i, c in enumerate("零壹贰叁肆伍陆柒捌玖")})
_CN_SMALL_UNITS = {"十": 10, "拾": 10, "百": 100, "佰": 100,
                   "千": 1000, "仟": 1000}
_CN_BIG_UNITS = {"万": 10000, "萬": 10000, "亿": 100000000, "億": 100000000}


def _chinese_number(text: str) -> Optional[float]:
    """把简单中文数字（含大写、万/亿）转成数值。如 壹拾贰万 -> 120000。"""
    if not text:
        return None
    if all(c in "0123456789.," for c in text):
        try:
            return float(text.replace(",", ""))
        except ValueError:
            return None
    total, section, current = 0, 0, 0
    saw_cn = False
    for ch in text:
        if ch in _CN_DIGITS:
            current = _CN_DIGITS[ch]
            saw_cn = True
        elif ch in _CN_SMALL_UNITS:
            section += (current or 1) * _CN_SMALL_UNITS[ch]
            current = 0
            saw_cn = True
        elif ch in _CN_BIG_UNITS:
            section += current
            total += section * _CN_BIG_UNITS[ch]
            section, current = 0, 0
            saw_cn = True
    if not saw_cn:
        return None
    return float(total + section + current)


def _normalize_phone(raw: str) -> str:
    """电话归一：保留国家码与数字主体，去掉连字符空白。"""
    for m in _PHONE_RE.finditer(raw):
        return _canonical_phone(m.group())
    return re.sub(r"[-\s]", "", raw)


def _canonical_phone(matched: str) -> str:
    m = re.match(r"(\+?86)?[-\s]?(.*)$", matched)
    digits = re.sub(r"[-\s]", "", m.group(2))
    return f"+86{digits}" if m.group(1) and "+" in m.group(1) else digits


def _normalize_money(raw: str) -> tuple[Optional[float], str]:
    """金额归一：返回 (数值, 单位)。

    同时支持阿拉伯数字（``120,000元``、``12万元``）与中文大写
    （``壹拾贰万元整``）。单位统一为 元/万元/亿元，外币保留币种。
    """
    text = raw.strip()
    currency = ""
    for cur in ("人民币", "美元", "港元", "港币", "欧元", "日元", "英镑"):
        if cur in text:
            currency = "港元" if cur == "港币" else cur
            break

    has_yi = "亿" in text
    has_wan = ("万" in text) or ("萬" in text)
    scale_unit = "亿元" if has_yi else ("万元" if has_wan else "元")

    body = text
    for cur in ("人民币", "美元", "港元", "港币", "欧元", "日元", "英镑"):
        body = body.replace(cur, "")
    # 从末尾剥离货币单位（不能按「万」切：中文数字内部也含「万」字）
    body = re.sub(
        r"(万元|亿元|万人民币|亿人民币|元整|圆整|元|圆"
        r"|美元|港元|港币|欧元|日元|英镑)$", "", body.strip())
    core = body.strip(" ，,。.、：:")
    amount = _chinese_number(core)
    if amount is None:
        return None, ""

    # 数值与单位搭配规则：
    # - 中文数字（壹拾贰万）解析出来是「元」口径，折算回万/亿单位；
    # - 阿拉伯数字按「数字 + 单位」直接表述处理：10万元 -> 10 万元，
    #   120,000元 -> 120000 元。（正则已经把数字和单位绑定，
    #   不存在「120000 + 万元」这种需要乘量级的组合）
    if not re.fullmatch(r"[0-9.,]+", core) and scale_unit != "元":
        amount /= 100000000 if has_yi else 10000

    if currency and currency != "人民币":
        # 外币统一以该币种的「元」为单位
        return amount, currency
    return amount, scale_unit


def _normalize_date(raw: str) -> str:
    """日期归一为 ``YYYY-MM-DD``（缺日则为 ``YYYY-MM``）。"""
    nums = re.findall(r"\d+", raw)
    if len(nums) >= 3 and len(nums[0]) == 4:
        y, mo, d = nums[0], int(nums[1]), int(nums[2])
        return f"{int(y):04d}-{mo:02d}-{d:02d}"
    if len(nums) == 2 and len(nums[0]) == 4:
        y, mo = nums[0], int(nums[1])
        return f"{int(y):04d}-{mo:02d}"
    return raw.strip()


def _normalize_daterange(raw: str) -> dict:
    """期限归一：拆出起止两个日期。"""
    parts = re.split(r"\s*(?:至|到|—|–|~|～|-)\s*|起至\s*", raw, maxsplit=1)
    if len(parts) == 2:
        return {"start": _normalize_date(parts[0]),
                "end": _normalize_date(parts[1]),
                "text": f"{_normalize_date(parts[0])} 至 {_normalize_date(parts[1])}"}
    return {"text": raw.strip()}


# ---------------------------------------------------------------------------
# 内置模板
# ---------------------------------------------------------------------------

BUILTIN_TEMPLATES: list[dict] = [
    {
        "key": "resume",
        "name": "简历模板",
        "description": "姓名、联系方式与工作经历",
        "fields": [
            {"key": "name", "name": "姓名", "type": "person",
             "labels": ["姓名", "名字"], "required": True},
            {"key": "phone", "name": "电话", "type": "phone",
             "labels": ["电话", "手机", "联系电话", "联系方式", "手机号码"],
             "multi": True},
            {"key": "email", "name": "邮箱", "type": "email",
             "labels": ["邮箱", "电子邮箱", "Email", "email", "E-mail"]},
            {"key": "education", "name": "学历", "type": "text",
             "labels": ["学历", "最高学历", "教育背景"]},
            {"key": "experience", "name": "工作经历", "type": "section",
             "labels": ["工作经历", "工作经验", "从业经历", "项目经历"],
             "multi": True},
        ],
    },
    {
        "key": "contract",
        "name": "合同模板",
        "description": "合同双方、金额与期限",
        "fields": [
            {"key": "party_a", "name": "甲方", "type": "org",
             "labels": ["甲方", "甲方（发包方）", "发包方", "买方", "出租方"],
             "required": True},
            {"key": "party_b", "name": "乙方", "type": "org",
             "labels": ["乙方", "承包方", "卖方", "承租方"],
             "required": True},
            {"key": "amount", "name": "合同金额", "type": "money",
             "labels": ["合同金额", "总金额", "金额", "合同总价", "价款", "合同价款"]},
            {"key": "period", "name": "合同期限", "type": "daterange",
             "labels": ["合同期限", "履行期限", "服务期限", "租赁期限", "有效期"]},
            {"key": "sign_date", "name": "签订日期", "type": "date",
             "labels": ["签订日期", "签署日期", "签约日期", "订立日期"]},
        ],
    },
    {
        "key": "notice",
        "name": "通知模板",
        "description": "发文单位、时间与事项",
        "fields": [
            {"key": "org", "name": "发文单位", "type": "org",
             "labels": ["发文单位", "发布单位", "发布机构"]},
            {"key": "date", "name": "发布日期", "type": "date",
             "labels": ["发布日期", "发布时间", "发文日期", "成文日期"]},
            {"key": "title", "name": "标题", "type": "text",
             "labels": ["标题", "通知标题"]},
            {"key": "contact", "name": "联系电话", "type": "phone",
             "labels": ["联系电话", "联系人电话"]},
        ],
    },
]


def get_builtin_templates() -> list[FieldTemplate]:
    return [FieldTemplate.from_dict(t) for t in BUILTIN_TEMPLATES]
