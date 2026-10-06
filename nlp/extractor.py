"""基于用户字段模板的信息抽取（IE）。

面向简历、合同、通知等**半结构化文本**：用户定义一套字段模板（字段名、
字段类型、同义写法、所属章节等），抽取器从原文把字段值抽成规整记录。

设计要点（对应业务诉求）
------------------------
1. **多种写法归一**：电话 ``138-1234-5678`` / ``+86 138...``、金额
   ``人民币壹拾贰万元整`` / ``120,000元`` 等，先各自匹配再做 *规范化*，
   规范值相同视为同一值的不同写法；规范值不同则标 ``conflict``（多处不一致），
   全部候选都保留，绝不悄悄挑一个填进去。
2. **缺失显式标记**：抽不到就是 ``status="missing"``、``value=None``，
   字段照样出现在结果里，由上层在表格中标「缺失」，而不是填错值 / 空串。
3. **逐处可回查**：每个命中（mention）都带原文的字符偏移 ``start/end``、
   命中文本、上下文片段与命中方式（标签 / 正则 / 实体 / 章节），
   前端可直接在原文上高亮回查；``text[start:end] == raw`` 恒成立。
4. **模板互不干扰**：抽取结果与 *模板版本 id* 绑定并快照模板字段，
   新增模板、调整字段产生新版本，老结果原样保留。

抽取策略（纯规则，与本工程其它 NLP 模块风格一致）
-------------------------------------------------
- ``label``  标签锚定：``姓名：张三``、``甲方（出租方）：XX公司``、
  ``金额为人民币……`` 等「同义标签 + 分隔符 + 值」。
- ``type``   类型正则：电话 / 邮箱 / 身份证 / 金额（含大写）/ 日期，
  仅在该字段没有标签命中时兜底。
- ``ner``    实体兜底：人名 / 机构 / 地名字段用 :class:`~nlp.ner.NERExtractor`。
- ``section`` 章节抽取：工作经历、项目经历这类列表字段，按章节标题切块，
  再按日期块 / 列表符号 / 换行拆成一条条记录。
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field as dc_field
from typing import Any, Optional

from .ner import NERExtractor

EXTRACTOR_VERSION = "1.0.0"

# 字段类型 -> 中文名
FIELD_TYPES = {
    "person": "人名",
    "org": "机构",
    "location": "地名",
    "phone": "电话",
    "email": "邮箱",
    "idcard": "身份证号",
    "money": "金额",
    "date": "日期",
    "text": "文本",
    "list": "条目列表",
}

# ---------------------------------------------------------------------------
# 类型正则
# ---------------------------------------------------------------------------

_PHONE_MOBILE = re.compile(
    r"(?:\+?86[-\s]?)?1[3-9]\d(?:[-\s]?\d){8}(?!\d)")
_PHONE_TEL = re.compile(r"(?<!\d)0\d{2,3}[-\s]?\d{7,8}(?:[-\s]?\d{1,5})?(?!\d)")
_EMAIL = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
_IDCARD = re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)")

# 阿拉伯数字金额：必须带币种前缀或「元/万/亿/块」等单位，避免误吞普通数字
_MONEY_CN_UNIT = re.compile(
    r"(?:人民币|RMB|rmb)?\s*[\d][\d,]*(?:\.\d+)?\s*(?:万元|亿元|万|亿|元整|元|块钱|块)"
)
_MONEY_PREFIX = re.compile(
    r"(?:美元|美金|欧元|港元|港币|日元|英镑|人民币)\s*[\d][\d,]*(?:\.\d+)?\s*(?:万元|亿元|万|亿|元)?"
)
# 中文大写金额
_CN_DIGITS = "零壹贰叁肆伍陆柒捌玖两"
_MONEY_CN = re.compile(
    r"(?:人民币)?[" + _CN_DIGITS + r"拾佰仟万亿]{2,}(?:元(?:整)?)?"
)

# 日期
_DATE_FULL = re.compile(r"\d{4}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日?")
_DATE_FULL_SEP = re.compile(r"\d{4}[-/.]\d{1,2}[-/.]\d{1,2}")
_DATE_MD = re.compile(r"(?<!\d)\d{1,2}\s*月\s*\d{1,2}\s*日")
_DATE_YM = re.compile(r"\d{4}\s*年\s*\d{1,2}\s*月")
_DATE_ANY = re.compile(
    r"\d{4}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日?"
    r"|\d{4}[-/.]\d{1,2}[-/.]\d{1,2}"
    r"|(?<!\d)\d{1,2}\s*月\s*\d{1,2}\s*日"
    r"|\d{4}\s*年\s*\d{1,2}\s*月"
)

# 标签与值之间允许的分隔符 / 空白
_LABEL_SEP = ":："
_DIRECT_GAP_CHARS = " \t　"
_STOP_PUNCT = "\n\r;；。！？!?|"
# 非列表字段遇到逗号也截断
_STOP_PUNCT_SHORT = _STOP_PUNCT + ",，、"
_TAIL_STRIP = " \t　\"'“”‘’()（）[]【】*＊·.。,，;；、与和及"

# 「甲方（出租方）」括号里的角色名，不是字段值
_ROLE_WORDS = {
    "出租方", "承租方", "发包方", "承包方", "买方", "卖方", "委托方",
    "受托方", "借款方", "贷款方", "用人方", "用工方", "劳动者", "用人单位",
    "采购方", "供应方", "供货方", "出让方", "转让方", "受让方", "担保方",
    "抵押人", "抵押权人", "债权人", "债务人", "定作方", "承揽方", "托运方",
    "承运方", "赠与人", "受赠人", "出借人", "借款人", "保证人",
}

# 列表章节切分时，跨字段的通用终止标题
_GENERAL_SECTION_HEADS = [
    "自我评价", "个人简介", "自我描述", "专业技能", "技能特长", "职业技能",
    "证书", "荣誉", "获奖", "求职意向", "期望职业", "联系方式", "基本信息",
    "个人信息", "兴趣爱好", "特长爱好", "培训经历", "备注", "附言", "附件",
    "参考文献", "致谢", "签字", "盖章", "落款",
]

# 括号里出现的「动作/落款」词，不是字段值：甲方（盖章）、乙方（签字）
_BRACKET_ACTIONS = {
    "盖章", "签章", "签字", "签名", "盖章处", "签字处", "指印", "公章",
    "合同章", "骑缝章", "此处盖章", "此处签字", "预留", "盖章位置",
}

_METHOD_RANK = {"label": 3, "type": 2, "ner": 1, "section": 2}

_CN_NUM = {"零": 0, "壹": 1, "贰": 2, "叁": 3, "肆": 4,
           "伍": 5, "陆": 6, "柒": 7, "捌": 8, "玖": 9, "两": 2}
_CN_SMALL_UNIT = {"拾": 10, "佰": 100, "仟": 1000}


@dataclass
class Mention:
    """一处原文命中。``text[start:end] == raw`` 恒成立。"""
    raw: str
    start: int
    end: int
    method: str                       # label / type / ner / section
    confidence: float
    normalized: Optional[str] = None  # 规范化后的可比值
    context: str = ""
    extra: dict = dc_field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {
            "raw": self.raw, "start": self.start, "end": self.end,
            "method": self.method, "confidence": self.confidence,
            "context": self.context,
        }
        if self.normalized is not None:
            d["normalized"] = self.normalized
        if self.extra:
            d.update(self.extra)
        return d


# ---------------------------------------------------------------------------
# 规范化工具
# ---------------------------------------------------------------------------

def normalize_phone(raw: str) -> str:
    digits = re.sub(r"\D", "", raw)
    if digits.startswith("86") and len(digits) == 13:
        digits = digits[2:]
    return digits


def normalize_email(raw: str) -> str:
    return raw.strip().lower()


def _cn_amount_to_number(s: str) -> Optional[float]:
    """把「壹拾贰万」这类中文大写金额转成数字。"""
    total = 0.0
    section = 0.0
    num = 0.0
    found = False
    for ch in s:
        if ch in _CN_NUM:
            num = _CN_NUM[ch]
            found = True
        elif ch in _CN_SMALL_UNIT:
            section += (num if num else 1) * _CN_SMALL_UNIT[ch]
            num = 0
            found = True
        elif ch == "万":
            section = (section + num) * 10_000
            total += section
            section, num = 0.0, 0.0
            found = True
        elif ch == "亿":
            section = (section + num) * 100_000_000
            total += section
            section, num = 0.0, 0.0
            found = True
        elif ch in ("元", "整", "人民币"):
            continue
    total += section + num
    return total if found else None


def parse_money(raw: str) -> Optional[dict]:
    """解析金额文本，返回 {amount, currency}；无法解析返回 None。"""
    text = raw.strip().replace(",", "").replace("，", "")
    currency = "CNY"
    for name, code in (("美元", "USD"), ("美金", "USD"), ("欧元", "EUR"),
                       ("港币", "HKD"), ("港元", "HKD"), ("日元", "JPY"),
                       ("英镑", "GBP")):
        if name in text:
            currency = code
            break
    m = re.search(r"[\d]+(?:\.\d+)?", text)
    amount: Optional[float] = None
    if m:
        amount = float(m.group())
        if "亿" in text:
            amount *= 100_000_000
        elif "万" in text:
            amount *= 10_000
    else:
        amount = _cn_amount_to_number(text)
    if amount is None:
        return None
    return {"amount": round(amount, 2), "currency": currency}


def _money_matches(text: str) -> list[Mention]:
    out: list[Mention] = []
    for m in _MONEY_CN.finditer(text):
        parsed = parse_money(m.group())
        if parsed:
            out.append(Mention(
                m.group(), m.start(), m.end(), "type", 0.9,
                normalized=f"{parsed['amount']:g}|{parsed['currency']}",
                extra={"amount": parsed["amount"], "currency": parsed["currency"]}))
    occupied = [(m.start, m.end) for m in out]
    for pat in (_MONEY_PREFIX, _MONEY_CN_UNIT):
        for m in pat.finditer(text):
            if any(s < m.end() and m.start() < e for s, e in occupied):
                continue
            parsed = parse_money(m.group())
            if not parsed:
                continue
            occupied.append((m.start(), m.end()))
            out.append(Mention(
                m.group().strip(), m.start(), m.end(), "type", 0.9,
                normalized=f"{parsed['amount']:g}|{parsed['currency']}",
                extra={"amount": parsed["amount"], "currency": parsed["currency"]}))
    out.sort(key=lambda x: x.start)
    return out


def parse_date(raw: str) -> Optional[str]:
    """规范化日期为 ISO 形式：``2024-01-05`` / ``2024-01`` / ``01-05``。"""
    m = _DATE_FULL.match(raw.strip())
    if m:
        nums = re.findall(r"\d+", raw)
        y, mo, d = int(nums[0]), int(nums[1]), int(nums[2])
        return f"{y:04d}-{mo:02d}-{d:02d}"
    m = _DATE_FULL_SEP.match(raw.strip())
    if m:
        nums = re.findall(r"\d+", raw)
        y, mo, d = int(nums[0]), int(nums[1]), int(nums[2])
        return f"{y:04d}-{mo:02d}-{d:02d}"
    m = _DATE_YM.search(raw)
    if m and not re.search(r"\d{1,2}\s*日", raw):
        nums = re.findall(r"\d+", m.group())
        return f"{int(nums[0]):04d}-{int(nums[1]):02d}"
    m = _DATE_MD.search(raw)
    if m:
        nums = re.findall(r"\d+", m.group())
        return f"{int(nums[0]):02d}-{int(nums[1]):02d}"
    return None


def _date_matches(text: str) -> list[Mention]:
    out = []
    for m in _DATE_ANY.finditer(text):
        norm = parse_date(m.group())
        if norm:
            out.append(Mention(m.group(), m.start(), m.end(), "type",
                               0.85, normalized=norm))
    return out


def _generic_matches(text: str, ftype: str) -> list[Mention]:
    if ftype == "phone":
        out = []
        for pat in (_PHONE_MOBILE, _PHONE_TEL):
            for m in pat.finditer(text):
                out.append(Mention(m.group(), m.start(), m.end(), "type", 0.95,
                                   normalized=normalize_phone(m.group())))
        out.sort(key=lambda x: (x.start, -(x.end - x.start)))
        # 去掉区间重叠（座机号可能是手机号的子串）
        dedup = []
        for men in out:
            if any(s < men.end and men.start < e for s, e in
                   [(d.start, d.end) for d in dedup]):
                continue
            dedup.append(men)
        return dedup
    if ftype == "email":
        return [Mention(m.group(), m.start(), m.end(), "type", 0.95,
                        normalized=normalize_email(m.group()))
                for m in _EMAIL.finditer(text)]
    if ftype == "idcard":
        return [Mention(m.group(), m.start(), m.end(), "type", 0.95,
                        normalized=m.group().upper())
                for m in _IDCARD.finditer(text)]
    if ftype == "money":
        return _money_matches(text)
    if ftype == "date":
        return _date_matches(text)
    return []


def text_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# 模板
# ---------------------------------------------------------------------------

class TemplateError(ValueError):
    """模板定义不合法。"""


class FieldTemplate:
    """单个字段的抽取配置。

    ``spec`` 支持的键：

    - ``key``        字段英文标识（必填，模板内唯一）
    - ``label``      字段中文名
    - ``type``       见 :data:`FIELD_TYPES`，默认 ``text``
    - ``required``   是否必填（仅影响缺失标记的严重程度）
    - ``aliases``    同义标签写法，如 ["电话", "手机", "联系方式"]
    - ``patterns``   自定义正则（可选；命中即作为标签级证据）
    - ``sections``   章节标题（``type=list`` 时生效）
    - ``multi``      是否多值（list 类型隐含 multi）
    - ``date_order`` 日期区间中取 ``first`` / ``last``
    - ``max_length`` 标签值最多抓取字符数
    """

    def __init__(self, spec: dict):
        key = str(spec.get("key") or "").strip()
        if not key:
            raise TemplateError("字段缺少 key")
        if not re.match(r"^[A-Za-z][A-Za-z0-9_]*$", key):
            raise TemplateError(f"字段 key 只能用字母数字下划线: {key}")
        ftype = spec.get("type") or "text"
        if ftype not in FIELD_TYPES:
            raise TemplateError(f"字段 {key} 的类型不支持: {ftype}")

        self.key = key
        self.label = str(spec.get("label") or key).strip()
        self.type = ftype
        self.required = bool(spec.get("required", False))
        self.aliases = [a for a in spec.get("aliases", []) if str(a).strip()]
        self.patterns = [p for p in spec.get("patterns", []) if str(p).strip()]
        self.sections = [s for s in spec.get("sections", []) if str(s).strip()]
        self.multi = bool(spec.get("multi", False)) or ftype == "list"
        self.date_order = spec.get("date_order") or "first"
        self.max_length = int(spec.get("max_length") or 80)
        self.fallback_positions = list(spec.get("fallback_positions", []))
        self.spec = self.dump()

    def dump(self) -> dict:
        return {
            "key": self.key, "label": self.label, "type": self.type,
            "required": self.required, "aliases": list(self.aliases),
            "patterns": list(self.patterns), "sections": list(self.sections),
            "multi": self.multi, "date_order": self.date_order,
            "max_length": self.max_length,
            "fallback_positions": list(self.fallback_positions),
        }


class Template:
    """一套字段模板（简历 / 合同 / 通知……）。"""

    def __init__(self, spec: dict):
        name = str(spec.get("name") or "").strip()
        if not name:
            raise TemplateError("模板缺少名称")
        fields = [FieldTemplate(f) for f in spec.get("fields", [])]
        if not fields:
            raise TemplateError(f"模板「{name}」至少需要一个字段")
        keys = [f.key for f in fields]
        if len(set(keys)) != len(keys):
            raise TemplateError(f"模板「{name}」字段 key 重复")

        self.name = name
        self.description = str(spec.get("description") or "").strip()
        self.fields = fields
        self._by_key = {f.key: f for f in fields}
        # 标签截断集合：抓值时遇到别的字段标签就停，防止吞到下一个字段
        self.stop_aliases = sorted(
            {a for f in fields for a in f.aliases}, key=len, reverse=True)
        self.all_sections = sorted(
            {s for f in fields for s in f.sections} | set(_GENERAL_SECTION_HEADS),
            key=len, reverse=True)

    def get(self, key: str) -> Optional[FieldTemplate]:
        return self._by_key.get(key)

    def dump(self) -> dict:
        return {"name": self.name, "description": self.description,
                "fields": [f.dump() for f in self.fields]}


# ---------------------------------------------------------------------------
# 抽取器
# ---------------------------------------------------------------------------

class FieldExtractor:
    """按 :class:`Template` 从文本抽取字段值。"""

    def __init__(self, ner: Optional[NERExtractor] = None):
        self.ner = ner or NERExtractor()

    # -- 主入口 -----------------------------------------------------------
    def extract(self, text: str, template_spec: dict) -> dict:
        template = Template(template_spec)
        return self.extract_template(text, template)

    def extract_template(self, text: str, template: Template) -> dict:
        if not isinstance(text, str):
            raise TypeError("text 必须是字符串")
        ner_entities = self.ner.recognize(text)
        result_fields: dict[str, dict] = {}
        missing: list[str] = []
        conflicts: list[str] = []

        for f in template.fields:
            fr = self._extract_field(text, f, template, ner_entities)
            result_fields[f.key] = fr
            if fr["status"] == "missing":
                missing.append(f.key)
            elif fr["status"] == "conflict":
                conflicts.append(f.key)

        return {
            "tpl_name": template.name,
            "fields": result_fields,
            "missing": missing,
            "missing_required": [k for k in missing
                                 if template.get(k).required],
            "conflicts": conflicts,
            "text_hash": text_hash(text),
            "extractor_version": EXTRACTOR_VERSION,
        }

    # -- 单字段 -----------------------------------------------------------
    def _extract_field(self, text: str, f: FieldTemplate,
                       template: Template, ner_entities: list[dict]) -> dict:
        base = {
            "key": f.key, "label": f.label, "type": f.type,
            "required": f.required, "multi": f.multi,
        }
        if f.type == "list":
            mentions = self._section_items(text, f, template)
            # 章节没切出来时，标签行里的值也算一条
            if not mentions:
                mentions = [m for m in self._label_anchors(text, f, template)
                            if m.raw]
            return self._assemble(base, mentions, text, multi=True)

        label_mentions = self._label_anchors(text, f, template)
        custom_mentions = self._custom_patterns(text, f)

        primary = label_mentions + custom_mentions
        if not primary:
            primary = self._fallback(text, f, ner_entities)

        return self._assemble(base, primary, text, multi=f.multi)

    # -- 标签锚定 ---------------------------------------------------------
    def _label_anchors(self, text: str, f: FieldTemplate,
                       template: Template) -> list[Mention]:
        out: list[Mention] = []
        for alias in f.aliases:
            flags = re.IGNORECASE if alias.isascii() else 0
            for m in re.finditer(re.escape(alias), text, flags):
                men = self._read_value_after(text, m.end(), f, template)
                if men is not None:
                    out.append(men)
        out.sort(key=lambda x: x.start)
        return self._dedupe_mentions(out)

    def _read_value_after(self, text: str, pos: int, f: FieldTemplate,
                          template: Template) -> Optional[Mention]:
        n = len(text)
        j = pos
        while j < n and text[j] in _DIRECT_GAP_CHARS:
            j += 1

        # 别名落在括号里角色名的末尾：「甲方（出租方）：XX」
        if j < n and text[j] in "）)】》>］]":
            j += 1
            while j < n and text[j] in _DIRECT_GAP_CHARS:
                j += 1

        # 括号包裹：姓名【张三】 / 甲方（出租方）→ 角色名要跳过
        if j < n and text[j] in "【[（(《<":
            close = {"【": "】", "[": "]", "（": "）", "(": ")",
                     "《": "》", "<": ">"}[text[j]]
            k = text.find(close, j + 1)
            if k != -1 and k - j <= 40:
                inner = text[j + 1:k].strip()
                if (inner in _ROLE_WORDS or inner.endswith("方")
                        or inner in _BRACKET_ACTIONS
                        or any(w in inner for w in ("盖章", "签字", "签章"))):
                    j = k + 1
                    while j < n and text[j] in _DIRECT_GAP_CHARS:
                        j += 1
                    # 跳过后紧跟分隔符（「甲方（出租方）：XX」）也要消费掉
                    if j < n and text[j] in _LABEL_SEP:
                        j += 1
                    while j < n and text[j] in _DIRECT_GAP_CHARS:
                        j += 1
                else:
                    return self._build_typed_or_text(
                        text, f, inner, j + 1, k, "label", 0.95)

        # 分隔符
        if j < n and text[j] in _LABEL_SEP:
            j += 1
        elif j < n and text[j] in "为是系":
            j += 1
        elif j < n and text[j] not in _STOP_PUNCT and text[j] not in "，,、":
            pass  # 允许「姓名张三」这种无分隔符写法
        else:
            return None
        while j < n and text[j] in _DIRECT_GAP_CHARS:
            j += 1

        end = self._value_end(text, j, f, template)
        raw = text[j:end].strip(_TAIL_STRIP)
        if not raw:
            return None
        # 截掉值内部撞到的别的字段标签（如「张三 性别：男」）
        raw = self._trim_at_alias(raw, template)
        if not raw:
            return None
        start = j + text[j:end].find(raw)
        end = start + len(raw)
        return self._build_typed_or_text(text, f, raw, start, end,
                                         "label", 0.95)

    def _value_end(self, text: str, start: int, f: FieldTemplate,
                   template: Template) -> int:
        punct = _STOP_PUNCT if f.type == "text" else _STOP_PUNCT_SHORT
        limit = min(len(text), start + f.max_length)
        for i in range(start, limit):
            ch = text[i]
            if ch in punct:
                return i
            if ch in _DIRECT_GAP_CHARS:
                # 连续两个空格常意味着进入下一个字段
                nxt = i + 1
                while nxt < len(text) and text[nxt] in _DIRECT_GAP_CHARS:
                    nxt += 1
                if nxt - i >= 2 and nxt < len(text):
                    return i
        return limit

    def _trim_at_alias(self, raw: str, template: Template) -> str:
        """截掉值内部撞到的后续标签（如「张三 性别：男」里的「性别：」）。

        只有别名后面**确实跟着标签分隔符/判断词**才算撞上，避免
        「自 2024 年起……」里的单字别名「自」「至」造成误截断。
        """
        cut = len(raw)
        for alias in template.stop_aliases:
            idx = raw.find(alias)
            while idx != -1:
                tail = raw[idx + len(alias):]
                stripped = tail.lstrip(" \t　")
                gap = len(tail) - len(stripped)
                follower = stripped[:1]
                if (follower and (follower in ":：为是系"
                                  or stripped.startswith("为")
                                  or stripped.startswith("是"))) \
                        or (gap >= 1 and idx + len(alias) + gap < len(raw)
                            and len(alias) >= 2):
                    cut = min(cut, idx)
                    break
                idx = raw.find(alias, idx + 1)
        return raw[:cut].strip(_TAIL_STRIP)

    # -- 自定义正则 / 类型化包装 / 兜底 -----------------------------------
    def _custom_patterns(self, text: str, f: FieldTemplate) -> list[Mention]:
        out = []
        for pat in f.patterns:
            try:
                rx = re.compile(pat)
            except re.error:
                continue
            for m in rx.finditer(text):
                raw = m.group(1) if m.groups() else m.group()
                out.append(Mention(raw, m.start(), m.end(), "label", 0.9,
                                   normalized=self._normalize(f, raw)))
        return out

    def _fallback(self, text: str, f: FieldTemplate,
                  ner_entities: list[dict]) -> list[Mention]:
        typed = _generic_matches(text, f.type)
        if typed:
            for men in typed:
                men.confidence = 0.75
                men.method = "type"
            return typed
        ner_map = {"person": "PERSON", "org": "ORGANIZATION",
                   "location": "LOCATION"}
        etype = ner_map.get(f.type)
        if etype:
            return [Mention(text[e["start"]:e["end"]], e["start"], e["end"],
                            "ner", 0.55,
                            normalized=text[e["start"]:e["end"]].strip())
                    for e in ner_entities if e["type"] == etype]
        if f.type == "text" and "first_line" in f.fallback_positions:
            for line_m in re.finditer(r"[^\n\r]+", text):
                line = line_m.group().strip(" \t　")
                if line:
                    return [Mention(line, line_m.start(),
                                    line_m.start() + len(line_m.group()),
                                    "position", 0.4, normalized=line)]
        return []

    def _build_typed_or_text(self, text: str, f: FieldTemplate, raw: str,
                             start: int, end: int, method: str,
                             confidence: float) -> Mention:
        """标签抓到的值，若是强类型字段则在窗口内再取结构化片段。"""
        window = raw
        w_start = start
        if f.type == "money":
            ms = _money_matches(window)
            if ms:
                pick = ms[0]
                return Mention(pick.raw, w_start + pick.start,
                               w_start + pick.end, method, confidence,
                               normalized=pick.normalized, extra=pick.extra)
        if f.type == "date":
            ds = _date_matches(window)
            if ds:
                pick = ds[0] if f.date_order != "last" else ds[-1]
                return Mention(pick.raw, w_start + pick.start,
                               w_start + pick.end, method, confidence,
                               normalized=pick.normalized)
        if f.type == "phone":
            for pat in (_PHONE_MOBILE, _PHONE_TEL):
                mm = pat.search(window)
                if mm:
                    return Mention(mm.group(), w_start + mm.start(),
                                   w_start + mm.end(), method, confidence,
                                   normalized=normalize_phone(mm.group()))
        if f.type == "email":
            mm = _EMAIL.search(window)
            if mm:
                return Mention(mm.group(), w_start + mm.start(),
                               w_start + mm.end(), method, confidence,
                               normalized=normalize_email(mm.group()))
        if f.type == "idcard":
            mm = _IDCARD.search(window)
            if mm:
                return Mention(mm.group(), w_start + mm.start(),
                               w_start + mm.end(), method, confidence,
                               normalized=mm.group().upper())
        normalized = self._normalize(f, raw)
        return Mention(raw, start, end, method, confidence,
                       normalized=normalized)

    @staticmethod
    def _normalize(f: FieldTemplate, raw: str) -> str:
        val = re.sub(r"\s+", "", raw.strip())
        if f.type in ("person", "org", "location", "text"):
            return re.sub(r"\s+", " ", raw.strip())
        return val

    # -- 章节列表 ---------------------------------------------------------
    def _section_items(self, text: str, f: FieldTemplate,
                       template: Template) -> list[Mention]:
        mentions: list[Mention] = []
        for marker in f.sections:
            head = self._find_section_head(text, marker)
            if head is None:
                continue
            body_start, body_end = self._section_body(text, head, template)
            mentions.extend(self._split_items(text, body_start, body_end))
        return self._dedupe_mentions(mentions)

    @staticmethod
    def _find_section_head(text: str, marker: str) -> Optional[tuple[int, int]]:
        rx = re.compile(r"(?:^|[\n\r])[ \t　]*" + re.escape(marker)
                        + r"\s*[:：]?[ \t　]*(?=[\n\r]|$)", re.M)
        m = rx.search(text)
        if m:
            return m.start(), m.end()
        # 行内标题：「工作经历：……」
        rx2 = re.compile(re.escape(marker) + r"\s*[:：]")
        m = rx2.search(text)
        if m:
            return m.start(), m.end()
        return None

    def _section_body(self, text: str, head: tuple[int, int],
                      template: Template) -> tuple[int, int]:
        head_text = text[head[0]:head[1]].strip(" :：\n\r \t　")
        body_start = head[1]
        body_end = len(text)
        for other in template.all_sections:
            if other == head_text:
                continue
            # 在 body_start 之后找下一个同级标题
            rx = re.compile(r"(?:^|[\n\r])[ \t　]*" + re.escape(other)
                            + r"\s*[:：]?[ \t　]*(?=[\n\r]|$)", re.M)
            m = rx.search(text, body_start)
            if m and m.start() < body_end:
                body_end = m.start()
        return body_start, body_end

    def _split_items(self, text: str, start: int, end: int) -> list[Mention]:
        body = text[start:end]
        if not body.strip():
            return []

        # 1) 显式列表符号
        bullet = re.compile(
            r"(?:^|[\n\r])[ \t　]*"
            r"(?:[•·●○◆■▪◦*＊\-–—]|\d+[.、)]|[①②③④⑤⑥⑦⑧⑨⑩])\s*", re.M)
        marks = list(bullet.finditer(body))
        if len(marks) >= 2:
            return self._blocks_to_mentions(body, start,
                                            [(m.end(),
                                              (marks[i + 1].start()
                                               if i + 1 < len(marks) else len(body)))
                                             for i, m in enumerate(marks)])

        # 2) 日期块：每段以日期（区间）开头
        date_begin = re.compile(r"(?:^|[\n\r])[ \t　]*\d{4}\s*"
                                r"(?:年|[-/.])", re.M)
        dm = list(date_begin.finditer(body))
        if len(dm) >= 2:
            return self._blocks_to_mentions(body, start,
                                            [(m.start(),
                                              (dm[i + 1].start()
                                               if i + 1 < len(dm) else len(body)))
                                             for i, m in enumerate(dm)])

        # 3) 换行拆条；不像条目开头的续行并到上一条
        mentions: list[Mention] = []
        for line_m in re.finditer(r"[^\n\r]+", body):
            line = line_m.group().strip(" \t　")
            if not line:
                continue
            ls, le = start + line_m.start(), start + line_m.end()
            stripped = line.lstrip(" \t　")
            ls += len(line) - len(stripped)
            if mentions and not self._looks_like_item_start(stripped):
                prev = mentions[-1]
                merged = (text[prev.start:prev.end] + " "
                          + re.sub(r"\s+", " ", stripped))
                prev.raw = merged
                prev.end = le
                prev.normalized = merged
            else:
                clean = re.sub(r"\s+", " ", stripped)
                mentions.append(Mention(clean, ls, le, "section", 0.8,
                                        normalized=clean))
        return mentions

    @staticmethod
    def _looks_like_item_start(line: str) -> bool:
        if re.match(r"(?:\d{4}\s*年|\d{4}[-/.]|\d{1,2}\s*月)", line):
            return True
        if re.match(r"(?:[•·●○◆■▪◦*＊]|\d+[.、)]|[①②③④⑤⑥⑦⑧⑨⑩])", line):
            return True
        return False

    def _blocks_to_mentions(self, body: str, base: int,
                            blocks: list[tuple[int, int]]) -> list[Mention]:
        out = []
        for s, e in blocks:
            seg = body[s:e]
            lead = len(seg) - len(seg.lstrip(" \t　"))
            raw = re.sub(r"\s+", " ", seg.strip(_TAIL_STRIP))
            if raw:
                out.append(Mention(raw, base + s + lead,
                                   base + s + lead + len(raw),
                                   "section", 0.85, normalized=raw))
        return out

    # -- 汇总 -------------------------------------------------------------
    def _assemble(self, base: dict, mentions: list[Mention], text: str,
                  multi: bool) -> dict:
        mentions = self._dedupe_mentions(mentions)
        for men in mentions:
            if not men.context:
                men.context = self._context(text, men.start, men.end)
            # 恒等式校验：偏移必须对得上原文
            if text[men.start:men.end] != men.raw:
                men.raw = text[men.start:men.end]

        if not mentions:
            return {**base, "status": "missing", "value": None, "raw": None,
                    "normalized": None, "mentions": [], "candidates": []}

        if multi:
            ordered_men = sorted(mentions, key=lambda x: x.start)
            values = [m.normalized if m.normalized is not None else m.raw
                      for m in ordered_men]
            return {**base, "status": "found", "value": values,
                    "raw": [m.raw for m in ordered_men],
                    "normalized": values,
                    "mentions": [m.to_dict() for m in ordered_men],
                    "candidates": []}

        # 单值：按规范化值分组
        groups: dict[str, list[Mention]] = {}
        for men in mentions:
            key = men.normalized if men.normalized is not None else men.raw
            groups.setdefault(key, []).append(men)

        def rank(ms: list[Mention]) -> tuple[int, int]:
            best = max(ms, key=lambda x: (_METHOD_RANK.get(x.method, 0),
                                          -x.start))
            return _METHOD_RANK.get(best.method, 0), -best.start

        ordered = sorted(groups.values(),
                         key=lambda ms: (rank(ms), -len(ms)), reverse=True)
        chosen = ordered[0]
        best = max(chosen, key=lambda x: (_METHOD_RANK.get(x.method, 0),
                                          -x.start))
        result = {
            **base,
            "raw": best.raw,
            "normalized": best.normalized
            if best.normalized is not None else best.raw,
            "mentions": [m.to_dict() for m in
                         sorted(mentions, key=lambda x: x.start)],
            "candidates": [],
        }
        if len(groups) > 1:
            result["status"] = "conflict"
            result["candidates"] = [
                {"value": (ms[0].normalized if ms[0].normalized is not None
                           else ms[0].raw),
                 "raw": ms[0].raw, "count": len(ms),
                 "methods": sorted({m.method for m in ms})}
                for ms in ordered
            ]
            # 冲突时不把猜测值当定论：value 置空，候选全部列出
            result["value"] = None
        else:
            result["status"] = "found"
            result["value"] = result["normalized"]
        return result

    @staticmethod
    def _context(text: str, start: int, end: int, radius: int = 14) -> str:
        s, e = max(0, start - radius), min(len(text), end + radius)
        snippet = text[s:e].replace("\n", " ").replace("\r", " ")
        return re.sub(r"\s+", " ", snippet).strip()

    @staticmethod
    def _dedupe_mentions(mentions: list[Mention]) -> list[Mention]:
        """同一处原文的重复命中去重。

        - 「甲方」「出租方」两个别名锚到同一个值（区间重叠且规范值相同），
          只保留证据最强的一条；
        - 标签锚定抓到的宽值（``AB123456，请审批``）**包住**了自定义正则/
          强类型的精确命中（``AB123456``）时，宽值属于同一次命中的噪声，
          被精确命中吸收，避免制造假冲突。
        """
        # 证据强的先入列（自定义正则 rank=3 > 普通标签行 rank=3），
        # 再按跨度短（更精确）优先；这样后处理的宽命中可以被精确命中吸收
        ordered = sorted(
            mentions,
            key=lambda m: (-_METHOD_RANK.get(m.method, 0),
                           (m.end - m.start), m.start))
        kept: list[Mention] = []
        for men in ordered:
            key_norm = men.normalized if men.normalized is not None else men.raw
            merged = False

            # 1) 与已有命中重叠且规范值相同：保留证据更强的
            for i, prev in enumerate(kept):
                prev_norm = prev.normalized if prev.normalized is not None \
                    else prev.raw
                if (men.start < prev.end and prev.start < men.end
                        and prev_norm == key_norm):
                    if (_METHOD_RANK.get(men.method, 0)
                            > _METHOD_RANK.get(prev.method, 0)):
                        kept[i] = men
                    merged = True
                    break
            if merged:
                continue

            # 2) 噪声吸收：两个命中区间重叠，且一个的规范值是另一个规范值
            #    的子串（短的那个是正则/类型化得到的干净值，长的是标签行
            #    抓到的宽值），则丢弃证据不更强的「宽」命中
            absorbed = False
            for ref in kept:
                prev_norm = ref.normalized if ref.normalized is not None \
                    else ref.raw
                overlap = men.start < ref.end and ref.start < men.end
                if not overlap or not prev_norm or not key_norm:
                    continue
                if key_norm == prev_norm:
                    continue  # 完全相等走规则 1
                short, long_ = sorted([key_norm, prev_norm], key=len)
                if short in long_ and len(short) >= 3:
                    ref_rank = _METHOD_RANK.get(ref.method, 0)
                    men_rank = _METHOD_RANK.get(men.method, 0)
                    # 同等证据：短者胜；证据有强弱：强者胜
                    short_is_ref = len(prev_norm) < len(key_norm)
                    if (ref_rank > men_rank
                            or (ref_rank == men_rank and short_is_ref)):
                        absorbed = True
                        break
                    if men_rank > ref_rank:
                        # 当前命中更强 -> 替换旧命中
                        kept[kept.index(ref)] = men
                        absorbed = True
                        break
            if absorbed:
                continue
            if not merged:
                kept.append(men)
        return kept


# ---------------------------------------------------------------------------
# 预置模板
# ---------------------------------------------------------------------------

BUILTIN_TEMPLATES: list[dict] = [
    {
        "key": "builtin_resume",
        "name": "简历",
        "description": "抽取姓名、联系方式、教育背景与工作/项目经历。",
        "fields": [
            {"key": "name", "label": "姓名", "type": "person",
             "required": True,
             "aliases": ["姓名", "应聘者", "候选人", "求职人", "本人姓名",
                         "名字"]},
            {"key": "gender", "label": "性别", "type": "text",
             "aliases": ["性别"], "max_length": 4},
            {"key": "phone", "label": "电话", "type": "phone",
             "required": True,
             "aliases": ["电话", "手机", "联系电话", "手机号码",
                         "联系方式", "移动电话", "联系手机"]},
            {"key": "email", "label": "邮箱", "type": "email",
             "aliases": ["邮箱", "电子邮箱", "电子邮件", "Email", "E-mail",
                         "email"]},
            {"key": "idcard", "label": "身份证号", "type": "idcard",
             "aliases": ["身份证号", "身份证", "证件号码", "身份证号码"]},
            {"key": "education", "label": "最高学历", "type": "text",
             "aliases": ["最高学历", "学历", "文化程度"], "max_length": 12},
            {"key": "school", "label": "毕业院校", "type": "org",
             "aliases": ["毕业院校", "学校", "毕业学校", "院校"]},
            {"key": "experience", "label": "工作经历", "type": "list",
             "sections": ["工作经历", "工作经验", "职业经历", "工作履历",
                          "从业经历"]},
            {"key": "projects", "label": "项目经历", "type": "list",
             "sections": ["项目经历", "项目经验", "项目履历"]},
            {"key": "edu_exp", "label": "教育经历", "type": "list",
             "sections": ["教育经历", "教育背景", "学习经历"]},
        ],
    },
    {
        "key": "builtin_contract",
        "name": "合同",
        "description": "抽取甲乙双方、金额（含大写）、期限与关键日期。",
        "fields": [
            {"key": "party_a", "label": "甲方", "type": "org",
             "required": True,
             "aliases": ["甲方", "发包方", "出租方", "买方", "委托方",
                         "采购方", "贷款方", "出借人"]},
            {"key": "party_b", "label": "乙方", "type": "org",
             "required": True,
             "aliases": ["乙方", "承包方", "承租方", "卖方", "受托方",
                         "供应方", "借款方", "借款人"]},
            {"key": "amount", "label": "合同金额", "type": "money",
             "required": True,
             "aliases": ["合同金额", "合同总金额", "合同总价", "总金额",
                         "合同价款", "合同款", "价款", "租金", "金额",
                         "标的额"]},
            {"key": "currency_text", "label": "大写金额", "type": "money",
             "aliases": ["大写", "人民币大写", "金额大写"]},
            {"key": "term", "label": "合同期限", "type": "text",
             "aliases": ["合同期限", "履行期限", "租赁期限", "服务期限",
                         "有效期", "合约期限"], "max_length": 60},
            {"key": "start_date", "label": "开始日期", "type": "date",
             "date_order": "first",
             "aliases": ["生效日期", "起始日期", "开始日期", "起租日",
                         "租期自", "有效期自", "自"]},
            {"key": "end_date", "label": "结束日期", "type": "date",
             "date_order": "last",
             "aliases": ["终止日期", "到期日期", "结束日期", "截止日期",
                         "租期至", "有效期至", "至"]},
            {"key": "sign_date", "label": "签订日期", "type": "date",
             "aliases": ["签订日期", "签署日期", "签约日期", "订立日期",
                         "签订时间", "签署时间", "日期"]},
        ],
    },
    {
        "key": "builtin_notice",
        "name": "通知",
        "description": "抽取通知标题、发文单位、时间地点等会务要素。",
        "fields": [
            {"key": "title", "label": "标题", "type": "text",
             "aliases": ["标题", "通知标题"], "max_length": 60,
             "fallback_positions": ["first_line"]},
            {"key": "organizer", "label": "发文单位", "type": "org",
             "aliases": ["发文单位", "发布单位", "主办单位", "承办单位",
                         "发文机关"]},
            {"key": "event_date", "label": "活动日期", "type": "date",
             "aliases": ["时间", "活动时间", "会议时间", "举行时间",
                         "举办时间", "日期"]},
            {"key": "location", "label": "地点", "type": "location",
             "aliases": ["地点", "会议地点", "活动地点", "举办地点",
                         "会场"]},
            {"key": "contact", "label": "联系人", "type": "person",
             "aliases": ["联系人", "对接人", "会务联系人"]},
            {"key": "contact_phone", "label": "联系电话", "type": "phone",
             "aliases": ["联系电话", "咨询电话", "报名电话", "电话"]},
        ],
    },
]


def validate_template_spec(spec: dict) -> dict:
    """校验模板定义，返回规整化后的 spec。"""
    tpl = Template(spec)
    return tpl.dump()
