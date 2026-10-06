"""Flask API 路由。

把所有 NLP 能力、存储与流水线编排暴露为 REST 接口，
前端 10 个页面通过 ``fetch`` 调用这些接口。
"""

from __future__ import annotations

import json
import re
import time
import uuid
from typing import Optional

from flask import Blueprint, current_app, jsonify, request, Response

from nlp import (get_constituency_parser, get_embeddings, get_keywords, get_ner,
                 get_parser, get_segmenter, get_sentiment, get_summarizer,
                 get_tagger, get_translator, get_extractor,
                 ENTITY_TYPE_NAMES, TAG_NAMES, DEP_REL_NAMES, PHRASE_NAMES,
                 POLARITY_NAMES, FIELD_TYPE_NAMES)
from nlp.extraction import (FieldTemplate, FieldSpec, BUILTIN_TEMPLATES,
                            get_builtin_templates)
from nlp.lexicon import STOPWORDS
from storage import StoreRegistry


api = Blueprint("api", __name__, url_prefix="/api")


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------

def _registry() -> StoreRegistry:
    return current_app.config["STORE_REGISTRY"]


def _engine():
    return current_app.config["PIPELINE_ENGINE"]


def _models_dir() -> str:
    import os
    path = os.path.join(current_app.config["DATA_ROOT"], "models")
    os.makedirs(path, exist_ok=True)
    return path


def _store_result(task: str, text: str, result: dict,
                  corpus_id: Optional[str] = None) -> str:
    record = {"text": text, "result": result, "created_at": time.time()}
    if corpus_id:
        record["corpus_id"] = corpus_id
    return _registry().task(task).insert(record)


def _payload() -> dict:
    data = request.get_json(silent=True) or {}
    return data


def _resolve_text(data: dict) -> tuple[str, Optional[str]]:
    """从请求中取文本：优先 text，其次 corpus_id。"""
    if data.get("text"):
        return data["text"], data.get("corpus_id")
    corpus_id = data.get("corpus_id")
    if corpus_id:
        record = _registry().task("corpus").get(corpus_id)
        if record:
            return record.get("text", ""), corpus_id
        return "", corpus_id
    return "", None


def _clean(text: str, remove_stopwords: bool = True) -> dict:
    text = re.sub(r"\s+", " ", text).strip()
    seg = get_segmenter()
    words = seg.cut(text)
    if remove_stopwords:
        kept = [w for w in words if w not in STOPWORDS]
    else:
        kept = words
    removed = len(words) - len(kept)
    return {
        "text": text,
        "cleaned": " ".join(kept),
        "tokens": kept,
        "original_tokens": words,
        "removed_stopwords": removed,
    }


# ---------------------------------------------------------------------------
# 状态
# ---------------------------------------------------------------------------

@api.get("/status")
def status():
    return jsonify({
        "ok": True,
        "version": "1.0.0",
        "tasks": _registry().tasks(),
        "time": time.time(),
    })


@api.get("/meta")
def meta():
    """给前端提供标签集合与可配置参数。"""
    return jsonify({
        "tag_names": TAG_NAMES,
        "dep_rel_names": DEP_REL_NAMES,
        "phrase_names": PHRASE_NAMES,
        "entity_type_names": ENTITY_TYPE_NAMES,
        "polarity_names": POLARITY_NAMES,
        "field_type_names": FIELD_TYPE_NAMES,
        "directions": [{"id": "zh2en", "name": "中文 → 英文"},
                       {"id": "en2zh", "name": "英文 → 中文"}],
    })


# ---------------------------------------------------------------------------
# 语料库管理
# ---------------------------------------------------------------------------

@api.get("/corpus")
def list_corpus():
    records = _registry().task("corpus").all()
    items = [{
        "id": r.get("id"),
        "name": r.get("name", "未命名"),
        "length": len(r.get("text", "")),
        "created_at": r.get("created_at"),
        "preview": r.get("text", "")[:80],
    } for r in records if not r.get("_deleted")]
    items.sort(key=lambda x: x.get("created_at", 0), reverse=True)
    return jsonify({"corpora": items})


@api.post("/corpus")
def create_corpus():
    data = _payload()
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "语料内容不能为空"}), 400
    record = {
        "name": data.get("name") or f"语料_{int(time.time())}",
        "text": text,
        "created_at": time.time(),
    }
    rid = _registry().task("corpus").insert(record)
    return jsonify({"id": rid, "ok": True})


@api.post("/corpus/upload")
def upload_corpus():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "未接收到文件"}), 400
    raw = file.read()
    text = None
    for enc in ("utf-8", "gbk", "gb18030", "utf-16"):
        try:
            text = raw.decode(enc)
            break
        except (UnicodeDecodeError, LookupError):
            continue
    if text is None:
        return jsonify({"error": "无法解码文件内容"}), 400
    name = data_name = file.filename or "上传文件"
    record = {"name": name, "text": text.strip(), "created_at": time.time()}
    rid = _registry().task("corpus").insert(record)
    return jsonify({"id": rid, "name": name, "length": len(text), "ok": True})


@api.get("/corpus/<cid>")
def get_corpus(cid: str):
    record = _registry().task("corpus").get(cid)
    if not record:
        return jsonify({"error": "语料不存在"}), 404
    return jsonify(record)


@api.delete("/corpus/<cid>")
def delete_corpus(cid: str):
    ok = _registry().task("corpus").delete(cid)
    return jsonify({"ok": ok})


@api.post("/corpus/<cid>/clean")
def clean_corpus(cid: str):
    record = _registry().task("corpus").get(cid)
    if not record:
        return jsonify({"error": "语料不存在"}), 404
    data = _payload()
    result = _clean(record.get("text", ""), data.get("remove_stopwords", True))
    _store_result("clean", record.get("text", ""), result, corpus_id=cid)
    return jsonify(result)


# ---------------------------------------------------------------------------
# 分词与词性标注
# ---------------------------------------------------------------------------

@api.post("/segment")
def segment():
    data = _payload()
    text, cid = _resolve_text(data)
    if not text:
        return jsonify({"error": "缺少文本"}), 400
    seg = get_segmenter()
    words = seg.cut(text)
    result = {"words": words, "count": len(words)}
    rid = _store_result("segment", text, result, corpus_id=cid)
    result["id"] = rid
    return jsonify(result)


@api.post("/pos")
def pos_tag():
    data = _payload()
    text, cid = _resolve_text(data)
    if not text:
        return jsonify({"error": "缺少文本"}), 400
    tagger = get_tagger()
    tokens = [[w, t] for w, t in tagger.tag(text)]
    result = {"tokens": tokens, "tag_names": TAG_NAMES}
    rid = _store_result("pos", text, result, corpus_id=cid)
    result["id"] = rid
    return jsonify(result)


# ---------------------------------------------------------------------------
# 句法分析
# ---------------------------------------------------------------------------

@api.post("/parse")
def parse():
    data = _payload()
    text, cid = _resolve_text(data)
    if not text:
        return jsonify({"error": "缺少文本"}), 400
    dep = get_parser().parse(text)
    const = get_constituency_parser().parse(text)
    result = {
        "dependency": dep,
        "constituency": const,
        "dep_rel_names": DEP_REL_NAMES,
        "phrase_names": PHRASE_NAMES,
    }
    rid = _store_result("parse", text, result, corpus_id=cid)
    result["id"] = rid
    return jsonify(result)


# ---------------------------------------------------------------------------
# 命名实体识别与标注
# ---------------------------------------------------------------------------

@api.post("/ner")
def ner():
    data = _payload()
    text, cid = _resolve_text(data)
    if not text:
        return jsonify({"error": "缺少文本"}), 400
    entities = get_ner().recognize(text)
    result = {"entities": entities, "entity_type_names": ENTITY_TYPE_NAMES}
    rid = _store_result("ner", text, result, corpus_id=cid)
    result["id"] = rid
    return jsonify(result)


@api.post("/ner/annotate")
def ner_annotate():
    data = _payload()
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "缺少文本"}), 400
    record = {
        "text": text,
        "entities": data.get("entities", []),
        "note": data.get("note", ""),
        "created_at": time.time(),
    }
    rid = _registry().task("annotation").insert(record)
    return jsonify({"id": rid, "ok": True})


@api.get("/ner/annotations")
def ner_annotations():
    records = _registry().task("annotation").all()
    return jsonify({"annotations": records})


# ---------------------------------------------------------------------------
# 情感分析
# ---------------------------------------------------------------------------

@api.post("/sentiment")
def sentiment():
    data = _payload()
    text, cid = _resolve_text(data)
    if not text:
        return jsonify({"error": "缺少文本"}), 400
    result = get_sentiment().analyze(text)
    result["polarity_name"] = POLARITY_NAMES.get(result["polarity"], "")
    rid = _store_result("sentiment", text, result, corpus_id=cid)
    result["id"] = rid
    return jsonify(result)


# ---------------------------------------------------------------------------
# 文本摘要
# ---------------------------------------------------------------------------

@api.post("/summary")
def summary():
    data = _payload()
    text, cid = _resolve_text(data)
    if not text:
        return jsonify({"error": "缺少文本"}), 400
    result = get_summarizer().summarize(
        text, ratio=data.get("ratio", 0.3),
        max_sentences=data.get("max_sentences"))
    rid = _store_result("summary", text, result, corpus_id=cid)
    result["id"] = rid
    return jsonify(result)


# ---------------------------------------------------------------------------
# 机器翻译（模拟）
# ---------------------------------------------------------------------------

@api.post("/translate")
def translate():
    data = _payload()
    text, cid = _resolve_text(data)
    if not text:
        return jsonify({"error": "缺少文本"}), 400
    result = get_translator().translate(text, direction=data.get("direction", "zh2en"))
    rid = _store_result("translate", text, result, corpus_id=cid)
    result["id"] = rid
    return jsonify(result)


# ---------------------------------------------------------------------------
# 关键词提取
# ---------------------------------------------------------------------------

@api.post("/keywords")
def keywords():
    data = _payload()
    text, cid = _resolve_text(data)
    if not text:
        return jsonify({"error": "缺少文本"}), 400
    result = get_keywords().extract(text, top_k=data.get("top_k", 10),
                                    method=data.get("method", "hybrid"))
    rid = _store_result("keywords", text, result, corpus_id=cid)
    result["id"] = rid
    return jsonify(result)


# ---------------------------------------------------------------------------
# 词向量
# ---------------------------------------------------------------------------

def _embedding_path() -> str:
    import os
    return os.path.join(_models_dir(), "embeddings.json")


@api.post("/embeddings/train")
def train_embeddings():
    data = _payload()
    corpus_ids = data.get("corpus_ids")
    store = _registry().task("corpus")
    if corpus_ids:
        texts = [store.get(c)["text"] for c in corpus_ids if store.get(c)]
    else:
        texts = [r["text"] for r in store.all() if not r.get("_deleted")]
    if not texts:
        return jsonify({"error": "没有可用语料，请先上传语料"}), 400

    emb = get_embeddings()
    emb.train(texts, vocab_size=data.get("vocab_size", 200),
              dim=data.get("dim", 20), window=data.get("window", 5),
              min_count=data.get("min_count", 1))

    payload = {
        "vocab": emb.vocab,
        "vectors": emb.vectors,
        "dim": emb.dim,
        "trained_at": time.time(),
        "corpus_count": len(texts),
    }
    with open(_embedding_path(), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False)
    return jsonify(emb.stats())


@api.get("/embeddings/vectors")
def embeddings_vectors():
    emb = get_embeddings()
    if not emb.vectors:
        _load_embeddings()
        emb = get_embeddings()
    if not emb.vectors:
        return jsonify({"error": "尚未训练词向量"}), 404
    n_clusters = int(request.args.get("clusters", 5))
    proj = emb.project_2d()
    clusters = emb.cluster(n_clusters)
    return jsonify({
        "points": [{"word": w, "x": round(p[0], 4), "y": round(p[1], 4),
                    "cluster": clusters.get(w, 0)} for w, p in proj.items()],
        "stats": emb.stats(),
    })


@api.get("/embeddings/neighbors")
def embeddings_neighbors():
    word = request.args.get("word", "")
    k = int(request.args.get("k", 10))
    emb = get_embeddings()
    if not emb.vectors:
        _load_embeddings()
        emb = get_embeddings()
    return jsonify({"word": word, "neighbors": emb.nearest(word, k)})


def _load_embeddings():
    import os
    path = _embedding_path()
    if not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        emb = get_embeddings()
        emb.vocab = data.get("vocab", [])
        emb.vectors = data.get("vectors", {})
        emb.dim = data.get("dim", 0)
    except (json.JSONDecodeError, OSError):
        pass


# ---------------------------------------------------------------------------
# 流水线配置与执行
# ---------------------------------------------------------------------------

@api.get("/pipeline/stages")
def pipeline_stages():
    return jsonify({"stages": _engine().list_stages()})


@api.post("/pipeline")
def save_pipeline():
    data = _payload()
    config = data.get("config") or data
    if not config.get("stages"):
        return jsonify({"error": "流水线至少需要一个阶段"}), 400
    name = config.get("name") or f"流水线_{int(time.time())}"
    record = {"name": name, "config": config, "created_at": time.time()}
    rid = _registry().task("pipeline_config").insert(record)
    return jsonify({"id": rid, "name": name, "ok": True})


@api.get("/pipeline")
def list_pipelines():
    records = _registry().task("pipeline_config").all()
    items = [{"id": r["id"], "name": r.get("name"), "config": r.get("config"),
              "created_at": r.get("created_at")}
             for r in records if not r.get("_deleted")]
    items.sort(key=lambda x: x.get("created_at", 0), reverse=True)
    return jsonify({"pipelines": items})


@api.get("/pipeline/<pid>")
def get_pipeline(pid: str):
    record = _registry().task("pipeline_config").get(pid)
    if not record:
        return jsonify({"error": "流水线不存在"}), 404
    return jsonify(record)


@api.post("/pipeline/preview")
def pipeline_preview():
    """对单条文本跑流水线（不持久化），供配置页预览。"""
    data = _payload()
    text = (data.get("text") or "").strip()
    config = data.get("config")
    if not text or not config:
        return jsonify({"error": "缺少文本或配置"}), 400
    try:
        result = _engine().build(config).run({"text": text})
        return jsonify({"ok": True, "output": result})
    except Exception as exc:  # noqa: BLE001
        return jsonify({"ok": False, "error": str(exc)}), 400


@api.post("/pipeline/<pid>/run")
def run_pipeline(pid: str):
    record = _registry().task("pipeline_config").get(pid)
    if not record:
        return jsonify({"error": "流水线不存在"}), 404
    config = record.get("config")
    data = _payload()

    run_id = uuid.uuid4().hex[:12]
    started = time.time()

    if data.get("batch"):
        # 批量：对语料库中的多篇文档执行
        corpus_ids = data.get("corpus_ids") or []
        store = _registry().task("corpus")
        docs = []
        if corpus_ids:
            docs = [store.get(c)["text"] for c in corpus_ids if store.get(c)]
        else:
            docs = [r["text"] for r in store.all() if not r.get("_deleted")]
        if not docs:
            return jsonify({"error": "没有可处理的文档"}), 400

        progress_state = {"done": 0, "total": len(docs)}

        def _progress(done, total):
            progress_state["done"] = done
            progress_state["total"] = total

        results = _engine().run_batch(
            config, docs, shared=data.get("shared"),
            max_workers=data.get("max_workers", 4),
            chunk_size=data.get("chunk_size", 16),
            progress=_progress)
        succeeded = sum(1 for r in results if r and r["ok"])
        failed = len(results) - succeeded
        run_record = {
            "run_id": run_id, "pipeline_id": pid, "batch": True,
            "doc_count": len(docs), "succeeded": succeeded, "failed": failed,
            "started": started, "finished": time.time(),
            "results": results,
        }
        rid = _registry().task("pipeline_run").insert(run_record)
        return jsonify({"run_id": run_id, "id": rid, "succeeded": succeeded,
                        "failed": failed, "doc_count": len(docs)})
    else:
        text = (data.get("text") or "").strip()
        if not text:
            return jsonify({"error": "缺少文本"}), 400
        try:
            output = _engine().build(config).run({"text": text})
            run_record = {
                "run_id": run_id, "pipeline_id": pid, "batch": False,
                "text": text, "output": output,
                "started": started, "finished": time.time(),
            }
            rid = _registry().task("pipeline_run").insert(run_record)
            return jsonify({"run_id": run_id, "id": rid, "ok": True,
                            "output": output})
        except Exception as exc:  # noqa: BLE001
            return jsonify({"ok": False, "error": str(exc)}), 400


@api.get("/pipeline/run/<run_id>")
def get_pipeline_run(run_id: str):
    records = _registry().task("pipeline_run").query(
        where=[("run_id", "eq", run_id)])
    if not records:
        return jsonify({"error": "执行记录不存在"}), 404
    return jsonify(records[0])


# ---------------------------------------------------------------------------
# 模板化字段抽取
# ---------------------------------------------------------------------------

TEMPLATE_TASK = "extract_template"
RESULT_TASK = "extract_result"
_BUILTIN_KEYS = {t["key"] for t in BUILTIN_TEMPLATES}


def _template_store():
    return _registry().task(TEMPLATE_TASK)


def _result_store():
    return _registry().task(RESULT_TASK)


def _parse_template(data: dict) -> FieldTemplate:
    """从请求构造模板，非法字段直接抛 ValueError（由路由转 400）。"""
    fields = [FieldSpec.from_dict(f) for f in data.get("fields", [])]
    return FieldTemplate(
        name=(data.get("name") or "").strip(),
        fields=fields,
        key=data.get("key"),
        version=int(data.get("version", 1)),
        description=data.get("description", ""),
    )


def _public_template(record: dict) -> dict:
    return {
        "id": record.get("id"),
        "key": record.get("template", {}).get("key"),
        "name": record.get("name"),
        "description": record.get("description", ""),
        "version": record.get("version", 1),
        "built_in": record.get("built_in", False),
        "created_at": record.get("created_at"),
        "updated_at": record.get("updated_at"),
        "template": record.get("template"),
    }


def seed_builtin_templates() -> int:
    """首次使用时植入内置模板（简历/合同/通知），已存在则跳过。

    内置模板以固定 key 存在 extract_template 存储里；用户后续调整会
    生成新版本记录，但已有抽取结果内嵌模板快照，不受影响。
    """
    store = _template_store()
    existing = {r.get("template", {}).get("key")
                for r in store.all() if not r.get("_deleted")}
    count = 0
    now = time.time()
    for tpl in get_builtin_templates():
        if tpl.key in existing:
            continue
        data = tpl.to_dict()
        store.insert({
            "name": tpl.name,
            "description": tpl.description,
            "version": 1,
            "built_in": True,
            "template": data,
            "updated_at": now,
        })
        count += 1
    return count


@api.get("/extract/templates")
def list_extract_templates():
    records = [r for r in _template_store().all() if not r.get("_deleted")]
    records.sort(key=lambda r: (not r.get("built_in", False),
                                r.get("created_at", 0)))
    return jsonify({"templates": [_public_template(r) for r in records]})


@api.post("/extract/templates")
def create_extract_template():
    data = _payload()
    if not (data.get("name") or "").strip():
        return jsonify({"error": "模板名称不能为空"}), 400
    try:
        tpl = _parse_template(data)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    if tpl.key:
        dup = any(r.get("template", {}).get("key") == tpl.key
                  for r in _template_store().all() if not r.get("_deleted"))
        if dup:
            return jsonify({"error": f"模板标识 {tpl.key} 已存在"}), 400
    record = {
        "name": tpl.name,
        "description": tpl.description,
        "version": 1,
        "built_in": False,
        "template": tpl.to_dict(),
        "updated_at": time.time(),
    }
    rid = _template_store().insert(record)
    saved = _template_store().get(rid)
    return jsonify({"ok": True, "id": rid, "template": _public_template(saved)})


def _find_template(ident: str) -> Optional[dict]:
    """按记录 id 或模板 key 找当前模板（墓碑除外）。"""
    store = _template_store()
    record = store.get(ident)
    if record and not record.get("_deleted"):
        return record
    for r in store.all():
        if r.get("_deleted"):
            continue
        if r.get("template", {}).get("key") == ident:
            return r
    return None



@api.get("/extract/templates/<ident>")
def get_extract_template(ident: str):
    record = _find_template(ident)
    if not record:
        return jsonify({"error": "模板不存在"}), 404
    return jsonify(_public_template(record))


@api.put("/extract/templates/<ident>")
def update_extract_template(ident: str):
    """调整模板：字段变化时版本号 +1，旧抽取结果仍引用旧快照，不被冲乱。"""
    store = _template_store()
    record = _find_template(ident)
    if not record:
        return jsonify({"error": "模板不存在"}), 404
    data = _payload()
    merged = {"name": data.get("name", record["name"]),
              "description": data.get("description",
                                      record.get("description", "")),
              "fields": data.get("fields",
                                 record["template"].get("fields", []))}
    old_tpl = record["template"]
    merged["key"] = data.get("key", old_tpl.get("key"))
    try:
        tpl = _parse_template(merged)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    version = record.get("version", 1)
    if data.get("fields") and data["fields"] != old_tpl.get("fields"):
        version += 1
    tpl.version = version
    updated = store.update(record["id"], {
        "name": tpl.name,
        "description": tpl.description,
        "version": version,
        "template": tpl.to_dict(),
        "updated_at": time.time(),
    })
    return jsonify({"ok": True, "template": _public_template(updated),
                    "version": version})


@api.delete("/extract/templates/<ident>")
def delete_extract_template(ident: str):
    record = _find_template(ident)
    if not record:
        return jsonify({"error": "模板不存在"}), 404
    ok = _template_store().delete(record["id"])
    return jsonify({"ok": ok})


@api.get("/extract/meta")
def extract_meta():
    return jsonify({
        "field_types": [{"id": k, "name": v}
                        for k, v in FIELD_TYPE_NAMES.items()],
    })


def _run_extraction(text: str, ident: str) -> tuple[Optional[dict], Optional[str]]:
    record = _find_template(ident)
    if not record:
        return None, "模板不存在"
    tpl = FieldTemplate.from_dict(record["template"])
    result = get_extractor().extract(text, tpl)
    return result, None


@api.post("/extract/run")
def extract_run():
    """对单条文本（或语料库文档）按模板抽取；默认持久化结果。"""
    data = _payload()
    text, cid = _resolve_text(data)
    ident = data.get("template_id") or data.get("template")
    if not text:
        return jsonify({"error": "缺少文本"}), 400
    if not ident:
        return jsonify({"error": "请选择抽取模板"}), 400
    result, err = _run_extraction(text, ident)
    if err:
        return jsonify({"error": err}), 404
    if data.get("save", True):
        record = _find_template(ident)
        rid = _result_store().insert({
            "text": text,
            "corpus_id": cid,
            "template_id": record["id"],
            "template_key": record["template"].get("key"),
            "template_version": result["template_version"],
            "result": result,
            "complete": result["complete"],
            "missing": result["missing"],
            "ambiguous": result["ambiguous"],
            "created_at": time.time(),
        })
        result["id"] = rid
    return jsonify({"ok": True, "result": result})


@api.post("/extract/batch")
def extract_batch():
    """批量对语料库文档抽取，逐篇容错；结果逐篇分片存储。"""
    data = _payload()
    ident = data.get("template_id") or data.get("template")
    record = _find_template(ident) if ident else None
    if not record:
        return jsonify({"error": "模板不存在"}), 404
    corpus_store = _registry().task("corpus")
    corpus_ids = data.get("corpus_ids") or []
    if corpus_ids:
        docs = [(c, corpus_store.get(c)) for c in corpus_ids]
        docs = [(c, d) for c, d in docs if d and not d.get("_deleted")]
    else:
        docs = [(r["id"], r) for r in corpus_store.all()
                if not r.get("_deleted")]
    if not docs:
        return jsonify({"error": "没有可处理的语料文档"}), 400

    tpl = FieldTemplate.from_dict(record["template"])
    extractor = get_extractor()
    items, succeeded, failed = [], 0, 0
    for cid, doc in docs:
        try:
            result = extractor.extract(doc["text"], tpl)
            rid = _result_store().insert({
                "text": doc["text"],
                "corpus_id": cid,
                "template_id": record["id"],
                "template_key": record["template"].get("key"),
                "template_version": result["template_version"],
                "result": result,
                "complete": result["complete"],
                "missing": result["missing"],
                "ambiguous": result["ambiguous"],
                "created_at": time.time(),
            })
            items.append({"corpus_id": cid, "id": rid, "ok": True,
                          "complete": result["complete"],
                          "missing": result["missing"],
                          "ambiguous": result["ambiguous"]})
            succeeded += 1
        except Exception as exc:  # noqa: BLE001
            items.append({"corpus_id": cid, "ok": False, "error": str(exc)})
            failed += 1
    return jsonify({"ok": True, "succeeded": succeeded, "failed": failed,
                    "results": items})


@api.get("/extract/results")
def list_extract_results():
    where = []
    for key in ("template_id", "template_key", "corpus_id", "complete"):
        val = request.args.get(key)
        if val is not None and val != "":
            where.append((key, "eq", val if key != "complete"
                          else (val in ("1", "true", "True"))))
    records = _result_store().query(
        where=where or None,
        order_by=request.args.get("order_by", "created_at"),
        order=request.args.get("order", "desc"),
        limit=request.args.get("limit", type=int),
        offset=request.args.get("offset", 0, type=int))
    return jsonify({"count": len(records), "records": records})


@api.get("/extract/results/<rid>")
def get_extract_result(rid: str):
    record = _result_store().get(rid)
    if not record or record.get("_deleted"):
        return jsonify({"error": "抽取结果不存在"}), 404
    return jsonify(record)


@api.delete("/extract/results/<rid>")
def delete_extract_result(rid: str):
    ok = _result_store().delete(rid)
    return jsonify({"ok": ok})


def _flatten_field_value(value) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return " | ".join(_flatten_field_value(v) for v in value)
    if isinstance(value, dict):
        return value.get("text") or json.dumps(value, ensure_ascii=False)
    return str(value)


@api.get("/extract/results/<rid>/table")
def extract_result_table(rid: str):
    """把单次（或同一模板的一批）结果转成表格列：字段 -> 取值。"""
    store = _result_store()
    if rid == "all":
        template_key = request.args.get("template_key")
        records = store.all()
        if template_key:
            records = [r for r in records
                       if r.get("template_key") == template_key]
    else:
        record = store.get(rid)
        if not record or record.get("_deleted"):
            return jsonify({"error": "抽取结果不存在"}), 404
        records = [record]
    records = [r for r in records if not r.get("_deleted")]

    columns: list[str] = []
    rows = []
    for r in records:
        result = r.get("result", {})
        snapshot = result.get("template_snapshot", {})
        fields = result.get("fields", {})
        for spec in snapshot.get("fields", []):
            if spec["key"] not in columns:
                columns.append(spec["key"])
        row = {
            "_id": r.get("id"),
            "_corpus_id": r.get("corpus_id", ""),
            "_status": {k: v.get("status") for k, v in fields.items()},
        }
        for key, f in fields.items():
            row[key] = _flatten_field_value(f.get("value"))
        rows.append(row)
    return jsonify({"columns": columns,
                     "field_meta": {s["key"]: s
                                    for r in records
                                    for s in r.get("result", {})
                                    .get("template_snapshot", {})
                                    .get("fields", [])},
                     "rows": rows})


@api.get("/extract/results/<rid>/export")
def extract_result_export(rid: str):
    """导出 CSV（UTF-8 BOM，Excel 可直接打开）。"""
    import csv
    import io

    store = _result_store()
    if rid == "all":
        template_key = request.args.get("template_key")
        records = [r for r in store.all() if not r.get("_deleted")]
        if template_key:
            records = [r for r in records
                       if r.get("template_key") == template_key]
    else:
        record = store.get(rid)
        if not record or record.get("_deleted"):
            return jsonify({"error": "抽取结果不存在"}), 404
        records = [record]

    columns, seen = [], set()
    for r in records:
        for s in r.get("result", {}).get("template_snapshot", {}).get("fields", []):
            if s["key"] not in seen:
                seen.add(s["key"])
                columns.append((s["key"], s["name"]))

    buf = io.StringIO()
    buf.write("﻿")
    writer = csv.writer(buf)
    writer.writerow(["记录ID", "语料ID"] + [name for _, name in columns])
    for r in records:
        fields = r.get("result", {}).get("fields", {})
        row = [r.get("id", ""), r.get("corpus_id", "")]
        for key, _ in columns:
            f = fields.get(key, {})
            if f.get("status") == "missing":
                row.append("【缺失】")
            elif f.get("status") == "ambiguous":
                row.append("【歧义:" + " / ".join(
                    c.get("text", "") for c in f.get("candidates", [])) + "】")
            else:
                row.append(_flatten_field_value(f.get("value")))
        writer.writerow(row)
    filename = f"extract_{rid}.csv"
    return Response(
        buf.getvalue(),
        mimetype="text/csv; charset=utf-8",
        headers={"Content-Disposition": f"attachment; filename={filename}"})


# ---------------------------------------------------------------------------
# 结果查询（分片合并与查询）
# ---------------------------------------------------------------------------

@api.get("/results")
def list_result_tasks():
    registry = _registry()
    tasks = []
    for name in registry.tasks():
        if name in ("corpus", "pipeline_config", "annotation"):
            continue
        stats = registry.task(name).stats()
        tasks.append(stats)
    return jsonify({"tasks": tasks})


@api.get("/results/<task>")
def query_results(task: str):
    registry = _registry()
    if task not in registry.tasks():
        return jsonify({"error": "任务不存在"}), 404
    store = registry.task(task)
    where = []
    for key in ("type", "corpus_id"):
        val = request.args.get(key)
        if val:
            where.append((key, "eq", val))
    order_by = request.args.get("order_by")
    order = request.args.get("order", "desc")
    limit = request.args.get("limit", type=int)
    offset = request.args.get("offset", 0, type=int)
    records = store.query(where=where or None, order_by=order_by,
                          order=order, limit=limit, offset=offset)
    return jsonify({
        "task": task,
        "count": len(records),
        "stats": store.stats(),
        "records": records,
    })


@api.post("/results/<task>/compact")
def compact_results(task: str):
    registry = _registry()
    if task not in registry.tasks():
        return jsonify({"error": "任务不存在"}), 404
    return jsonify(registry.task(task).compact())


@api.get("/results/<task>/merge")
def merge_results(task: str):
    registry = _registry()
    if task not in registry.tasks():
        return jsonify({"error": "任务不存在"}), 404
    return jsonify(registry.task(task).merge())


@api.post("/results/compact_all")
def compact_all():
    return jsonify({"compacted": _registry().compact_all()})
