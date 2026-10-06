"""模板化字段抽取测试。

运行：``python -m unittest tests.test_extraction -v``

覆盖：
- 锚点命中、类型兜底、全文推断的可信度区分；
- 缺失显式标记（不填错值）、多写法歧义（保留全部候选）；
- 同一段文本多模板互不干扰；
- 电话/中文大写金额/日期期限归一与多种写法合并；
- 区块字段（工作经历）逐行偏移可回查；
- 模板版本快照：改模板不冲乱旧结果；
- API：模板 CRUD、抽取、结果表格与 CSV 导出。
"""

from __future__ import annotations

import csv
import io
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nlp import get_builtin_templates
from nlp.extraction import (FieldSpec, FieldTemplate, TemplateExtractor,
                            _canonical_phone, _normalize_money)


RESUME = """个人简历

姓名：李明
电话：138-1234-5678，备用手机 13987654321
邮箱：liming@example.com
最高学历：硕士研究生

工作经历：
2020年-2023年 在腾讯科技有限公司任后端工程师
2023年至今 在字节跳动负责推荐系统
"""

CONTRACT = """技术服务合同

甲方（发包方）：华云科技有限公司
乙方：北京数智信息技术有限公司

合同总金额为人民币壹拾贰万元整（小写：120,000元）。
合同期限：2024年3月1日至2025年2月28日
签订日期：2024年2月20日
"""


class TestExtractionBasic(unittest.TestCase):
    def setUp(self):
        self.ext = TemplateExtractor()
        self.resume_tpl, self.contract_tpl, self.notice_tpl = \
            get_builtin_templates()

    def test_resume_fields(self):
        r = self.ext.extract(RESUME, self.resume_tpl)
        fields = r["fields"]
        self.assertEqual(fields["name"]["value"], "李明")
        self.assertEqual(fields["name"]["status"], "found")
        self.assertEqual(fields["email"]["value"], "liming@example.com")
        self.assertEqual(fields["education"]["value"], "硕士研究生")
        # 两种电话写法都要收进来（锚点 + 换标签的全文补扫）
        phones = fields["phone"]["value"]
        self.assertEqual(set(phones), {"13812345678", "13987654321"})

    def test_evidence_offsets_map_to_source(self):
        r = self.ext.extract(CONTRACT, self.contract_tpl)
        for key, f in r["fields"].items():
            self.assertTrue(f["candidates"], key)
            for cand in f["candidates"]:
                for e in cand["evidence"]:
                    self.assertEqual(
                        CONTRACT[e["start"]:e["end"]], e["text"],
                        f"{key} 偏移与原文不一致: {e}")

    def test_experience_block_split(self):
        r = self.ext.extract(RESUME, self.resume_tpl)
        exp = r["fields"]["experience"]
        self.assertEqual(exp["status"], "found")
        self.assertIsInstance(exp["value"], str)
        self.assertIn("腾讯科技有限公司", exp["value"])
        # 区块按行拆成候选，每条都能回到原文
        texts = [c["text"] for c in exp["candidates"]]
        self.assertTrue(any("腾讯" in t for t in texts))
        self.assertTrue(any("字节跳动" in t for t in texts))

    def test_contract_fields_and_normalization(self):
        r = self.ext.extract(CONTRACT, self.contract_tpl)
        fields = r["fields"]
        self.assertEqual(fields["party_a"]["value"], "华云科技有限公司")
        self.assertEqual(fields["party_b"]["value"], "北京数智信息技术有限公司")
        # 中文大写 壹拾贰万元 与阿拉伯 120,000元 归并为同一金额
        self.assertEqual(fields["amount"]["status"], "found")
        self.assertAlmostEqual(fields["amount"]["value"]["amount"], 12.0)
        self.assertEqual(fields["amount"]["value"]["unit"], "万元")
        period = fields["period"]["value"]
        self.assertEqual(period["start"], "2024-03-01")
        self.assertEqual(period["end"], "2025-02-28")
        self.assertEqual(fields["sign_date"]["value"], "2024-02-20")

    def test_missing_is_marked_not_filled(self):
        # 合同文本里没有金额字段；缺字段必须是 missing + value=None
        text = "甲方：某公司\n乙方：另一家公司\n"
        r = self.ext.extract(text, self.contract_tpl)
        self.assertEqual(r["fields"]["amount"]["status"], "missing")
        self.assertIsNone(r["fields"]["amount"]["value"])
        self.assertIn("amount", r["missing"])
        # 金额为选填字段：缺失被标出，但必填的甲乙方齐全 => 记录仍可入库
        self.assertNotIn("party_a", r["required_missing"])
        self.assertNotIn("party_b", r["required_missing"])

        # 若把金额设为必填，缺失就应使记录不完整
        tpl = FieldTemplate.from_dict({
            **self.contract_tpl.to_dict(),
            "fields": [
                {**f, "required": True} if f["key"] == "amount" else f
                for f in self.contract_tpl.to_dict()["fields"]
            ],
        })
        r2 = self.ext.extract(text, tpl)
        self.assertFalse(r2["complete"])
        self.assertIn("amount", r2["required_missing"])

    def test_ambiguous_values_kept_separate(self):
        # 单值字段出现两种金额：标记 ambiguous，不替用户选
        text = ("甲方：甲公司\n乙方：乙公司\n"
                "合同金额：10万元。\n备注：实际价款为15万元。\n")
        r = self.ext.extract(text, self.contract_tpl)
        amount = r["fields"]["amount"]
        self.assertEqual(amount["status"], "ambiguous")
        self.assertIsNone(amount["value"])
        self.assertEqual(len(amount["candidates"]), 2)
        self.assertIn("amount", r["ambiguous"])

    def test_templates_isolated_on_same_text(self):
        # 同一段文本用简历模板和合同模板抽，结果互不干扰
        r_resume = self.ext.extract(CONTRACT, self.resume_tpl)
        r_contract = self.ext.extract(CONTRACT, self.contract_tpl)
        # 简历模板里没有甲乙方字段
        self.assertNotIn("party_a", r_resume["fields"])
        self.assertIn("name", r_resume["fields"])
        # 合同模板里没有工作经历
        self.assertNotIn("experience", r_contract["fields"])
        self.assertIn("party_a", r_contract["fields"])

    def test_custom_regex_template(self):
        tpl = FieldTemplate("文号模板", [
            FieldSpec("doc_no", "文号", "text",
                      labels=["文号", "发文字号"],
                      pattern=r"[A-Z一-龥]{1,10}〔20\d{2}〕\d+号"),
        ])
        text = "发文单位：某某局\n文号：云建〔2024〕37号\n请遵照执行。"
        r = self.ext.extract(text, tpl)
        self.assertEqual(r["fields"]["doc_no"]["value"], "云建〔2024〕37号")
        self.assertEqual(r["fields"]["doc_no"]["status"], "found")

    def test_normalizers(self):
        self.assertEqual(_canonical_phone("138-1234-5678"), "13812345678")
        self.assertEqual(_canonical_phone("13987654321"), "13987654321")
        self.assertEqual(_canonical_phone("010-88889999"), "01088889999")
        amount, unit = _normalize_money("人民币壹拾贰万元整")
        self.assertEqual((amount, unit), (12.0, "万元"))
        amount, unit = _normalize_money("120,000元")
        self.assertEqual((amount, unit), (120000.0, "元"))
        amount, unit = _normalize_money("3.5亿元")
        self.assertEqual((amount, unit), (3.5, "亿元"))


class TestTemplateSnapshot(unittest.TestCase):
    """改模板不冲乱旧结果：结果内嵌模板快照。"""

    def test_snapshot_pinned(self):
        ext = TemplateExtractor()
        tpl_v1 = FieldTemplate("自定义", [
            FieldSpec("name", "姓名", "person", required=True),
            FieldSpec("phone", "电话", "phone"),
        ], key="custom", version=1)
        r1 = ext.extract(RESUME, tpl_v1)
        self.assertEqual(len(r1["template_snapshot"]["fields"]), 2)

        # 模板迭代：删字段、加字段、版本号 +1
        tpl_v2 = FieldTemplate("自定义", [
            FieldSpec("name", "姓名", "person", required=True),
            FieldSpec("email", "邮箱", "email"),
            FieldSpec("city", "城市", "text"),
        ], key="custom", version=2)
        r2 = ext.extract(RESUME, tpl_v2)

        # 旧结果保持旧形状
        self.assertEqual(r1["template_version"], 1)
        self.assertIn("phone", r1["fields"])
        self.assertNotIn("email", r1["fields"])
        # 新结果按新模板
        self.assertEqual(r2["template_version"], 2)
        self.assertIn("email", r2["fields"])
        self.assertEqual(r2["fields"]["email"]["value"], "liming@example.com")


class TestExtractionAPI(unittest.TestCase):
    def setUp(self):
        try:
            import flask  # noqa: F401
        except ImportError:
            self.skipTest("Flask 未安装")
        from app import create_app
        self.tmp = tempfile.mkdtemp()
        self.app = create_app(data_root=self.tmp)
        self.client = self.app.test_client()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_full_flow(self):
        # 内置模板已植入
        resp = self.client.get("/api/extract/templates")
        keys = {t["key"] for t in resp.get_json()["templates"]}
        self.assertIn("resume", keys)
        self.assertIn("contract", keys)

        # 用合同模板抽取
        resp = self.client.post("/api/extract/run", json={
            "text": CONTRACT, "template_id": "contract",
        })
        data = resp.get_json()
        self.assertTrue(data["ok"])
        result = data["result"]
        self.assertEqual(result["fields"]["party_a"]["value"],
                         "华云科技有限公司")
        rid = result["id"]

        # 表格视图
        resp = self.client.get(
            f"/api/extract/results/all/table?template_key=contract")
        table = resp.get_json()
        self.assertIn("party_a", table["columns"])
        self.assertTrue(table["rows"])
        self.assertEqual(table["rows"][0]["party_a"], "华云科技有限公司")

        # CSV 导出，缺失/歧义有显式占位
        resp = self.client.get("/api/extract/results/all/export"
                               "?template_key=contract")
        self.assertEqual(resp.status_code, 200)
        content = resp.data.decode("utf-8-sig")
        reader = list(csv.reader(io.StringIO(content)))
        self.assertIn("甲方", reader[0])
        self.assertTrue(any("华云科技有限公司" in "".join(row)
                            for row in reader[1:]))

        # 单条结果回查
        resp = self.client.get(f"/api/extract/results/{rid}")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.get_json()["result"]["fields"]["sign_date"]
                         ["value"], "2024-02-20")

    def test_template_version_bump_keeps_old_results(self):
        # 新建模板 -> 抽取 -> 调整字段（版本+1）-> 旧结果仍可按旧快照读取
        resp = self.client.post("/api/extract/templates", json={
            "name": "测试模板", "key": "tpl_test_ver",
            "fields": [{"key": "name", "name": "姓名", "type": "person",
                        "required": True}],
        })
        tid = resp.get_json()["id"]
        r1 = self.client.post("/api/extract/run", json={
            "text": "姓名：王芳\n", "template_id": tid,
        }).get_json()["result"]
        old_rid = r1["id"]
        self.assertEqual(r1["fields"]["name"]["value"], "王芳")

        resp = self.client.put(f"/api/extract/templates/{tid}", json={
            "fields": [{"key": "name", "name": "姓名", "type": "person"},
                       {"key": "phone", "name": "电话", "type": "phone"}],
        })
        self.assertEqual(resp.get_json()["version"], 2)

        # 旧结果不动
        old = self.client.get(f"/api/extract/results/{old_rid}").get_json()
        self.assertEqual(old["template_version"], 1)
        self.assertNotIn("phone", old["result"]["fields"])

    def test_template_isolation_across_api(self):
        # 同文本分别用简历/合同模板抽，存储里两条结果各自独立
        for key in ("resume", "contract"):
            resp = self.client.post("/api/extract/run", json={
                "text": CONTRACT, "template_id": key,
            })
            self.assertTrue(resp.get_json()["ok"])
        r_resume = self.client.get(
            "/api/extract/results?template_key=resume").get_json()
        r_contract = self.client.get(
            "/api/extract/results?template_key=contract").get_json()
        self.assertTrue(r_resume["records"])
        self.assertTrue(r_contract["records"])
        self.assertNotIn("party_a",
                         r_resume["records"][0]["result"]["fields"])
        self.assertIn("party_a",
                      r_contract["records"][0]["result"]["fields"])


class TestStoreUpdate(unittest.TestCase):
    """模板版本迭代依赖 ShardedStore.update 保持 id 与分片不变。"""

    def setUp(self):
        from storage import ShardedStore
        self.tmp = tempfile.mkdtemp()
        self.store = ShardedStore(self.tmp, "t", shard_size=2)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_update_preserves_id(self):
        rid = self.store.insert({"name": "v1", "version": 1})
        updated = self.store.update(rid, {"version": 2, "name": "v2"})
        self.assertEqual(updated["id"], rid)
        self.assertEqual(updated["version"], 2)
        self.assertEqual(updated["name"], "v2")
        fetched = self.store.get(rid)
        self.assertEqual(fetched["version"], 2)
        self.assertEqual(self.store.stats()["total"], 1)

    def test_update_missing_returns_none(self):
        self.assertIsNone(self.store.update("nope", {"x": 1}))


if __name__ == "__main__":
    unittest.main()
