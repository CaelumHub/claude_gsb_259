"""信息抽取（IE）模块测试。

运行：``python3 -m unittest tests.test_ie -v``

覆盖：
- 简历 / 合同预置模板抽取（归一化、多种写法、缺失、冲突、列表）
- 命中偏移恒等式 text[start:end] == raw（可回查）
- 同一段文本不同模板 / 不同版本互不干扰
- 模板新版本不冲乱历史结果；同文本同版本 upsert 不产生重复
- 缺失显式标记，绝不填错值
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ie import InfoExtractionService
from nlp.extractor import (BUILTIN_TEMPLATES, FieldExtractor, Template,
                           TemplateError, parse_money, text_hash)
from storage import StoreRegistry


RESUME = """个人简历
基本信息
姓名：李明  性别：男
手机：138 1234 5678
电子邮箱：liming@example.com
身份证号：110101199003071234
最高学历：本科
毕业院校：北京大学

工作经历：
2016年7月-2019年6月 腾讯科技有限公司 后端工程师
2019年7月至今 华为技术有限公司 架构师

项目经历：
1. 订单中台建设
2. 风控规则引擎

自我评价
踏实肯干。
"""

CONTRACT = """技术服务合同
甲方（委托方）：上海星河科技有限公司
乙方（受托方）：北京云智数据有限公司
第一条 服务内容：数据平台开发。
第二条 合同期限：自2025年3月1日起至2025年12月31日止。
第三条 合同总金额为人民币捌拾万元整（小写：800,000元）。
联系电话：010-88889999
签订日期：2025年2月20日
甲方（盖章）：上海星河科技有限公司
乙方（盖章）：北京云智数据有限公司
"""


class TestExtractResume(unittest.TestCase):
    def setUp(self):
        self.r = FieldExtractor().extract(RESUME, BUILTIN_TEMPLATES[0])
        self.f = self.r["fields"]

    def test_scalar_fields(self):
        self.assertEqual(self.f["name"]["value"], "李明")
        self.assertEqual(self.f["email"]["value"], "liming@example.com")
        # 多种写法归一：138 1234 5678 -> 11 位
        self.assertEqual(self.f["phone"]["value"], "13812345678")
        self.assertEqual(self.f["idcard"]["value"], "110101199003071234")
        self.assertEqual(self.f["school"]["value"], "北京大学")

    def test_list_fields(self):
        exps = self.f["experience"]["value"]
        self.assertEqual(len(exps), 2)
        self.assertIn("腾讯科技有限公司", exps[0])
        self.assertIn("华为技术有限公司", exps[1])
        projs = self.f["projects"]["value"]
        self.assertEqual(len(projs), 2)
        self.assertTrue(all("自我评价" not in p for p in exps + projs))

    def test_missing_marked(self):
        # 文本里没有教育经历章节 -> 显式缺失，而不是瞎填
        self.assertIn("edu_exp", self.r["missing"])
        self.assertEqual(self.f["edu_exp"]["status"], "missing")
        self.assertIsNone(self.f["edu_exp"]["value"])

    def test_offsets_traceable(self):
        for key, fr in self.f.items():
            for m in fr["mentions"]:
                self.assertEqual(
                    RESUME[m["start"]:m["end"]], m["raw"],
                    msg=f"{key} 偏移对不上原文: {m}")

    def test_context_present(self):
        m = self.f["name"]["mentions"][0]
        self.assertTrue(m["context"])
        self.assertIn("李明", m["context"])


class TestExtractContract(unittest.TestCase):
    def setUp(self):
        self.r = FieldExtractor().extract(CONTRACT, BUILTIN_TEMPLATES[1])
        self.f = self.r["fields"]

    def test_parties_skip_role_brackets(self):
        # 「甲方（委托方）」角色括号不能当值；「（盖章）」落款不制造冲突
        self.assertEqual(self.r["conflicts"], [])
        self.assertEqual(self.f["party_a"]["value"], "上海星河科技有限公司")
        self.assertEqual(self.f["party_b"]["value"], "北京云智数据有限公司")

    def test_money_variants_unify(self):
        # 大写「捌拾万」与阿拉伯「800,000」归一为同一金额
        self.assertEqual(self.f["amount"]["status"], "found")
        self.assertEqual(self.f["amount"]["value"], "800000|CNY")

    def test_dates(self):
        self.assertEqual(self.f["start_date"]["value"], "2025-03-01")
        self.assertEqual(self.f["end_date"]["value"], "2025-12-31")
        self.assertEqual(self.f["sign_date"]["value"], "2025-02-20")

    def test_landline(self):
        self.assertEqual(self.f["start_date"]["status"], "found")

    def test_offsets(self):
        for key, fr in self.f.items():
            for m in fr["mentions"]:
                self.assertEqual(CONTRACT[m["start"]:m["end"]], m["raw"])


class TestConflictAndMissing(unittest.TestCase):
    def test_conflict_keeps_all_candidates(self):
        text = "甲方：甲公司\n乙方：乙公司\n合同总金额为人民币伍万元整。\n补充协议约定合同总金额：80000元。"
        r = FieldExtractor().extract(text, BUILTIN_TEMPLATES[1])
        amount = r["fields"]["amount"]
        self.assertEqual(amount["status"], "conflict")
        self.assertIn("amount", r["conflicts"])
        # 不猜值：value 为 None，两个候选都在
        self.assertIsNone(amount["value"])
        cand = {c["value"] for c in amount["candidates"]}
        self.assertEqual(cand, {"50000|CNY", "80000|CNY"})

    def test_same_value_different_writings_not_conflict(self):
        text = "合同总金额为人民币壹拾贰万元整（小写：120,000元）。"
        r = FieldExtractor().extract(text, BUILTIN_TEMPLATES[1])
        # 大写金额字段同时看到两种写法，规范值一致 => 非冲突
        self.assertEqual(r["fields"]["currency_text"]["status"], "found")

    def test_absent_fields_stay_missing(self):
        r = FieldExtractor().extract("甲方：甲公司\n乙方：乙公司\n",
                                     BUILTIN_TEMPLATES[1])
        self.assertIn("amount", r["missing"])
        self.assertIn("amount", r["missing_required"])
        self.assertEqual(r["fields"]["sign_date"]["status"], "missing")
        self.assertEqual(r["fields"]["sign_date"]["mentions"], [])


class TestMoneyParser(unittest.TestCase):
    def test_cn_amounts(self):
        self.assertEqual(parse_money("壹拾贰万元")["amount"], 120000)
        self.assertEqual(parse_money("捌拾万元整")["amount"], 800000)
        self.assertEqual(parse_money("人民币伍万元整")["amount"], 50000)
        self.assertEqual(parse_money("叁仟元整")["amount"], 3000)

    def test_arabic_amounts(self):
        self.assertEqual(parse_money("120,000元")["amount"], 120000)
        self.assertEqual(parse_money("80万元")["amount"], 800000)
        self.assertEqual(parse_money("1.5亿元")["amount"], 150000000)
        self.assertEqual(parse_money("100美元")["currency"], "USD")

    def test_plain_number_is_not_money(self):
        # 底层解析器对纯数字按「元」兜底；但抽取层的金额正则要求带币种/单位，
        # 保证正文里的普通数字、日期不会被当成金额。
        from nlp.extractor import _money_matches
        self.assertEqual(_money_matches("共120人参会，2024年3月开会"), [])
        self.assertEqual([m.raw for m in _money_matches("费用120元")], ["120元"])


# ---------------------------------------------------------------------------
# 服务层：模板版本、隔离、upsert、回查
# ---------------------------------------------------------------------------

class TestService(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.svc = InfoExtractionService(StoreRegistry(self.tmp, shard_size=50))
        self.svc.seed_builtins()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_builtin_templates_seeded_idempotent(self):
        names = [t["name"] for t in self.svc.list_templates()]
        self.assertIn("简历", names)
        self.assertIn("合同", names)
        self.assertEqual(self.svc.seed_builtins(), 0)  # 不重复写

    def test_extract_and_query(self):
        out = self.svc.extract(RESUME, "builtin_resume")
        rid = out["id"]
        got = self.svc.get_result(rid)
        self.assertIsNotNone(got)
        self.assertEqual(got["result"]["fields"]["name"]["value"], "李明")
        # 快照完整
        keys = {f["key"] for f in got["template_snapshot"]["fields"]}
        self.assertIn("phone", keys)

    def test_different_templates_isolated(self):
        # 同一段文本用简历、合同两套模板各抽一次
        a = self.svc.extract(RESUME, "builtin_resume")
        b = self.svc.extract(RESUME, "builtin_contract")
        self.assertNotEqual(a["id"], b["id"])
        res = self.svc.query_results(text_hash_eq=text_hash(RESUME))
        self.assertEqual(len(res), 2)
        tids = {r["template_id"] for r in res}
        self.assertEqual(tids, {"builtin_resume", "builtin_contract"})

    def test_upsert_same_text_same_version(self):
        o1 = self.svc.extract(RESUME, "builtin_resume")
        o2 = self.svc.extract(RESUME, "builtin_resume")
        self.assertIsNotNone(o2["replaced"])
        rows = self.svc.query_results(template_id="builtin_resume")
        self.assertEqual(len(rows), 1)

    def test_new_version_keeps_old_results(self):
        old = self.svc.extract(CONTRACT, "builtin_contract")
        old_version = old["record"]["template_version"]

        # 调整模板：新增「联系电话」字段
        latest = self.svc.get_template("builtin_contract")
        spec = {"name": latest["name"], "description": latest["description"],
                "fields": latest["fields"] + [{
                    "key": "contact_phone", "label": "联系电话",
                    "type": "phone", "aliases": ["联系电话", "电话"]}]}
        new_tpl = self.svc.new_version("builtin_contract", spec, note="加字段")
        self.assertEqual(new_tpl["version"], old_version + 1)

        new = self.svc.extract(CONTRACT, "builtin_contract")
        self.assertEqual(new["record"]["template_version"],
                         old_version + 1)
        # 新版本结果里有新字段
        self.assertIn("contact_phone", new["record"]["result"]["fields"])
        # 旧版本结果原封不动，仍可回查
        old_rec = self.svc.get_result(old["id"])
        self.assertIsNotNone(old_rec)
        self.assertEqual(old_rec["template_version"], old_version)
        self.assertNotIn("contact_phone", old_rec["result"]["fields"])
        # 两个版本结果并存、互不覆盖
        rows = self.svc.query_results(template_id="builtin_contract")
        self.assertEqual(len(rows), 2)
        # 显式按版本查询
        old_rows = self.svc.query_results(
            template_id="builtin_contract", version=old_version)
        self.assertEqual(len(old_rows), 1)

    def test_create_and_version_custom_template(self):
        spec = {"name": "请假条", "fields": [
            {"key": "applicant", "label": "申请人", "type": "person",
             "aliases": ["申请人", "请假人"], "required": True},
            {"key": "days", "label": "请假天数", "type": "text",
             "aliases": ["天数", "请假时间"]}]}
        t = self.svc.create_template(spec)
        self.assertEqual(t["version"], 1)
        text = "申请人：王芳\n因事请假，天数：3天。"
        out = self.svc.extract(text, t["template_id"])
        self.assertEqual(out["record"]["result"]["fields"]["applicant"]["value"],
                         "王芳")
        with self.assertRaises(TemplateError):
            self.svc.create_template(spec)  # 重名不允许直接新建

    def test_invalid_template(self):
        with self.assertRaises(TemplateError):
            self.svc.create_template({"name": "", "fields": []})
        with self.assertRaises(TemplateError):
            self.svc.create_template({"name": "x", "fields": [{"key": "1bad"}]})

    def test_flat_record_and_csv(self):
        self.svc.extract(CONTRACT, "builtin_contract")
        rows = self.svc.query_results(template_id="builtin_contract")
        flat = self.svc.flat_record(rows[0])
        self.assertIn("甲方", flat)
        self.assertEqual(flat["甲方__状态"], "已抽取")
        csv_text = self.svc.export_csv(rows)
        self.assertIn("甲方", csv_text)

    def test_missing_shown_in_flat_record(self):
        out = self.svc.extract("甲方：甲公司\n乙方：乙公司\n",
                               "builtin_contract")
        flat = self.svc.flat_record(self.svc.get_result(out["id"]))
        self.assertEqual(flat["合同金额__状态"], "缺失")
        self.assertEqual(flat["合同金额"], "")


class TestTemplateObject(unittest.TestCase):
    def test_duplicate_field_key(self):
        with self.assertRaises(TemplateError):
            Template({"name": "t", "fields": [
                {"key": "a"}, {"key": "a"}]})


if __name__ == "__main__":
    unittest.main()
