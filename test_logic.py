"""ทดสอบตรรกะหลักของ Smart Tax Planner: python test_logic.py"""
from pathlib import Path
import tax_core as tc

# 1) ภาษีขั้นบันไดตรงกับค่าที่คำนวณมือ
for net, exp in {0: 0, 150_000: 0, 300_000: 7_500, 500_000: 27_500, 750_000: 65_000, 1_000_000: 115_000, 2_000_000: 365_000, 5_000_000: 1_265_000}.items():
    assert abs(tc.income_tax(net) - exp) < 1e-6, (net, exp)

# 2) สิทธิผู้สูงอายุ 65+
assert tc.net_income_before_funds(600_000, 0, 0, 0, age=64) - tc.net_income_before_funds(600_000, 0, 0, 0, age=65) == 190_000

# 3) แผนลงทุน: ไม่เกินงบ ไม่เกินวงเงิน แบ่งช่องทางถูก และภาษีไม่ติดลบ
import itertools
for age, inc, dep, ins, pvd, bud in itertools.product([25, 50, 70], [0, 120_000, 500_000, 3_000_000], [0, 3], [0, 50_000], [0, 400_000], [0, 30_000, 5_000_000]):
    r = tc.plan(age, inc, dep, ins, pvd, bud, 0.5)
    assert r["invest"] <= bud + 1e-6 and r["invest"] <= r["cap_total"] + 1e-6
    assert abs(r["rmf"] + r["thaiesg"] - r["invest"]) < 1e-6
    assert r["rmf"] <= r["rmf_cap"] + 1e-6 and r["thaiesg"] <= r["esg_cap"] + 1e-6
    assert r["tax_saved"] >= -1e-6 and r["tax_after"] <= r["tax_before"] + 1e-6 and r["max_saved"] >= r["tax_saved"] - 1e-6

# 4) โมเดลโหลดได้ และทุกกลุ่มมีคำอธิบาย/สัดส่วนหุ้น
model, info, profile, source = tc.load_model(Path(__file__).parent)
assert len(profile) == info["n_clusters"] and profile["equity_share"].between(0.1, 0.9).all()
c = tc.assign_cluster(model, 55, 700_000, 2, 20_000, 0, 1)
assert c in profile.index
print("ผ่านทุกการทดสอบ | แหล่งโมเดล:", source, "| จำนวนกลุ่ม:", len(profile))
