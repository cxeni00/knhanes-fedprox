"""KNHANES 기본DB(SPSS) 5개 연도 → 연합학습용 분석 데이터.

입력: data/raw/<연도>/HN<YY>_ALL.sav  (질병관리청 누리집에서 직접 내려받는다)
출력: data/processed/knhanes_fl.parquet  (year, y, 설명변수 13개)
"""
import glob
import os
import re

import numpy as np
import pandas as pd
import pyreadstat

YEARS = [2018, 2019, 2020, 2021, 2024]  # 2022·2023은 암 이환 문항이 없다
RAW = ["age", "sex", "incm", "edu", "town_t", "BS3_1", "BD1_11", "pa_aerobic", "D_1_1",
       "HE_BMI", "HE_sbp", "HE_glu", "HE_chol"]
FEATURES = ["age", "female", "incm", "edu", "rural", "smoke_cur", "drink_monthly", "pa_aerobic",
            "health_poor", "HE_BMI", "HE_sbp", "HE_glu", "HE_chol"]


def load_year(year):
    path = glob.glob(f"data/raw/{year}/*.sav")[0]
    _, meta = pyreadstat.read_sav(path, metadataonly=True)
    by_lower = {c.lower(): c for c in meta.column_names}  # 연도마다 대소문자가 다를 수 있다
    # 암 의사진단 여부. 2018·2020·2021은 9개(기타암 2개), 2019는 8개(기타암 1개), 2024는 통합 변수 1개
    dg = [c for c in meta.column_names if re.fullmatch(r"DC\d+_dg", c, re.IGNORECASE)]
    assert dg, f"{year}: 암 의사진단 변수가 없다"
    df, _ = pyreadstat.read_sav(path, usecols=[by_lower[c.lower()] for c in RAW] + dg)
    df = df.rename(columns={by_lower[c.lower()]: c for c in RAW})
    df = df[df.age >= 19]

    pos = (df[dg] == 1).any(axis=1)
    neg = (df[dg] == 0).all(axis=1)  # 8(비해당)·9(모름)·결측이 섞이면 제외
    out = pd.DataFrame({
        "year": year,
        "y": np.where(pos, 1.0, np.where(neg, 0.0, np.nan)),
        "age": df.age,
        "female": (df.sex == 2).astype(float),
        "incm": df.incm,
        "edu": df.edu,
        "rural": (df.town_t == 2).astype(float),
        # 현재흡연: 1 매일, 2 가끔 / 3 과거, 8 비해당(비흡연) / 9 무응답
        "smoke_cur": df.BS3_1.map({1: 1, 2: 1, 3: 0, 8: 0}),
        # 월간음주: 3~6 월 1회 이상 / 1 최근 1년 안 마심, 2 월 1회 미만, 8 비해당(비음주) / 9 무응답
        "drink_monthly": df.BD1_11.map({1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 1, 8: 0}),
        "pa_aerobic": df.pa_aerobic,
        # 주관적 건강: 1 매우 좋음 ~ 5 매우 나쁨 / 9 무응답
        "health_poor": df.D_1_1.where(df.D_1_1 <= 5),
        "HE_BMI": df.HE_BMI, "HE_sbp": df.HE_sbp, "HE_glu": df.HE_glu, "HE_chol": df.HE_chol,
    })
    return out, len(df)


if __name__ == "__main__":
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    rows, parts = [], []
    for y in YEARS:
        d, n_adult = load_year(y)
        known = d.dropna(subset=["y"])
        cc = d.dropna()  # 원 논문과 같이 결측이 있는 대상자는 제외
        rows.append({"year": y, "성인": n_adult, "암 문항 응답": len(known), "분석 대상": len(cc),
                     "양성": int(cc.y.sum()), "양성 비율(%)": round(100 * cc.y.mean(), 2)})
        parts.append(cc)
    data = pd.concat(parts, ignore_index=True)
    assert list(data.columns) == ["year", "y"] + FEATURES and not data.isna().any().any()
    data.to_parquet("data/processed/knhanes_fl.parquet")
    table = pd.DataFrame(rows)
    table.to_csv("results/cohort.csv", index=False)
    print(table.to_string(index=False))
