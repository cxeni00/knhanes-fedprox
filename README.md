# KNHANES 조사 연도별 연합학습 재현 실험

Asad et al. (arXiv:2601.17183)의 FedProx 로지스틱 회귀 실험을 국민건강영양조사 5개 연도(2018, 2019, 2020, 2021, 2024)로 재현한다. 조사 연도 1개가 client 1개다.

## 실행

Python 3.10 이상이 필요하다.

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

# 1. 원시자료 준비 (아래 "데이터" 참고)
# 2. 전처리 → data/processed/knhanes_fl.parquet, results/cohort.csv
.venv/bin/python prep.py
# 3. 구현 검증
.venv/bin/python test_fl.py
# 4. 전체 실험 (μ 5종 × 50 seed) → results/*.csv
.venv/bin/python run.py
# 5. 그림 → results/fig*.png
.venv/bin/python plots.py
```

`python run.py 2` 처럼 seed 개수를 주면 빠르게 돌려 볼 수 있다. `python run.py analyse` 는 학습 없이 저장된 `results/runs.csv` 로 요약과 검정만 다시 계산한다.

## 데이터

원시자료는 이용 서약상 재배포할 수 없어 포함하지 않았다. 질병관리청 국민건강영양조사 누리집(https://knhanes.kdca.go.kr) → 원시자료 → 다운로드에서 2018, 2019, 2020, 2021, 2024년 기본DB를 SPSS 형식으로 받아 아래처럼 둔다.

```
data/raw/2018/HN18_all.sav
data/raw/2019/HN19_all.sav
data/raw/2020/HN20_all.sav
data/raw/2021/HN21_all.sav
data/raw/2024/HN24_ALL.sav
```

## 파일

| 파일 | 내용 |
|---|---|
| `prep.py` | 대상자 선정(만 19세 이상), 결과변수(암 의사진단 경험), 설명변수 13개 생성, 결측 제외 |
| `fl.py` | 로지스틱 회귀, FedProx 로컬 업데이트, 표본 수 가중 집계. 상수는 원 논문 설정 |
| `run.py` | 중앙집중 / 기관 단독 / FedAvg / FedProx 실행, 지표, t-검정 |
| `plots.py` | 그림 2개 |
| `test_fl.py` | 구현 검증 테스트 |
| `results/` | 실행 결과 (`runs.csv` 는 seed·방법·client 별 원자료) |

## 결과 파일 읽는 법

- `summary.csv`: 방법 × 범위(`pooled` 또는 조사 연도)별 평균과 표준편차
- `tests.csv`: 선택된 FedProx 와 비교 대상의 대응표본 t-검정
- 이름이 `_prev` 로 끝나는 지표는 임계값을 학습 데이터 양성 비율로 둔 추가 분석이다. 나머지는 임계값 0.5 이다.
