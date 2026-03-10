# Battery Data Standardizer (BDS)

AI 기반 이기종 배터리 데이터 자동 표준화 도구.

LLM(EXAONE)이 **어떤 파일 포맷이든** 구조를 스스로 파악하고, 필드명을 이해해서 스키마에 매핑하고, 추출 코드를 직접 생성·실행하여 데이터를 변환합니다. 하드코딩된 파서 없이, 새 데이터셋도 코드 작성 없이 표준화할 수 있습니다.

## 아키텍처

```
Input (어떤 파일이든)
    │
    ▼
[File Inspector] ─── 파일 구조를 텍스트 프리뷰로 변환
    │
    ▼
[LLM Agent] ─── EXAONE이 구조를 보고 판단
    │   ├─ 1차: 추출 코드 생성 & 실행 (Code Generation)
    │   └─ 폴백: Tool Use로 반복 탐색 & 추출
    │
    ▼
[Sandbox Executor] ─── 생성된 코드를 안전하게 실행
    │
    ▼
[Validator & Exporter] ─── 결과 검증 → CellRecord pickle 출력
```

## 지원 포맷

| 포맷 | 확장자 | 비고 |
|------|--------|------|
| CSV/TSV | `.csv`, `.tsv`, `.txt` | 자동 구분자 감지 |
| Excel | `.xlsx`, `.xls` | 멀티시트 지원 |
| MATLAB v5 | `.mat` | scipy.io.loadmat |
| MATLAB v7.3 | `.mat` | HDF5 (h5py) |
| HDF5 | `.h5`, `.hdf5` | 계층 구조 탐색 |
| JSON | `.json` | 중첩 구조 지원 |
| Pickle | `.pkl`, `.pickle` | Python 객체 |
| ZIP/TAR/GZ | `.zip`, `.tar.gz` 등 | 자동 해제 후 처리 |

## 출력

[BatteryFoundationFramework](https://github.com/pjmbatman/BatteryFoundationFramework)의 `CellRecord` 호환 pickle 파일로 출력됩니다.

```python
# BFF에서 바로 사용 가능
from pipeline.standardizer.cell_record import CellRecord
cell = CellRecord.load("output/B0025.pkl")
```

## 빠른 시작

### 1. 설치

```bash
# uv 설치 (없는 경우)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 프로젝트 클론 & 의존성 설치
git clone https://github.com/pjmbatman/battery-data-standardizer.git
cd battery-data-standardizer
uv venv
uv pip install -e ".[dev]"
```

### 2. LLM 서버 실행

vLLM으로 EXAONE 모델을 서빙합니다. GPU 서버에서 실행하세요.

```bash
# 기본 모델 (EXAONE-4.0-32B-FP8, ~34GB VRAM)
bash scripts/serve_model.sh

# 또는 다른 모델 지정
MODEL=LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct bash scripts/serve_model.sh

# 서버 상태 확인
curl http://localhost:8000/v1/models
```

> 모델이 없으면 자동으로 다운로드됩니다. 처음 실행 시 시간이 걸릴 수 있습니다.

### 3. 데이터 표준화

```bash
# 파일 하나 표준화
uv run bds standardize ./data/sample.csv -o ./output/

# 디렉토리 전체 표준화 (배치 모드)
uv run bds standardize ./raw_data/ -o ./standardized/ --batch

# 다른 vLLM 서버 사용
uv run bds standardize ./data/ -o ./output/ --api-base http://gpu-server:8000/v1
```

### 4. 파일 구조 미리보기 (LLM 호출 없이)

```bash
uv run bds inspect ./data/B0025.mat
```

출력 예시:
```
[MAT v5, 2,753,821 bytes]
B0025: structured array, shape=(1, 1), fields=['cycle']
  .cycle: shape=(1, 80), dtype=[('type', 'O'), ('ambient_temperature', 'O'), ...]

## B0025.cycle: 80 records
Types: {'impedance': 21, 'charge': 31, 'discharge': 28}
...
```

### 5. 캐시 관리

같은 구조의 파일은 LLM 호출 없이 캐시된 코드를 재활용합니다.

```bash
# 캐시 목록 조회
uv run bds cache list

# 캐시 초기화
uv run bds cache clear
```

## 설정

`configs/default.yaml`에서 설정을 변경할 수 있습니다.

```yaml
llm:
  api_base: "http://localhost:8000/v1"   # vLLM 서버 주소
  model: "LGAI-EXAONE/EXAONE-4.0-32B-FP8"
  temperature: 0.1
  max_retries: 3          # 코드 생성 재시도 횟수

sandbox:
  timeout: 120            # 코드 실행 타임아웃 (초)

agent:
  max_tool_steps: 20      # Tool Use 모드 최대 스텝
  fallback_to_tool_use: true

cache:
  enabled: true
  db_path: ".bds_cache/cache.db"
```

## 동작 원리

1. **File Inspector**: 파일을 텍스트 프리뷰로 변환 (구조, 필드명, 샘플 데이터)
2. **Code Generation Agent (1차)**: LLM이 프리뷰를 보고 완전한 추출 Python 스크립트를 생성
3. **Sandbox Execution**: 생성된 코드를 subprocess로 실행, JSON 출력을 파싱
4. **Validation**: 전압 범위(0~6V), 시간 단조증가, 배열 길이 일관성 등 검증
5. **Auto-retry**: 실행 오류 또는 검증 실패 시 에러를 LLM에게 보여주고 코드 수정 (최대 3회)
6. **Tool Use Agent (폴백)**: 코드 생성 실패 시, LLM이 도구(inspect, read_sample, extract, execute_code, profile)를 사용하여 단계적 탐색·추출
7. **Cache**: 성공한 추출 코드를 파일 구조 시그니처(헤더 해시 등)로 캐싱 → 같은 구조의 다른 파일에 재활용

## 테스트

```bash
uv run python -m pytest tests/
```

## 프로젝트 구조

```
battery-data-standardizer/
├── configs/default.yaml          # 설정
├── scripts/
│   ├── serve_model.sh            # vLLM 서빙 스크립트
│   └── download_model.py         # 모델 다운로드
├── src/bds/
│   ├── cli.py                    # CLI (standardize, inspect, cache)
│   ├── config.py                 # 설정 로드
│   ├── pipeline.py               # 전체 파이프라인
│   ├── inspector/
│   │   ├── preview.py            # 파일 구조 프리뷰
│   │   └── archive.py            # 아카이브 해제
│   ├── agent/
│   │   ├── orchestrator.py       # Agent 오케스트레이터
│   │   ├── code_generator.py     # 코드 생성 Agent
│   │   ├── tool_use.py           # Tool Use Agent (폴백)
│   │   ├── tools.py              # Tool 정의
│   │   ├── prompts.py            # 프롬프트 템플릿
│   │   └── llm_client.py         # vLLM 클라이언트
│   ├── sandbox/
│   │   └── executor.py           # 코드 실행 환경
│   ├── schema.py                 # CellRecord/CycleRecord 스키마
│   ├── validator.py              # 출력 데이터 검증
│   ├── exporter.py               # pickle 출력
│   └── cache.py                  # SQLite 캐시
└── tests/                        # 유닛 테스트 (38개)
```

## 검증 결과

| 데이터셋 | 포맷 | 사이클 수 | 성공 시도 |
|---------|------|----------|----------|
| SNL | CSV | 388 | 1차 |
| CALCE | XLSX | 7 | 1차 |
| UL-PUR | CSV | 205 | 1차 |
| HNEI | CSV (45MB) | 1101 | 캐시 |
| NASA | MAT v5 | 59 | 2차 |

## 요구 사항

- Python >= 3.10
- GPU 서버 (vLLM 서빙용, EXAONE-4.0-32B-FP8 기준 ~34GB VRAM)
- [uv](https://github.com/astral-sh/uv) (패키지 관리)
