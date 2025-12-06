# Golf AI

로컬에서는 VS Code로 개발하고, Google Colab의 무료 GPU에서 실행하는 워크플로우를 위한 기본 스캐폴드입니다. 동일한 코드베이스로 로컬 디버깅(소규모)과 Colab 학습(대규모)을 함께 지원합니다.

## 디렉터리 구조와 역할

- `notebooks/`: Colab/로컬에서 열어볼 노트북(.ipynb) 보관. 셋업과 실험 기록용.
- `src/`: 실제 파이썬 패키지 코드. 데이터셋/모델/유틸 모듈 포함.
  - `src/dataset/golfdb_dataset.py`: GolfDB 데이터셋 + 변환.
  - `src/models/event_detector.py`, `src/models/backbone/mobilenet_v2.py`: EventDetector와 백본.
  - `src/utils/paths.py`: 로컬/Colab 공용 경로 헬퍼(`get_data_path` 등).
  - `src/utils/video_utils.py`: 학습 보조 유틸(AverageMeter, freeze_layers 등).
  - `src/golf_ai/trainer.py`: Trainer 클래스와 공용 학습 로직.
- `data/`: 로컬/Colab에서 공통으로 쓰는 데이터 위치(추적 제외). 필요하면 Google Drive 마운트 후 심볼릭 링크 사용.
- `configs/`: 실험/학습 설정 YAML. 여러 실험을 구성 파일로 관리.
- `requirements.txt`: Colab/로컬 공용 파이썬 의존성 목록.
- `README.md`: 프로젝트 개요와 Colab 실행 가이드.
- `train.py`: 공용 학습 엔트리포인트. `--debug`로 소규모 디버깅 모드 지원. `golfDB.pkl`을 내부 80/20으로 나눠 학습/검증에 사용.

```text
Raw Video
   ↓
Dataset (GolfDBDataset)
   ↓
Model (EventDetector / Backbone)
   ↓
Trainer (train.py / src/golf_ai/trainer.py)
   ↓
Checkpoints & Metrics
```

## 초기 포함 파일

- `notebooks/hailo-study.ipynb`: Colab/로컬에서 열어볼 실험 노트북 예시.
- `src/dataset/golfdb_dataset.py`: 데이터셋/변환 구현.
- `src/models/event_detector.py`: EventDetector 구현.
- `src/utils/paths.py`: 경로 헬퍼.
- `src/utils/video_utils.py`: 학습/검증 유틸.
- `src/golf_ai/trainer.py`: Trainer/CLI 공용 로직.
- `configs/example.yaml`: 예제 학습 설정.
- `data/.gitkeep`: 빈 데이터 디렉터리를 git에 포함하기 위한 파일.

## Colab에서 돌리는 방법 (초기 설정 셀)

아래 코드를 Colab 노트북 첫 셀에 붙여 넣고 실행하세요. 깃 저장소가 이미 있으면 `git pull`로 업데이트합니다. GPU 사용 가능 여부도 함께 확인합니다.

```python
# 런타임: GPU 선택 (Runtime > Change runtime type > GPU)
import os
import subprocess
from pathlib import Path

REPO_URL = "https://github.com/Raunplaymore/hailo-ai.git"
REPO_DIR = "hailo-ai"


def run(cmd: str, cwd: str | None = None) -> None:
    print(f"\n[cmd] {cmd}")
    subprocess.run(cmd, shell=True, check=True, cwd=cwd)


print("Step 1) clone or update repo")
if not Path(REPO_DIR).exists():
    run(f"git clone {REPO_URL}")
else:
    run("git pull", cwd=REPO_DIR)

print("Step 2) install requirements")
run(f"pip install -r {REPO_DIR}/requirements.txt")

print("Step 3) add src to PYTHONPATH")
src_path = str((Path(REPO_DIR) / "src").resolve())
os.environ["PYTHONPATH"] = src_path + os.pathsep + os.environ.get("PYTHONPATH", "")
print("PYTHONPATH ->", os.environ["PYTHONPATH"])

print("Step 4) check GPU availability")
try:
    import torch

    gpu_ok = torch.cuda.is_available()
    print("torch.cuda.is_available() ->", gpu_ok)
    if gpu_ok:
        print("GPU name ->", torch.cuda.get_device_name(0))
except ImportError:
    print("PyTorch가 아직 설치되지 않았습니다. requirements 설치를 먼저 진행하세요.")
```

## 학습 실행 예시

```bash
# 기본 학습 (golfDB.pkl을 내부 80/20으로 분할)
python train.py --iterations 2000

# 로컬 디버깅(작은 데이터/짧은 반복/더 잦은 로그)
python train.py --debug

# 사전학습 미사용 + 가중치 경로 지정
python train.py --no-pretrain --weights-path /path/to/mobilenet_v2.pth.tar

# 체크포인트에서 이어서 학습
python train.py  --split 1 --iterations 2000 --resume checkpoints/split1_iter1200.pth.tar
# (모델 state_dict만 있는 .pth도 지원, 파일명에서 iter 숫자를 추출해 이어서 진행)

# (선택) 데이터 루트 지정: 기본은 data/ (Colab은 /content/hailo-ai/data)
# export DATA_ROOT=/content/hailo-ai/data/golf_db
```

## 평가/추론 실행 예시

```bash
# 평가 (val_split_{n}.pkl 사용)
python eval.py --split 1 --checkpoint checkpoints/split1_iter2000.pth

# 비디오 추론 (프레임별 클래스/확률 출력)
python inference.py --video data/test_video.mp4 --checkpoint checkpoints/split1_iter2000.pth
```

## 경로/환경 헬퍼

- 데이터 루트: `get_data_root()`는 Colab이면 `/content/hailo-ai/data`, 로컬이면 `<프로젝트>/data`를 기본으로 사용 (`DATA_ROOT` 환경변수로 오버라이드 가능).
- 하위 경로: `get_data_path("videos_160")`처럼 호출해 상대 경로를 조합.

## 로컬 개발 메모 (VS Code)

- 루트(여기)에서 VS Code를 열고 `src/`를 워크스페이스 루트로 삼아 파이썬 패키지 개발.
- 데이터/모델/유틸은 각각 `src/dataset/`, `src/models/`, `src/utils/`에 추가.
- 테스트/스크립트는 `tests/`나 노트북에서 실행.
- 가상환경(venv/conda)에서 `pip install -r requirements.txt` 후 개발.
- 로컬 개발 시 Python 3.10/3.11 기반 venv 권장. PyTorch 2.4+ 기준으로 작성되었으며, Python 3.13에서는 일부 휠이 아직 제공되지 않을 수 있습니다(`torch~=2.4` 등 확인).

## Dataset 준비 (GolfDB)

- `data/golf_db/golfDB.mat` 또는 `golfDB.pkl` 위치에 GolfDB annotation 파일을 두면 됩니다.
- 비디오 프레임/클립 데이터는 선택 사항이며, YOLO 기반 트래킹이나 시각화에 사용할 수 있습니다.
- 내부 경로 헬퍼(`src/utils/paths.py`)는 기본적으로 `data/` 아래를 루트로 가정합니다(필요 시 `DATA_ROOT`로 오버라이드, 예: `DATA_ROOT=/content/hailo-ai/data/golf_db`).

## 다음에 할 일 아이디어

- 데이터 전처리/다운로드 스크립트 추가(`scripts/` 또는 노트북).
- `configs/`에 실험별 YAML을 늘리고, 노트북에서 옵션 인자로 선택하도록 개선.
- CI에서 lint/format/test 자동화 추가.

## Roadmap

- YOLO 기반 공/클럽 트래킹 통합
- ConvNeXt + Transformer 기반 EventDetector 도입
- Hailo-8 / Raspberry Pi 5용 경량 추론 파이프라인 추가
- 실제 골프 스윙 데이터(Down-the-line, face-on) 지원
- 테스트/CI에서 기본 학습 스모크 테스트 자동화

## DTL (Down-the-Line) 이벤트 라벨링 가이드

- **영상/라벨 위치**
  - 원본: `data/dtl_raw/` (예: `dtl_001.mp4`, `dtl_002.mp4` …)
  - 라벨: `data/dtl_labels/` (JSON)
- **이벤트 정의** (단일 프레임 인덱스)  
  `address`, `top`, `impact`, `finish`
- **라벨러 실행** (`tools/label_dtl_events.py`)
  ```bash
  python tools/label_dtl_events.py \
    --video data/dtl_raw/dtl_001.mp4 \
    --output data/dtl_labels/dtl_001.json
  ```
  조작: `a/d`(±1), `s/w`(±10), `1~4`(이벤트 설정), `space/enter`(4개 설정 시 저장), `q`(저장 없이 종료)
- **JSON 예시**
  ```json
  {
    "video": "dtl_001.mp4",
    "fps": 60.0,
    "num_frames": 240,
    "events": {
      "address": 10,
      "top": 115,
      "impact": 150,
      "finish": 210
    }
  }
  ```
- **워크플로우**  
  영상 수집 → `data/dtl_raw/` 저장 → 라벨러로 4 이벤트 지정 → JSON 확인/수정 → `data/dtl_labels/`에 축적 → DTL 전용 데이터셋/모델 파인튜닝  
  (라벨 5~10개만 있어도 초기 모델 가능, 늘릴수록 성능 안정화)

### DTL 후속 아이디어

- GolfDTLDataset 구현 → JSON 기반 로더
- 기존 EventDetector에 DTL 파인튜닝 스크립트 추가 (예: `train_dtl.py`)
- DTL 전용 temporal smoothing/GRU 확장
- Hailo 가속 버전으로 변환해 Pi 디바이스 실시간 실행
