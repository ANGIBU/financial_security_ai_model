# config.py

"""
금융보안 AI 시스템 설정 파일
- 모델 설정
- 시스템 환경 설정
- 성능 최적화 설정
- 평가 기준 설정
"""

import os
from pathlib import Path

# === 기본 환경 설정 ===
DEFAULT_MODEL_NAME = "upstage/SOLAR-10.7B-Instruct-v1.0"
DEVICE_AUTO_SELECT = True  # True면 자동으로 cuda/cpu 선택
VERBOSE_MODE = False

# 오프라인 모드 설정
OFFLINE_MODE = {"TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"}

# === 디렉토리 설정 ===
BASE_DIR = Path(__file__).parent.absolute()
PKL_DIR = BASE_DIR / "pkl"
JSON_CONFIG_DIR = BASE_DIR / "configs"

# 기본 파일 경로
DEFAULT_FILES = {
    "test_file": "./test.csv",
    "submission_file": "./sample_submission.csv",
    "output_file": "./final_submission.csv",
    "test_output_file": "./test_result.csv",
}

# === 모델 설정 ===
MODEL_CONFIG = {
    "torch_dtype": "bfloat16",
    "device_map": "auto",
    "trust_remote_code": True,
    "use_fast_tokenizer": True,
}

# 생성 설정 (반복 방지 강화)
GENERATION_CONFIG = {
    "multiple_choice": {
        "max_new_tokens": 15,
        "temperature": 0.3,
        "top_p": 0.8,
        "do_sample": True,
        "repetition_penalty": 1.2,
        "no_repeat_ngram_size": 3,
    },
    "subjective": {
        "max_new_tokens": 300,
        "temperature": 0.5,
        "top_p": 0.85,
        "do_sample": True,
        "repetition_penalty": 1.3,
        "no_repeat_ngram_size": 4,
        "length_penalty": 1.1,
    },
}

# === 성능 최적화 설정 ===
OPTIMIZATION_CONFIG = {
    "intent_confidence_threshold": 0.6,
    "quality_threshold": 0.7,
    "korean_ratio_threshold": 0.8,
    "max_retry_attempts": 2,
    "template_preference": True,
    "adaptive_prompt": True,
    "mc_pattern_priority": True,
    "domain_specific_optimization": True,
    "institution_question_priority": True,
    "mc_context_weighting": True,
    "repetition_detection_enabled": True,
    "critical_pattern_monitoring": True,
    "early_repetition_cutoff": True,
    "repetition_penalty_adaptive": True,
}

# === 한국어 처리 설정 ===
KOREAN_REQUIREMENTS = {
    "min_korean_ratio": 0.8,
    "max_english_ratio": 0.1,
    "min_length": 30,
    "max_length": 500,
    "repetition_tolerance": 2,
    "critical_repetition_limit": 3,
}

# === 메모리 관리 설정 ===
MEMORY_CONFIG = {
    "gc_frequency": 50,  # 몇 문항마다 가비지 컬렉션 실행
    "save_interval": 1000,  # 학습 데이터 저장 간격
    "max_learning_records": {
        "successful_answers": 1000,
        "failed_answers": 500,
        "quality_scores": 1000,
        "choice_range_errors": 100,
        "repetitive_answers": 200,
    },
}

# === 시간 제한 설정 ===
TIME_LIMITS = {
    "total_inference_minutes": 270,  # 4시간 30분
    "warmup_timeout": 30,  # 워밍업 제한시간 (초)
    "single_question_timeout": 30,  # 단일 질문 제한시간 (초)
    "generation_timeout": 20,  # 생성 제한시간 (초)
}

# === 진행률 표시 설정 ===
PROGRESS_CONFIG = {
    "bar_length": 50,
    "update_frequency": 1,  # 몇 문항마다 진행률 업데이트
}

# === 로깅 설정 ===
LOGGING_CONFIG = {
    "enable_stats_logging": True,
    "enable_error_logging": True,
    "log_processing_times": True,
    "log_quality_scores": True,
    "log_repetition_patterns": True,
}

# === 템플릿 품질 평가 기준 ===
TEMPLATE_QUALITY_CRITERIA = {
    "length_range": (50, 400),
    "korean_ratio_min": 0.9,
    "structure_keywords": ["법", "규정", "조치", "관리", "절차", "기준"],
    "intent_keywords": {
        "기관_묻기": ["위원회", "기관", "담당", "업무"],
        "특징_묻기": ["특징", "특성", "성질", "기능"],
        "지표_묻기": ["지표", "징후", "패턴", "탐지"],
        "방안_묻기": ["방안", "대책", "조치", "관리"],
        "절차_묻기": ["절차", "과정", "단계", "순서"],
        "조치_묻기": ["조치", "대응", "보안", "예방"],
    },
    "forbidden_patterns": [
        "갈취 묻는 말",
        "묻고 갈취",
        "갈취",
        r"(.{2,8})\s*\1\s*\1\s*\1",
        r"(.{1,3})\s*(\1\s*){4,}",
    ],
}

# === 테스트 설정 ===
TEST_CONFIG = {
    "test_sizes": {"mini": 8, "basic": 50, "detailed": 100},
    "default_test_size": 50,
}

# === 파일 검증 설정 ===
FILE_VALIDATION = {
    "required_files": ["test.csv", "sample_submission.csv"],
    "encoding": "utf-8-sig",
    "max_file_size_mb": 100,
}

# === 통계 설정 ===
STATS_CONFIG = {
    "track_answer_distribution": True,
    "track_domain_performance": True,
    "track_intent_accuracy": True,
    "track_template_effectiveness": True,
    "calculate_reliability_score": True,
    "track_repetition_patterns": True,
    "monitor_quality_degradation": True,
}

# === 반복 패턴 모니터링 설정 ===
REPETITION_MONITORING = {
    "critical_patterns": [
        "갈취 묻는 말",
        "묻고 갈취",
        "갈취",
    ],
    "repetition_thresholds": {
        "word_repeat_limit": 4,
        "phrase_repeat_limit": 3,
        "sentence_repeat_limit": 2,
    },
    "pattern_detection_sensitivity": 0.8,
    "early_termination_enabled": True,
    "fallback_trigger_count": 3,
}

# === JSON 설정 파일 경로 ===
JSON_CONFIG_FILES = {
    "knowledge_data": JSON_CONFIG_DIR / "knowledge_data.json",
    "model_config": JSON_CONFIG_DIR / "model_config.json",
    "processing_config": JSON_CONFIG_DIR / "processing_config.json",
}


# === 환경 변수 설정 함수 ===
def setup_environment():
    """환경 변수 설정"""
    for key, value in OFFLINE_MODE.items():
        os.environ[key] = value


def get_device():
    """디바이스 자동 선택"""
    if DEVICE_AUTO_SELECT:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cpu"


def ensure_directories():
    """필요한 디렉토리 생성"""
    PKL_DIR.mkdir(exist_ok=True)
    JSON_CONFIG_DIR.mkdir(exist_ok=True)


# === 설정 검증 함수 ===
def validate_config():
    """설정 값 검증"""
    errors = []

    # 한국어 비율 검증
    if not 0 <= KOREAN_REQUIREMENTS["min_korean_ratio"] <= 1:
        errors.append("min_korean_ratio는 0과 1 사이여야 합니다")

    # 시간 제한 검증
    if TIME_LIMITS["total_inference_minutes"] <= 0:
        errors.append("total_inference_minutes는 양수여야 합니다")

    # 최적화 임계값 검증
    if not 0 <= OPTIMIZATION_CONFIG["intent_confidence_threshold"] <= 1:
        errors.append("intent_confidence_threshold는 0과 1 사이여야 합니다")

    # 반복 패턴 설정 검증
    if (
        "repetition_thresholds" in REPETITION_MONITORING
        and "word_repeat_limit" in REPETITION_MONITORING["repetition_thresholds"]
    ):
        if REPETITION_MONITORING["repetition_thresholds"]["word_repeat_limit"] < 2:
            errors.append("word_repeat_limit는 2 이상이어야 합니다")

    if not 0 <= REPETITION_MONITORING["pattern_detection_sensitivity"] <= 1:
        errors.append("pattern_detection_sensitivity는 0과 1 사이여야 합니다")

    if errors:
        raise ValueError(f"설정 오류: {'; '.join(errors)}")

    return True


# === 반복 패턴 감지 설정 조정 함수 ===
def adjust_repetition_sensitivity(level: str = "medium"):
    """반복 패턴 감지 민감도 조정"""
    sensitivity_levels = {
        "low": {
            "word_repeat_limit": 6,
            "phrase_repeat_limit": 4,
            "sentence_repeat_limit": 3,
            "pattern_detection_sensitivity": 0.6,
        },
        "medium": {
            "word_repeat_limit": 4,
            "phrase_repeat_limit": 3,
            "sentence_repeat_limit": 2,
            "pattern_detection_sensitivity": 0.8,
        },
        "high": {
            "word_repeat_limit": 3,
            "phrase_repeat_limit": 2,
            "sentence_repeat_limit": 1,
            "pattern_detection_sensitivity": 0.9,
        },
    }

    if level in sensitivity_levels:
        # repetition_thresholds 키 업데이트
        for key, value in sensitivity_levels[level].items():
            if key != "pattern_detection_sensitivity":
                REPETITION_MONITORING["repetition_thresholds"][key] = value

        # 패턴 감지 민감도 업데이트
        REPETITION_MONITORING["pattern_detection_sensitivity"] = sensitivity_levels[
            level
        ]["pattern_detection_sensitivity"]


# === 생성 설정 동적 조정 함수 ===
def adjust_generation_for_repetition_risk(
    question_type: str, risk_level: str = "medium"
):
    """반복 위험에 따른 생성 설정 조정"""
    risk_adjustments = {
        "low": {
            "repetition_penalty": 1.1,
            "no_repeat_ngram_size": 2,
            "temperature": 0.6,
        },
        "medium": {
            "repetition_penalty": 1.3,
            "no_repeat_ngram_size": 4,
            "temperature": 0.5,
        },
        "high": {
            "repetition_penalty": 1.5,
            "no_repeat_ngram_size": 5,
            "temperature": 0.4,
        },
    }

    if risk_level in risk_adjustments and question_type in GENERATION_CONFIG:
        # 기존 설정을 유지하면서 업데이트
        for key, value in risk_adjustments[risk_level].items():
            GENERATION_CONFIG[question_type][key] = value


# === 초기화 함수 ===
def initialize_system():
    """시스템 초기화"""
    setup_environment()
    ensure_directories()

    # REPETITION_MONITORING 초기 설정 확인
    if "repetition_thresholds" not in REPETITION_MONITORING:
        REPETITION_MONITORING["repetition_thresholds"] = {
            "word_repeat_limit": 4,
            "phrase_repeat_limit": 3,
            "sentence_repeat_limit": 2,
        }

    validate_config()

    # 반복 패턴 모니터링 기본 설정
    adjust_repetition_sensitivity("medium")

    if VERBOSE_MODE:
        print("시스템 설정 완료")
        print(f"기본 모델: {DEFAULT_MODEL_NAME}")
        print(f"디바이스: {get_device()}")
        print(f"오프라인 모드: {OFFLINE_MODE}")
        print(
            f"반복 패턴 모니터링: {OPTIMIZATION_CONFIG['repetition_detection_enabled']}"
        )
        print(
            f"반복 감지 민감도: {REPETITION_MONITORING['pattern_detection_sensitivity']}"
        )


# 자동 초기화 (모듈 import 시 실행)
if __name__ != "__main__":
    try:
        initialize_system()
    except Exception as e:
        print(f"설정 초기화 중 오류: {e}")
        # 기본 설정으로 폴백
        try:
            setup_environment()
            ensure_directories()
            print("기본 설정으로 시스템을 시작합니다.")
        except Exception as fallback_error:
            print(f"기본 설정 로드도 실패: {fallback_error}")
            print("수동으로 설정을 확인해주세요.")
