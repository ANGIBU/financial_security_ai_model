# config.py

import os
from pathlib import Path

DEFAULT_MODEL_NAME = "upstage/SOLAR-10.7B-Instruct-v1.0"
DEVICE_AUTO_SELECT = True
VERBOSE_MODE = False

OFFLINE_MODE = {"TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"}

BASE_DIR = Path(__file__).parent.absolute()
PKL_DIR = BASE_DIR / "pkl"
LOG_DIR = BASE_DIR / "log"

DEFAULT_FILES = {
    "test_file": "./test.csv",
    "submission_file": "./sample_submission.csv",
    "output_file": "./final_submission.csv",
    "test_output_file": "./test_result.csv",
}

# pkl 학습 데이터 파일 경로
PKL_FILES = {
    "successful_answers": PKL_DIR / "successful_answers.pkl",
    "failed_answers": PKL_DIR / "failed_answers.pkl",
    "question_patterns": PKL_DIR / "question_patterns.pkl",
    "domain_templates": PKL_DIR / "domain_templates.pkl",
    "mc_patterns": PKL_DIR / "mc_patterns.pkl",
    "performance_data": PKL_DIR / "performance_data.pkl",
}

# 로그 파일 경로
LOG_FILES = {
    "main_log": LOG_DIR / "inference_log.txt",
    "performance_log": LOG_DIR / "performance_log.txt",
    "error_log": LOG_DIR / "error_log.txt",
    "monitoring_log": LOG_DIR / "monitoring_log.txt",
}

MODEL_CONFIG = {
    "torch_dtype": "bfloat16",
    "device_map": "auto",
    "trust_remote_code": True,
    "use_fast_tokenizer": True,
}

GENERATION_CONFIG = {
    "multiple_choice": {
        "max_new_tokens": 15,
        "temperature": 0.2,
        "top_p": 0.7,
        "do_sample": True,
        "repetition_penalty": 1.15,
        "no_repeat_ngram_size": 3,
    },
    "subjective": {
        "max_new_tokens": 400,
        "temperature": 0.4,
        "top_p": 0.8,
        "do_sample": True,
        "repetition_penalty": 1.2,
        "no_repeat_ngram_size": 4,
        "length_penalty": 1.05,
    },
}

OPTIMIZATION_CONFIG = {
    "intent_confidence_threshold": 0.7,
    "quality_threshold": 0.8,
    "korean_ratio_threshold": 0.85,
    "max_retry_attempts": 3,
    "template_preference": True,
    "adaptive_prompt": True,
    "mc_pattern_priority": True,
    "domain_specific_optimization": True,
    "institution_question_priority": True,
    "mc_context_weighting": True,
    "pkl_learning_enabled": True,
    "performance_tracking": True,
}

KOREAN_REQUIREMENTS = {
    "min_korean_ratio": 0.85,
    "max_english_ratio": 0.05,
    "min_length": 40,
    "max_length": 600,
    "repetition_tolerance": 2,
    "critical_repetition_limit": 3,
}

MEMORY_CONFIG = {
    "gc_frequency": 30,
    "save_interval": 50,
    "pkl_save_frequency": 10,
    "max_learning_records": {
        "successful_answers": 2000,
        "failed_answers": 1000,
        "question_patterns": 1500,
        "domain_templates": 500,
        "mc_patterns": 300,
        "performance_data": 1000,
    },
}

TIME_LIMITS = {
    "total_inference_minutes": 270,
    "warmup_timeout": 30,
    "single_question_timeout": 25,
    "generation_timeout": 15,
}

TEMPLATE_QUALITY_CRITERIA = {
    "length_range": (60, 500),
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
        r"(.{2,8})\s*\1\s*\1\s*\1",
        r"(.{1,3})\s*(\1\s*){4,}",
    ],
}

FILE_VALIDATION = {
    "required_files": ["test.csv", "sample_submission.csv"],
    "encoding": "utf-8-sig",
    "max_file_size_mb": 100,
}

# 성능 추적 설정
PERFORMANCE_TRACKING = {
    "track_accuracy": True,
    "track_response_time": True,
    "track_template_usage": True,
    "track_domain_distribution": True,
    "track_failure_reasons": True,
}

# 모니터링 설정
MONITORING_CONFIG = {
    "log_level": "INFO",
    "log_format": "%(asctime)s - %(levelname)s - %(message)s",
    "log_rotation": True,
    "max_log_size": 10 * 1024 * 1024,  # 10MB
}


def setup_environment():
    """환경 설정"""
    for key, value in OFFLINE_MODE.items():
        os.environ[key] = value


def get_device():
    """디바이스 설정"""
    if DEVICE_AUTO_SELECT:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cpu"


def ensure_directories():
    """디렉토리 생성"""
    PKL_DIR.mkdir(exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)


def validate_config():
    """설정 검증"""
    errors = []

    if not 0 <= KOREAN_REQUIREMENTS["min_korean_ratio"] <= 1:
        errors.append("min_korean_ratio는 0과 1 사이여야 합니다")

    if TIME_LIMITS["total_inference_minutes"] <= 0:
        errors.append("total_inference_minutes는 양수여야 합니다")

    if not 0 <= OPTIMIZATION_CONFIG["intent_confidence_threshold"] <= 1:
        errors.append("intent_confidence_threshold는 0과 1 사이여야 합니다")

    if errors:
        raise ValueError(f"설정 오류: {'; '.join(errors)}")

    return True


def initialize_system():
    """시스템 초기화"""
    setup_environment()
    ensure_directories()
    validate_config()

    if VERBOSE_MODE:
        print("시스템 설정 완료")
        print(f"기본 모델: {DEFAULT_MODEL_NAME}")
        print(f"디바이스: {get_device()}")


if __name__ != "__main__":
    try:
        initialize_system()
    except Exception as e:
        print(f"설정 초기화 중 오류: {e}")
        try:
            setup_environment()
            ensure_directories()
            print("기본 설정으로 시스템을 시작합니다.")
        except Exception as fallback_error:
            print(f"기본 설정 로드 실패: {fallback_error}")