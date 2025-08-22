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

# 생성 설정 - 주관식에 더 관대한 설정
GENERATION_CONFIG = {
    "multiple_choice": {
        "max_new_tokens": 15,
        "temperature": 0.3,
        "top_p": 0.8,
        "do_sample": True,
        "repetition_penalty": 1.1,  # 완화 (기존 1.2)
        "no_repeat_ngram_size": 2,  # 완화 (기존 3)
    },
    "subjective": {
        "max_new_tokens": 400,      # 증가 (기존 300)
        "temperature": 0.7,         # 증가 (기존 0.5)
        "top_p": 0.9,              # 증가 (기존 0.85)
        "do_sample": True,
        "repetition_penalty": 1.1,  # 완화 (기존 1.3)
        "no_repeat_ngram_size": 2,  # 완화 (기존 4)
        "length_penalty": 1.0,      # 완화 (기존 1.1)
    },
}

# === 성능 최적화 설정 - 기준 완화 ===
OPTIMIZATION_CONFIG = {
    "intent_confidence_threshold": 0.3,  # 완화 (기존 0.6)
    "quality_threshold": 0.4,           # 완화 (기존 0.7)
    "korean_ratio_threshold": 0.4,      # 완화 (기존 0.8)
    "max_retry_attempts": 3,            # 증가 (기존 2)
    "template_preference": True,
    "adaptive_prompt": True,
    "mc_pattern_priority": True,
    "domain_specific_optimization": True,
    "institution_question_priority": True,
    "mc_context_weighting": True,
    "enhanced_template_usage": True,     # 추가
    "relaxed_validation": True,          # 추가
    "improved_fallback": True,           # 추가
}

# === 한국어 처리 설정 - 기준 대폭 완화 ===
KOREAN_REQUIREMENTS = {
    "min_korean_ratio": 0.4,            # 완화 (기존 0.8)
    "max_english_ratio": 0.3,           # 완화 (기존 0.1)
    "min_length": 15,                   # 완화 (기존 30)
    "max_length": 600,                  # 증가 (기존 500)
    "repetition_tolerance": 5,          # 증가 (기존 2)
    "critical_repetition_limit": 15,    # 증가 (기존 3)
}

# === 메모리 관리 설정 ===
MEMORY_CONFIG = {
    "gc_frequency": 30,  # 줄임 (기존 50) - 더 자주 가비지 컬렉션
    "save_interval": 1000,
    "max_learning_records": {
        "successful_answers": 1500,      # 증가
        "failed_answers": 500,
        "quality_scores": 1500,          # 증가
        "choice_range_errors": 100,
        "repetitive_answers": 200,
    },
}

# === 시간 제한 설정 ===
TIME_LIMITS = {
    "total_inference_minutes": 300,     # 증가 (기존 270) - 5시간
    "warmup_timeout": 45,               # 증가 (기존 30)
    "single_question_timeout": 45,      # 증가 (기존 30)
    "generation_timeout": 30,           # 증가 (기존 20)
}

# === 템플릿 품질 평가 기준 - 기준 완화 ===
TEMPLATE_QUALITY_CRITERIA = {
    "length_range": (30, 500),          # 완화 (기존 50, 400)
    "korean_ratio_min": 0.7,            # 완화 (기존 0.9)
    "structure_keywords": ["법", "규정", "조치", "관리", "절차", "기준", "필요", "중요", "수행", "실시"],
    "intent_keywords": {
        "기관_묻기": ["위원회", "기관", "담당", "업무", "관련", "소관"],
        "특징_묻기": ["특징", "특성", "성질", "기능", "원리", "방식"],
        "지표_묻기": ["지표", "징후", "패턴", "탐지", "모니터링", "분석"],
        "방안_묻기": ["방안", "대책", "조치", "관리", "개선", "강화"],
        "절차_묻기": ["절차", "과정", "단계", "순서", "프로세스"],
        "조치_묻기": ["조치", "대응", "보안", "예방", "보완"],
    },
    "forbidden_patterns": [
        "갈취 묻는 말",
        "묻고 갈취",
        "갈취",
        r"(.{2,8})\s*\1\s*\1\s*\1\s*\1",  # 5회 이상 반복 (기존 4회에서 완화)
        r"(.{1,3})\s*(\1\s*){8,}",        # 8회 이상 반복 (기존 4회에서 완화)
    ],
}

# === 파일 검증 설정 ===
FILE_VALIDATION = {
    "required_files": ["test.csv", "sample_submission.csv"],
    "encoding": "utf-8-sig",
    "max_file_size_mb": 200,  # 증가 (기존 100)
}

# === JSON 설정 파일 경로 ===
JSON_CONFIG_FILES = {
    "knowledge_data": JSON_CONFIG_DIR / "knowledge_data.json",
    "model_config": JSON_CONFIG_DIR / "model_config.json",
    "processing_config": JSON_CONFIG_DIR / "processing_config.json",
}

# === 템플릿 활용 강화 설정 ===
TEMPLATE_ENHANCEMENT_CONFIG = {
    "max_templates_per_request": 7,     # 증가 (기존 5)
    "template_diversity_factor": 0.8,   # 다양성 증가
    "fallback_template_quality": 0.6,   # 폴백 템플릿 품질 기준
    "dynamic_template_generation": True,
    "cross_domain_template_usage": True,
    "intent_priority_mapping": {
        "기관_묻기": 1.0,
        "특징_묻기": 0.9,
        "방안_묻기": 0.9,
        "지표_묻기": 0.8,
        "절차_묻기": 0.7,
        "조치_묻기": 0.7,
    }
}

# === 답변 품질 검증 설정 - 기준 완화 ===
ANSWER_QUALITY_CONFIG = {
    "min_quality_score": 0.3,           # 완화 (기존 0.5)
    "korean_ratio_threshold": 0.4,      # 완화 (기존 0.6)
    "min_answer_length": 15,            # 완화 (기존 30)
    "max_answer_length": 600,           # 증가 (기존 500)
    "repetition_detection_threshold": 15, # 완화 (기존 8)
    "intent_match_weight": 0.3,         # 완화 (기존 0.5)
    "template_similarity_bonus": 0.2,   # 템플릿 유사성 보너스
    "fallback_acceptance_rate": 0.8,    # 폴백 수용률
}

# === 디버깅 및 로깅 설정 ===
DEBUG_CONFIG = {
    "verbose_template_usage": True,
    "log_intent_analysis": True,
    "track_generation_stats": True,
    "save_failed_answers": True,
    "monitor_quality_scores": True,
    "template_usage_analytics": True,
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

    # 템플릿 설정 검증
    if TEMPLATE_ENHANCEMENT_CONFIG["max_templates_per_request"] < 1:
        errors.append("max_templates_per_request는 1 이상이어야 합니다")

    if errors:
        raise ValueError(f"설정 오류: {'; '.join(errors)}")

    return True


# === 생성 설정 동적 조정 함수 ===
def adjust_generation_for_repetition_risk(question_type: str, risk_level: str = "medium"):
    """반복 위험에 따른 생성 설정 조정 - 더 관대한 기준"""
    risk_adjustments = {
        "low": {
            "repetition_penalty": 1.05,    # 더 완화
            "no_repeat_ngram_size": 2,     # 더 완화
            "temperature": 0.8,            # 더 창의적
        },
        "medium": {
            "repetition_penalty": 1.1,     # 완화 (기존 1.3)
            "no_repeat_ngram_size": 2,     # 완화 (기존 4)
            "temperature": 0.7,            # 증가 (기존 0.5)
        },
        "high": {
            "repetition_penalty": 1.2,     # 완화 (기존 1.5)
            "no_repeat_ngram_size": 3,     # 완화 (기존 5)
            "temperature": 0.6,            # 증가 (기존 0.4)
        },
    }

    if risk_level in risk_adjustments and question_type in GENERATION_CONFIG:
        # 기존 설정을 유지하면서 업데이트
        for key, value in risk_adjustments[risk_level].items():
            GENERATION_CONFIG[question_type][key] = value


# === 동적 설정 조정 함수 ===
def adjust_config_for_performance(performance_stats: dict):
    """성능 통계에 따른 동적 설정 조정"""
    if not performance_stats:
        return
    
    # 템플릿 사용률이 낮으면 기준 완화
    template_usage_rate = performance_stats.get("template_usage_rate", 0.5)
    if template_usage_rate < 0.3:
        OPTIMIZATION_CONFIG["intent_confidence_threshold"] = max(
            OPTIMIZATION_CONFIG["intent_confidence_threshold"] - 0.1, 0.1
        )
    
    # 품질 점수가 낮으면 기준 완화
    avg_quality_score = performance_stats.get("avg_quality_score", 0.5)
    if avg_quality_score < 0.4:
        ANSWER_QUALITY_CONFIG["min_quality_score"] = max(
            ANSWER_QUALITY_CONFIG["min_quality_score"] - 0.1, 0.2
        )
    
    # 폴백 사용률이 높으면 검증 완화
    fallback_rate = performance_stats.get("fallback_rate", 0.3)
    if fallback_rate > 0.5:
        KOREAN_REQUIREMENTS["min_korean_ratio"] = max(
            KOREAN_REQUIREMENTS["min_korean_ratio"] - 0.1, 0.3
        )


# === 품질 기준 완화 함수 ===
def relax_quality_standards():
    """품질 기준을 더 관대하게 조정"""
    KOREAN_REQUIREMENTS["min_korean_ratio"] = 0.3
    KOREAN_REQUIREMENTS["min_length"] = 10
    OPTIMIZATION_CONFIG["quality_threshold"] = 0.3
    OPTIMIZATION_CONFIG["intent_confidence_threshold"] = 0.2
    ANSWER_QUALITY_CONFIG["min_quality_score"] = 0.2
    ANSWER_QUALITY_CONFIG["korean_ratio_threshold"] = 0.3


# === 초기화 함수 ===
def initialize_system():
    """시스템 초기화"""
    setup_environment()
    ensure_directories()
    validate_config()

    if VERBOSE_MODE:
        print("시스템 설정 완료")
        print(f"기본 모델: {DEFAULT_MODEL_NAME}")
        print(f"디바이스: {get_device()}")
        print(f"오프라인 모드: {OFFLINE_MODE}")
        print(f"한국어 최소 비율: {KOREAN_REQUIREMENTS['min_korean_ratio']}")
        print(f"의도 신뢰도 임계값: {OPTIMIZATION_CONFIG['intent_confidence_threshold']}")
        print(f"최대 템플릿 수: {TEMPLATE_ENHANCEMENT_CONFIG['max_templates_per_request']}")


# === 설정 요약 출력 함수 ===
def print_config_summary():
    """현재 설정 요약 출력"""
    print("\n=== 현재 시스템 설정 요약 ===")
    print(f"모델: {DEFAULT_MODEL_NAME}")
    print(f"디바이스: {get_device()}")
    print(f"한국어 최소 비율: {KOREAN_REQUIREMENTS['min_korean_ratio']}")
    print(f"최소 답변 길이: {KOREAN_REQUIREMENTS['min_length']}")
    print(f"의도 신뢰도 임계값: {OPTIMIZATION_CONFIG['intent_confidence_threshold']}")
    print(f"품질 임계값: {ANSWER_QUALITY_CONFIG['min_quality_score']}")
    print(f"최대 템플릿 수: {TEMPLATE_ENHANCEMENT_CONFIG['max_templates_per_request']}")
    print(f"최대 토큰 수 (주관식): {GENERATION_CONFIG['subjective']['max_new_tokens']}")
    print(f"Temperature (주관식): {GENERATION_CONFIG['subjective']['temperature']}")
    print(f"반복 패널티 (주관식): {GENERATION_CONFIG['subjective']['repetition_penalty']}")
    print("="*50)


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