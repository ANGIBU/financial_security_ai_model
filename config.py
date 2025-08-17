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
OFFLINE_MODE = {
    'TRANSFORMERS_OFFLINE': '1',
    'HF_DATASETS_OFFLINE': '1'
}

# === 디렉터리 설정 ===
BASE_DIR = Path(__file__).parent.absolute()
PKL_DIR = BASE_DIR / "pkl"
JSON_CONFIG_DIR = BASE_DIR / "configs"

# 기본 파일 경로
DEFAULT_FILES = {
    'test_file': './test.csv',
    'submission_file': './sample_submission.csv',
    'output_file': './final_submission.csv',
    'test_output_file': './test_result.csv'
}

# === 모델 설정 ===
MODEL_CONFIG = {
    'torch_dtype': 'bfloat16',
    'device_map': 'auto',
    'trust_remote_code': True,
    'use_fast_tokenizer': True
}

# 생성 설정 - 안정화된 버전
GENERATION_CONFIG = {
    'multiple_choice': {
        'max_new_tokens': 5,
        'temperature': 0.01,  # 매우 낮은 온도로 안정성 확보
        'top_p': 0.7,
        'do_sample': True,
        'repetition_penalty': 1.05,
        'pad_token_id': None,
        'eos_token_id': None
    },
    'subjective': {
        'max_new_tokens': 150,  # 적당한 길이로 제한
        'temperature': 0.1,   # 낮은 온도로 안정성 확보
        'top_p': 0.8,
        'do_sample': True,
        'repetition_penalty': 1.1,
        'pad_token_id': None,
        'eos_token_id': None
    }
}

# === 텍스트 정리 설정 - 안전성 우선 ===
TEXT_CLEANUP_CONFIG = {
    'remove_brackets': False,     # 과도한 정리 방지
    'remove_english': False,      # 과도한 정리 방지
    'fix_korean_typos': True,
    'normalize_spacing': True,
    'remove_special_chars': False, # 과도한 정리 방지
    'korean_only_mode': False     # 과도한 정리 방지
}

# 한국어 오타 수정 매핑 - 안전한 패턴만
KOREAN_TYPO_MAPPING = {
    '전자금윋': '전자금융',
    '캉터': '컴퓨터',
    '트래픁': '트래픽',
    '하웨어': '하드웨어',
    '네됴크': '네트워크',
    '메세지': '메시지'
}

# === 성능 최적화 설정 - 안정성 우선 ===
OPTIMIZATION_CONFIG = {
    'intent_confidence_threshold': 0.6,
    'quality_threshold': 0.6,      # 더 관대한 기준
    'korean_ratio_threshold': 0.7,  # 더 관대한 기준
    'max_retry_attempts': 3,
    'template_preference': True,
    'adaptive_prompt': True,
    'mc_pattern_priority': True,
    'domain_specific_optimization': True,
    'institution_question_priority': True,
    'mc_context_weighting': True,
    'text_cleanup_enabled': True,
    'typo_correction_enabled': True,
    'bracket_removal_enabled': False,  # 안전성 우선
    'english_removal_enabled': False   # 안전성 우선
}

# === 한국어 처리 설정 - 관대한 기준 ===
KOREAN_REQUIREMENTS = {
    'min_korean_ratio': 0.7,        # 더 관대한 기준
    'max_english_ratio': 0.1,       # 더 관대한 기준
    'min_length': 20,               # 더 관대한 기준
    'max_length': 300,              # 적당한 길이
    'allow_numbers': True,
    'allow_punctuation': True,
    'strict_korean_only': False     # 과도한 제한 해제
}

# === 신뢰도 평가 설정 ===
RELIABILITY_CONFIG = {
    'base_accuracy': 0.70,  # 더 현실적인 기준 정답률
    'confidence_factors': {
        'mc_success_weight': 0.4,      # 객관식 비중 증가
        'korean_compliance_weight': 0.2,
        'intent_match_weight': 0.2,    # 의도 일치 비중 감소
        'quality_weight': 0.2
    },
    'reliability_thresholds': {
        'excellent': 85.0,    # 더 현실적인 기준
        'good': 75.0,
        'acceptable': 65.0,
        'poor': 55.0
    }
}

# === 메모리 관리 설정 ===
MEMORY_CONFIG = {
    'gc_frequency': 50,
    'save_interval': 1000,
    'max_learning_records': {
        'successful_answers': 1000,
        'failed_answers': 500,
        'quality_scores': 1000,
        'choice_range_errors': 100
    }
}

# === 시간 제한 설정 ===
TIME_LIMITS = {
    'total_inference_minutes': 270,
    'warmup_timeout': 30,
    'single_question_timeout': 25
}

# === 진행률 표시 설정 ===
PROGRESS_CONFIG = {
    'bar_length': 50,
    'update_frequency': 1
}

# === 로깅 설정 ===
LOGGING_CONFIG = {
    'enable_stats_logging': True,
    'enable_error_logging': True,
    'log_processing_times': True,
    'log_quality_scores': True
}

# === 템플릿 품질 평가 기준 - 관대한 기준 ===
TEMPLATE_QUALITY_CRITERIA = {
    'length_range': (30, 300),      # 더 관대한 길이 기준
    'korean_ratio_min': 0.7,        # 더 관대한 한국어 비율
    'structure_keywords': ["법", "규정", "조치", "관리", "절차", "기준"],
    'intent_keywords': {
        "기관_묻기": ["위원회", "기관", "담당", "업무"],
        "특징_묻기": ["특징", "특성", "성질", "기능"],
        "지표_묻기": ["지표", "징후", "패턴", "탐지"],
        "방안_묻기": ["방안", "대책", "조치", "관리"],
        "절차_묻기": ["절차", "과정", "단계", "순서"],
        "조치_묻기": ["조치", "대응", "보안", "예방"]
    }
}

# === 텍스트 생성 안전성 설정 ===
TEXT_SAFETY_CONFIG = {
    'corruption_detection_enabled': True,
    'max_generation_attempts': 3,
    'safe_fallback_enabled': True,
    'corruption_patterns': [
        r'감추인', r'컨퍼머시', r'피-에', r'백-도어', r'키-로거', r'스크리너',
        r'채팅-클라언트', r'파일-업-', r'[가-힣]-[가-힣]{2,}',
        r'^[^가-힣]*$'  # 한국어가 전혀 없는 경우
    ],
    'min_korean_chars': 10,
    'quality_check_enabled': True
}

# === 테스트 설정 ===
TEST_CONFIG = {
    'test_sizes': {
        'quick': 10,
        'basic': 50,
        'detailed': 100,
        'full': 515
    },
    'default_test_size': 50
}

# === 파일 검증 설정 ===
FILE_VALIDATION = {
    'required_files': ['test.csv', 'sample_submission.csv'],
    'encoding': 'utf-8-sig',
    'max_file_size_mb': 100
}

# === 통계 설정 ===
STATS_CONFIG = {
    'track_answer_distribution': True,
    'track_domain_performance': True,
    'track_intent_accuracy': True,
    'track_template_effectiveness': True,
    'calculate_reliability_score': True
}

# === JSON 설정 파일 경로 ===
JSON_CONFIG_FILES = {
    'knowledge_data': JSON_CONFIG_DIR / 'knowledge_data.json',
    'model_config': JSON_CONFIG_DIR / 'model_config.json',
    'processing_config': JSON_CONFIG_DIR / 'processing_config.json'
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
    if not 0 <= KOREAN_REQUIREMENTS['min_korean_ratio'] <= 1:
        errors.append("min_korean_ratio는 0과 1 사이여야 합니다")
    
    # 시간 제한 검증
    if TIME_LIMITS['total_inference_minutes'] <= 0:
        errors.append("total_inference_minutes는 양수여야 합니다")
    
    # 최적화 임계값 검증
    if not 0 <= OPTIMIZATION_CONFIG['intent_confidence_threshold'] <= 1:
        errors.append("intent_confidence_threshold는 0과 1 사이여야 합니다")
    
    # 신뢰도 설정 검증
    if not 0 <= RELIABILITY_CONFIG['base_accuracy'] <= 1:
        errors.append("base_accuracy는 0과 1 사이여야 합니다")
    
    confidence_factors = RELIABILITY_CONFIG['confidence_factors']
    total_weight = sum(confidence_factors.values())
    if abs(total_weight - 1.0) > 0.01:
        errors.append(f"confidence_factors의 총 가중치는 1.0이어야 합니다 (현재: {total_weight})")
    
    # 생성 설정 검증
    for config_type, config in GENERATION_CONFIG.items():
        if config['temperature'] <= 0:
            errors.append(f"{config_type} temperature는 양수여야 합니다")
        if not 0 < config['top_p'] <= 1:
            errors.append(f"{config_type} top_p는 0과 1 사이여야 합니다")
    
    if errors:
        raise ValueError(f"설정 오류: {'; '.join(errors)}")
    
    return True

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
        print(f"텍스트 안전성 검사: {TEXT_SAFETY_CONFIG['corruption_detection_enabled']}")

# === 안전성 검증 함수 ===
def get_safe_generation_config(question_type: str) -> dict:
    """안전한 생성 설정 반환"""
    config = GENERATION_CONFIG[question_type].copy()
    
    # 추가 안전 장치
    if question_type == "subjective":
        config['temperature'] = min(config['temperature'], 0.2)  # 최대 온도 제한
        config['max_new_tokens'] = min(config['max_new_tokens'], 200)  # 최대 길이 제한
    
    return config

def check_text_safety(text: str) -> bool:
    """텍스트 안전성 검사"""
    if not TEXT_SAFETY_CONFIG['corruption_detection_enabled']:
        return True
    
    import re
    
    # 깨진 텍스트 패턴 검사
    for pattern in TEXT_SAFETY_CONFIG['corruption_patterns']:
        if re.search(pattern, text):
            return False
    
    # 한국어 문자 수 검사
    korean_chars = len(re.findall(r'[가-힣]', text))
    if korean_chars < TEXT_SAFETY_CONFIG['min_korean_chars']:
        return False
    
    return True

# 자동 초기화 (모듈 import 시 실행)
if __name__ != "__main__":
    try:
        initialize_system()
    except Exception as e:
        print(f"설정 초기화 중 오류: {e}")