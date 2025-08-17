# config.py

"""
금융보안 AI 시스템 설정 파일
- 모델 설정
- 시스템 환경 설정
- 성능 최적화 설정
- 평가 기준 설정
- LLM 생성 중심 최적화
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

# 생성 설정 - LLM 생성 최적화
GENERATION_CONFIG = {
    'multiple_choice': {
        'max_new_tokens': 5,               # 객관식용 적정 길이
        'temperature': 0.15,               # 적절한 다양성과 일관성
        'top_p': 0.8,                      # 충분한 선택 폭
        'top_k': 50,                       # 적절한 후보 범위
        'do_sample': True,
        'repetition_penalty': 1.03,        # 적절한 반복 방지
        'pad_token_id': None,
        'eos_token_id': None,
        'no_repeat_ngram_size': 2,
        'early_stopping': True
    },
    'subjective': {
        'max_new_tokens': 150,             # 주관식용 충분한 길이
        'temperature': 0.25,               # LLM 창의성과 일관성 균형
        'top_p': 0.9,                      # 풍부한 표현력
        'top_k': 60,                       # 적절한 어휘 선택 범위
        'do_sample': True,
        'repetition_penalty': 1.08,        # 반복 방지 강화
        'pad_token_id': None,
        'eos_token_id': None,
        'no_repeat_ngram_size': 3,
        'length_penalty': 1.1,             # 적절한 길이 유도
        'early_stopping': True
    }
}

# === 텍스트 정리 설정 - LLM 생성 결과 후처리 중심 ===
TEXT_CLEANUP_CONFIG = {
    'remove_brackets': False,           # LLM 생성 결과 보존
    'remove_english': False,            # 전문용어 보존
    'fix_korean_typos': True,           # 오타 수정은 유지
    'normalize_spacing': True,          # 띄어쓰기 정규화
    'remove_special_chars': False,      # 특수문자 보존
    'korean_only_mode': False,          # 혼재 허용
    'enhance_llm_output': True,         # LLM 출력 개선 활성화
    'post_process_generation': True     # 생성 후처리 활성화
}

# 한국어 오타 수정 매핑 - LLM 생성 결과 개선용
KOREAN_TYPO_MAPPING = {
    '전자금윋': '전자금융',
    '캉터': '컴퓨터',
    '트래픁': '트래픽',
    '하웨어': '하드웨어',
    '네됴크': '네트워크',
    '메세지': '메시지',
    '보안조최': '보안조치',
    '관리방안': '관리 방안',
    '데이타': '데이터',
    '시스탬': '시스템',
    '프로그럼': '프로그램'
}

# === 성능 최적화 설정 - LLM 생성 우선 ===
OPTIMIZATION_CONFIG = {
    'intent_confidence_threshold': 0.4,        # LLM 생성 활용 증대
    'quality_threshold': 0.6,                  # 적절한 품질 기준
    'korean_ratio_threshold': 0.6,             # 한국어 비율 기준
    'max_retry_attempts': 3,                   # LLM 재시도 증가
    'template_preference': False,              # 템플릿 의존도 감소
    'llm_generation_priority': True,           # LLM 생성 우선
    'adaptive_prompt': True,                   # 적응적 프롬프트
    'mc_pattern_priority': True,               # 객관식 패턴 우선
    'domain_specific_optimization': True,       # 도메인 특화 최적화
    'institution_question_priority': True,     # 기관 질문 우선 처리
    'mc_context_weighting': True,              # 객관식 컨텍스트 가중
    'text_cleanup_enabled': True,              # 텍스트 정리 활성화
    'typo_correction_enabled': True,           # 오타 수정 활성화
    'llm_guided_generation': True,             # LLM 가이드 생성
    'knowledge_hint_integration': True,        # 지식 힌트 통합
    'generation_quality_enhancement': True     # 생성 품질 개선
}

# === 한국어 처리 설정 - LLM 생성 결과 중심 ===
KOREAN_REQUIREMENTS = {
    'min_korean_ratio': 0.6,            # LLM 생성 결과 허용 기준
    'max_english_ratio': 0.25,          # 전문용어 허용
    'min_length': 15,                   # 최소 길이
    'max_length': 400,                  # 최대 길이
    'allow_numbers': True,
    'allow_punctuation': True,
    'strict_korean_only': False,        # LLM 생성 다양성 허용
    'llm_output_tolerance': True,       # LLM 출력 관대 평가
    'generation_focused_validation': True  # 생성 중심 검증
}

# === 신뢰도 평가 설정 - LLM 생성 품질 중심 ===
RELIABILITY_CONFIG = {
    'base_accuracy': 0.72,              # LLM 생성 기반 기준
    'confidence_factors': {
        'mc_success_weight': 0.35,      # 객관식 비중
        'korean_compliance_weight': 0.25,  # 한국어 준수
        'llm_generation_weight': 0.25,  # LLM 생성 품질
        'intent_match_weight': 0.15     # 의도 일치
    },
    'llm_quality_bonus': {
        'original_generation': 0.1,     # 독창적 생성 보너스
        'knowledge_integration': 0.05,  # 지식 통합 보너스
        'coherence_bonus': 0.05        # 일관성 보너스
    },
    'reliability_thresholds': {
        'excellent': 85.0,
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
        'successful_answers': 1500,     # LLM 생성 성공 기록 증가
        'failed_answers': 500,
        'quality_scores': 1500,
        'choice_range_errors': 100,
        'generation_patterns': 1000     # 생성 패턴 기록
    }
}

# === 시간 제한 설정 ===
TIME_LIMITS = {
    'total_inference_minutes': 270,
    'warmup_timeout': 30,
    'single_question_timeout': 30,     # LLM 생성 시간 여유
    'generation_timeout': 25           # 생성 전용 시간
}

# === 진행률 표시 설정 ===
PROGRESS_CONFIG = {
    'bar_length': 50,
    'update_frequency': 1,
    'show_generation_stats': True      # 생성 통계 표시
}

# === 로깅 설정 ===
LOGGING_CONFIG = {
    'enable_stats_logging': True,
    'enable_error_logging': True,
    'log_processing_times': True,
    'log_quality_scores': True,
    'log_generation_patterns': True,   # 생성 패턴 로깅
    'log_llm_performance': True        # LLM 성능 로깅
}

# === 템플릿 품질 평가 기준 ===
TEMPLATE_QUALITY_CRITERIA = {
    'length_range': (25, 300),          # 템플릿 길이 기준
    'korean_ratio_min': 0.6,            # 한국어 비율
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

# === LLM 생성 품질 평가 기준 ===
LLM_QUALITY_CRITERIA = {
    'length_range': (20, 350),          # 적절한 길이 범위
    'korean_ratio_min': 0.6,            # 한국어 비율
    'coherence_keywords': ["법", "규정", "조치", "관리", "절차", "기준", "체계"],
    'domain_relevance_check': True,     # 도메인 관련성 검사
    'repetition_penalty_threshold': 0.3, # 반복 허용 임계값
    'intent_matching_bonus': 0.1,      # 의도 일치 보너스
    'knowledge_integration_bonus': 0.05, # 지식 통합 보너스
    'generation_quality_weights': {
        'fluency': 0.3,                 # 유창성
        'relevance': 0.4,               # 관련성
        'completeness': 0.3             # 완전성
    }
}

# === 텍스트 생성 안전성 설정 - LLM 중심 ===
TEXT_SAFETY_CONFIG = {
    'corruption_detection_enabled': True,
    'max_generation_attempts': 3,       # LLM 재시도 증가
    'safe_fallback_enabled': True,
    'llm_output_validation': True,      # LLM 출력 검증
    'generation_quality_check': True,   # 생성 품질 검사
    'corruption_patterns': [
        r'감추인', r'컨퍼머시', r'피-에', r'백-도어', r'키-로거', r'스크리너',
        r'채팅-클라언트', r'파일-업-', r'[가-힣]-[가-힣]{2,}',
        r'^[^가-힣]*$',  # 한국어가 전혀 없는 경우
        r'[가-힣]{1,2}-[영어단어]',  # 한글-영어 패턴
        r'[가-힣]+[A-Za-z]+[가-힣]+',  # 한영 혼재
        r'[\u0000-\u001F]',  # 제어 문자
        r'감.*추.*인', r'컨.*퍼.*머.*시', r'피.*에', r'백.*도.*어'
    ],
    'min_korean_chars': 8,
    'quality_check_enabled': True,
    'llm_specific_validation': True,    # LLM 특화 검증
    'generation_coherence_check': True  # 생성 일관성 검사
}

# === 테스트 설정 ===
TEST_CONFIG = {
    'test_sizes': {
        'quick': 10,
        'basic': 50,
        'detailed': 100,
        'full': 515
    },
    'default_test_size': 50,
    'generation_test_enabled': True    # 생성 테스트 활성화
}

# === 파일 검증 설정 ===
FILE_VALIDATION = {
    'required_files': ['test.csv', 'sample_submission.csv'],
    'encoding': 'utf-8-sig',
    'max_file_size_mb': 100
}

# === 통계 설정 - LLM 생성 중심 ===
STATS_CONFIG = {
    'track_answer_distribution': True,
    'track_domain_performance': True,
    'track_intent_accuracy': True,
    'track_template_effectiveness': False,  # 템플릿 의존도 감소
    'track_llm_generation_quality': True,   # LLM 생성 품질 추적
    'track_generation_patterns': True,      # 생성 패턴 추적
    'calculate_reliability_score': True,
    'monitor_generation_performance': True  # 생성 성능 모니터링
}

# === JSON 설정 파일 경로 ===
JSON_CONFIG_FILES = {
    'knowledge_data': JSON_CONFIG_DIR / 'knowledge_data.json',
    'model_config': JSON_CONFIG_DIR / 'model_config.json',
    'processing_config': JSON_CONFIG_DIR / 'processing_config.json'
}

# === 토크나이저 설정 - LLM 생성 최적화 ===
TOKENIZER_SAFETY_CONFIG = {
    'add_special_tokens': True,
    'return_tensors': "pt",
    'truncation': True,
    'max_length': 1200,              # LLM 생성용 충분한 길이
    'padding': False,
    'clean_up_tokenization_spaces': True,
    'generation_optimized': True     # 생성 최적화
}

# === LLM 프롬프트 최적화 설정 ===
PROMPT_OPTIMIZATION_CONFIG = {
    'enable_context_hints': True,      # 컨텍스트 힌트 활성화
    'enable_domain_guidance': True,    # 도메인 가이드 활성화
    'enable_intent_prompting': True,   # 의도 기반 프롬프팅
    'enable_quality_instructions': True, # 품질 지시사항
    'dynamic_prompt_adjustment': True,  # 동적 프롬프트 조정
    'knowledge_integration_prompting': True, # 지식 통합 프롬프팅
    'prompt_templates': {
        'mc_base': "다음 객관식 문제를 분석하여 정답을 선택하세요.",
        'subj_base': "다음 질문에 대해 한국어로만 정확한 답변을 작성하세요.",
        'institution': "구체적인 기관명과 역할을 명시하여 답변하세요.",
        'feature': "주요 특징과 특성을 체계적으로 설명하세요.",
        'indicator': "탐지 지표와 징후를 구체적으로 나열하세요."
    }
}

# === 환경 변수 설정 함수 ===
def setup_environment():
    """환경 변수 설정"""
    for key, value in OFFLINE_MODE.items():
        os.environ[key] = value
    
    # LLM 생성 최적화 설정
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'

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
    
    # 기본 JSON 설정 파일이 없으면 생성
    if not JSON_CONFIG_FILES['knowledge_data'].exists():
        print("기본 knowledge_data.json 파일을 생성합니다...")
        # 기본 설정 파일은 별도로 제공되어야 함
    
    if not JSON_CONFIG_FILES['model_config'].exists():
        print("기본 model_config.json 파일을 생성합니다...")
        # 기본 설정 파일은 별도로 제공되어야 함
    
    if not JSON_CONFIG_FILES['processing_config'].exists():
        print("기본 processing_config.json 파일을 생성합니다...")
        # 기본 설정 파일은 별도로 제공되어야 함

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
    
    # LLM 품질 기준 검증
    quality_weights = LLM_QUALITY_CRITERIA['generation_quality_weights']
    quality_total = sum(quality_weights.values())
    if abs(quality_total - 1.0) > 0.01:
        errors.append(f"generation_quality_weights의 총합은 1.0이어야 합니다 (현재: {quality_total})")
    
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
        print(f"LLM 생성 우선: {OPTIMIZATION_CONFIG['llm_generation_priority']}")
        print(f"지식 힌트 통합: {OPTIMIZATION_CONFIG['knowledge_hint_integration']}")

# === LLM 생성 설정 함수 ===
def get_optimized_generation_config(question_type: str, domain: str = "일반") -> dict:
    """최적화된 생성 설정 반환"""
    base_config = GENERATION_CONFIG[question_type].copy()
    
    # 도메인별 최적화
    domain_adjustments = {
        "사이버보안": {
            "temperature": base_config["temperature"] * 0.9,  # 기술적 정확성 중시
            "top_k": min(base_config.get("top_k", 50), 40)
        },
        "전자금융": {
            "temperature": base_config["temperature"] * 0.8,  # 법적 정확성 중시
            "repetition_penalty": base_config["repetition_penalty"] * 1.05
        },
        "개인정보보호": {
            "temperature": base_config["temperature"] * 0.85, # 법적 정확성 중시
            "top_p": min(base_config["top_p"], 0.85)
        }
    }
    
    if domain in domain_adjustments:
        base_config.update(domain_adjustments[domain])
    
    return base_config

def get_llm_prompt_template(intent_type: str, domain: str = "일반") -> str:
    """LLM 프롬프트 템플릿 반환"""
    templates = PROMPT_OPTIMIZATION_CONFIG['prompt_templates']
    
    if intent_type in ["기관_묻기"]:
        return templates['institution']
    elif intent_type in ["특징_묻기"]:
        return templates['feature']
    elif intent_type in ["지표_묻기"]:
        return templates['indicator']
    else:
        return templates['subj_base']

# === 안전성 검증 함수 ===
def check_text_safety(text: str) -> bool:
    """텍스트 안전성 검사 - LLM 생성 결과 중심"""
    if not TEXT_SAFETY_CONFIG['corruption_detection_enabled']:
        return True
    
    import re
    
    # 빈 텍스트 검사
    if not text or len(text.strip()) == 0:
        return False
    
    # LLM 생성 특화 검사
    if TEXT_SAFETY_CONFIG['llm_output_validation']:
        # 과도한 반복 검사
        words = text.split()
        if len(words) > 10:
            word_counts = {}
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
            max_repeat = max(word_counts.values()) if word_counts else 0
            if max_repeat > len(words) * 0.4:  # 40% 이상 반복
                return False
    
    # 깨진 텍스트 패턴 검사
    for pattern in TEXT_SAFETY_CONFIG['corruption_patterns']:
        if re.search(pattern, text):
            return False
    
    # 한국어 문자 수 검사
    korean_chars = len(re.findall(r'[가-힣]', text))
    if korean_chars < TEXT_SAFETY_CONFIG['min_korean_chars']:
        return False
    
    # 기본 텍스트 품질 검사
    total_chars = len(re.sub(r'[^\w가-힣]', '', text))
    if total_chars > 0:
        korean_ratio = korean_chars / total_chars
        if korean_ratio < 0.5:  # 50% 이상 한국어여야 함
            return False
    
    # LLM 생성 일관성 검사
    if TEXT_SAFETY_CONFIG['generation_coherence_check']:
        # 불완전한 문장 검사
        if text.endswith(('이', '가', '를', '의', '에', '와', '과')):
            return False
    
    return True

def calculate_llm_quality_score(text: str, intent_match: float, domain_relevance: float) -> float:
    """LLM 생성 품질 점수 계산"""
    weights = LLM_QUALITY_CRITERIA['generation_quality_weights']
    
    # 유창성 점수 (한국어 자연스러움)
    fluency = 0.8  # 기본값
    if check_text_safety(text):
        fluency = min(1.0, fluency + 0.2)
    
    # 관련성 점수
    relevance = domain_relevance
    
    # 완전성 점수 (적절한 길이와 구조)
    length_score = 0.5
    if LLM_QUALITY_CRITERIA['length_range'][0] <= len(text) <= LLM_QUALITY_CRITERIA['length_range'][1]:
        length_score = 1.0
    
    completeness = (length_score + intent_match) / 2
    
    # 가중 평균 계산
    quality_score = (
        fluency * weights['fluency'] +
        relevance * weights['relevance'] + 
        completeness * weights['completeness']
    )
    
    # 보너스 적용
    if intent_match > 0.8:
        quality_score += LLM_QUALITY_CRITERIA['intent_matching_bonus']
    
    return min(quality_score, 1.0)

# 자동 초기화 (모듈 import 시 실행)
if __name__ != "__main__":
    try:
        initialize_system()
    except Exception as e:
        print(f"설정 초기화 중 오류: {e}")