# config.py

"""
시스템 설정 및 최적화 매개변수
- 모델 설정
- Self-Consistency 설정
- 한국어 검증 기준
- 성능 최적화 매개변수
"""

# 모델 설정
MODEL_CONFIG = {
    "model_name": "upstage/SOLAR-10.7B-Instruct-v1.0",
    "device": "auto",  # cuda/cpu/auto
    "torch_dtype": "bfloat16",
    "trust_remote_code": True,
    "use_fast_tokenizer": True
}

# Self-Consistency 설정 (성능 최적화)
SELF_CONSISTENCY_CONFIG = {
    "enabled": True,
    "num_samples": 3,  # 생성할 샘플 수
    "temperature_range": [0.1, 0.3, 0.5],  # 다양한 온도값
    "max_agreement_threshold": 0.7,  # 합의 임계값
    "use_weighted_voting": True,  # 신뢰도 기반 가중 투표
    "fallback_on_failure": True   # 실패시 폴백 사용
}

# 생성 매개변수 최적화
GENERATION_CONFIG = {
    "multiple_choice": {
        "max_new_tokens": 20,
        "temperature": 0.2,  # 일관성을 위해 낮은 온도
        "top_p": 0.85,
        "do_sample": True,
        "pad_token_id": None,  # 토크나이저에서 설정
        "eos_token_id": None   # 토크나이저에서 설정
    },
    "subjective": {
        "max_new_tokens": 300,
        "temperature": 0.4,   # 창의성과 일관성의 균형
        "top_p": 0.9,
        "do_sample": True,
        "pad_token_id": None,
        "eos_token_id": None
    }
}

# 한국어 검증 기준 (강화)
KOREAN_VALIDATION_CONFIG = {
    "min_korean_ratio": 0.8,      # 최소 한국어 비율 80%
    "max_english_ratio": 0.1,     # 최대 영어 비율 10%
    "min_length": 30,             # 최소 길이
    "max_length": 500,            # 최대 길이
    "min_korean_chars": 20,       # 최소 한국어 문자 수
    "require_korean_ending": True,  # 한국어 어미 필수
    "forbidden_patterns": [       # 금지된 패턴
        r'[a-zA-Z]{3,}',         # 3글자 이상 영어 단어
        r'http[s]?://',          # URL
        r'www\.',                # 웹사이트
        r'@[a-zA-Z]',           # 이메일
    ]
}

# 질문 의도 분석 설정
INTENT_ANALYSIS_CONFIG = {
    "enabled": True,
    "confidence_threshold": 0.5,   # 의도 인식 신뢰도 임계값
    "use_morphological_analysis": False,  # 형태소 분석 (선택적)
    "intent_categories": [
        "기관_요청",
        "특징_분석", 
        "지표_나열",
        "절차_설명",
        "법령_해석"
    ],
    "require_intent_match": True,   # 의도 일치 필수
    "fallback_on_mismatch": True   # 불일치시 폴백 사용
}

# 도메인 분류 설정
DOMAIN_CONFIG = {
    "domains": [
        "개인정보보호",
        "전자금융",
        "사이버보안",
        "정보보안",
        "금융투자",
        "위험관리"
    ],
    "multi_domain_support": True,   # 다중 도메인 지원
    "domain_specific_templates": True,  # 도메인별 템플릿 사용
    "cross_domain_fallback": True  # 도메인간 폴백 지원
}

# 답변 품질 평가 기준
QUALITY_ASSESSMENT_CONFIG = {
    "korean_ratio_weight": 0.25,      # 한국어 비율 가중치
    "length_appropriateness_weight": 0.20,  # 길이 적절성 가중치
    "sentence_structure_weight": 0.15,      # 문장 구조 가중치
    "intent_match_weight": 0.25,           # 의도 일치성 가중치
    "professionalism_weight": 0.15,        # 전문성 가중치
    "min_quality_threshold": 0.6,          # 최소 품질 임계값
    "use_comprehensive_scoring": True      # 종합 점수 시스템 사용
}

# 객관식 답변 최적화 설정
MULTIPLE_CHOICE_CONFIG = {
    "use_context_analysis": True,     # 컨텍스트 분석 사용
    "balanced_distribution": True,    # 균등 분포 유지
    "negative_question_bias": {       # 부정형 질문 편향
        3: [1, 1, 2],
        4: [1, 1, 2, 3], 
        5: [1, 1, 2, 3, 4]
    },
    "positive_question_bias": {       # 긍정형 질문 편향
        3: [3, 2, 1],
        4: [3, 3, 2, 1],
        5: [3, 3, 2, 1, 1]
    },
    "range_validation": True,         # 범위 검증 필수
    "fallback_strategy": "context_based"  # 폴백 전략
}

# 성능 최적화 설정
PERFORMANCE_CONFIG = {
    "memory_management": {
        "gc_frequency": 50,           # 가비지 컬렉션 주기
        "model_offload": False,       # 모델 오프로드 (메모리 부족시)
        "cache_size_limit": 1000      # 캐시 크기 제한
    },
    "processing": {
        "batch_processing": False,     # 배치 처리 (현재 미지원)
        "progress_update_frequency": 1, # 진행률 업데이트 주기
        "timeout_per_question": 60     # 질문당 타임아웃 (초)
    },
    "logging": {
        "detailed_stats": True,        # 상세 통계 수집
        "save_learning_data": True,    # 학습 데이터 저장
        "error_tracking": True         # 오류 추적
    }
}

# 시스템 안정성 설정
RELIABILITY_CONFIG = {
    "max_retries": 3,                 # 최대 재시도 횟수
    "fallback_on_error": True,        # 오류시 폴백 사용
    "validation_strictness": "high",  # 검증 엄격도 (low/medium/high)
    "safety_checks": True,            # 안전성 검사 활성화
    "graceful_degradation": True      # 우아한 성능 저하
}

# 한국 금융기관 데이터베이스 설정
INSTITUTION_CONFIG = {
    "include_all_institutions": True,  # 모든 기관 포함
    "update_frequency": "monthly",     # 업데이트 주기
    "verification_required": True,     # 검증 필수
    "detailed_info": True            # 상세 정보 포함
}

# 실험적 기능 설정 (대회 규칙 준수)
EXPERIMENTAL_CONFIG = {
    "advanced_prompting": True,       # 고급 프롬프팅 기법
    "chain_of_thought": True,         # Chain-of-Thought 추론
    "few_shot_examples": False,       # Few-shot 예시 (데이터 부족으로 비활성화)
    "response_calibration": True,     # 응답 보정
    "uncertainty_estimation": True    # 불확실성 추정
}

# 대회 규칙 준수 설정
COMPETITION_COMPLIANCE = {
    "single_llm_only": True,          # 단일 LLM만 사용
    "local_execution_only": True,     # 로컬 실행만
    "no_external_api": True,          # 외부 API 금지
    "korean_only_responses": True,    # 한국어 전용 응답
    "offline_mode": True,             # 오프라인 모드
    "no_internet_dependency": True    # 인터넷 의존성 금지
}

# 파일 입출력 설정
FILE_CONFIG = {
    "encoding": "utf-8-sig",          # 파일 인코딩
    "backup_results": False,          # 결과 백업 (권한 문제 방지)
    "overwrite_existing": True,       # 기존 파일 덮어쓰기
    "create_directories": True        # 디렉토리 자동 생성
}

def get_optimized_config():
    """최적화된 설정 반환"""
    return {
        "model": MODEL_CONFIG,
        "self_consistency": SELF_CONSISTENCY_CONFIG,
        "generation": GENERATION_CONFIG,
        "korean_validation": KOREAN_VALIDATION_CONFIG,
        "intent_analysis": INTENT_ANALYSIS_CONFIG,
        "domain": DOMAIN_CONFIG,
        "quality_assessment": QUALITY_ASSESSMENT_CONFIG,
        "multiple_choice": MULTIPLE_CHOICE_CONFIG,
        "performance": PERFORMANCE_CONFIG,
        "reliability": RELIABILITY_CONFIG,
        "institution": INSTITUTION_CONFIG,
        "experimental": EXPERIMENTAL_CONFIG,
        "compliance": COMPETITION_COMPLIANCE,
        "file": FILE_CONFIG
    }

def validate_config():
    """설정 유효성 검증"""
    config = get_optimized_config()
    
    # 대회 규칙 준수 확인
    if not config["compliance"]["single_llm_only"]:
        raise ValueError("대회 규칙 위반: 단일 LLM만 사용 가능")
    
    if not config["compliance"]["korean_only_responses"]:
        raise ValueError("대회 규칙 위반: 한국어 전용 응답 필수")
    
    if not config["compliance"]["offline_mode"]:
        raise ValueError("대회 규칙 위반: 오프라인 모드 필수")
    
    # 한국어 검증 기준 확인
    korean_config = config["korean_validation"]
    if korean_config["min_korean_ratio"] < 0.5:
        raise ValueError("한국어 비율이 너무 낮음")
    
    # Self-Consistency 설정 확인
    sc_config = config["self_consistency"]
    if sc_config["num_samples"] < 1:
        raise ValueError("Self-Consistency 샘플 수는 1 이상이어야 함")
    
    return True

if __name__ == "__main__":
    # 설정 검증
    try:
        validate_config()
        print("설정 검증 완료")
        print("모든 설정이 대회 규칙을 준수합니다.")
    except ValueError as e:
        print(f"설정 오류: {e}")