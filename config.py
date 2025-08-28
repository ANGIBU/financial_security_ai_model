# config.py

import os
import sys
from pathlib import Path

DEFAULT_MODEL_NAME = "upstage/SOLAR-10.7B-Instruct-v1.0"
DEVICE_AUTO_SELECT = True
VERBOSE_MODE = False

OFFLINE_MODE = {
    "TRANSFORMERS_OFFLINE": "1", 
    "HF_DATASETS_OFFLINE": "1",
    "HF_HUB_OFFLINE": "1",
    "TOKENIZERS_PARALLELISM": "false"
}

BASE_DIR = Path(__file__).parent.absolute()
PKL_DIR = BASE_DIR / "pkl"

DEFAULT_FILES = {
    "test_file": BASE_DIR / "test.csv",
    "submission_file": BASE_DIR / "sample_submission.csv",
    "output_file": BASE_DIR / "final_submission.csv",
    "test_output_file": BASE_DIR / "test_result.csv",
}

PKL_FILES = {
    "successful_answers": PKL_DIR / "successful_answers.pkl",
    "failed_answers": PKL_DIR / "failed_answers.pkl",
    "question_patterns": PKL_DIR / "question_patterns.pkl",
    "domain_templates": PKL_DIR / "domain_templates.pkl",
    "mc_patterns": PKL_DIR / "mc_patterns.pkl",
    "performance_data": PKL_DIR / "performance_data.pkl",
    "positional_patterns": PKL_DIR / "positional_patterns.pkl",
    "complexity_analysis": PKL_DIR / "complexity_analysis.pkl",
}

MODEL_CONFIG = {
    "torch_dtype": "bfloat16",
    "device_map": "auto",
    "trust_remote_code": True,
    "use_fast_tokenizer": True,
    "local_files_only": True,
}

GENERATION_CONFIG = {
    "multiple_choice": {
        "max_new_tokens": 15,
        "temperature": 0.03,
        "top_p": 0.4,
        "do_sample": True,
        "repetition_penalty": 1.02,
        "no_repeat_ngram_size": 2,
    },
    "subjective": {
        "max_new_tokens": 500,
        "temperature": 0.35,
        "top_p": 0.85,
        "do_sample": True,
        "repetition_penalty": 1.15,
        "no_repeat_ngram_size": 4,
        "length_penalty": 1.1,
    },
    "domain_specific": {
        "사이버보안": {
            "max_new_tokens": 550,
            "temperature": 0.3,
            "top_p": 0.8,
            "repetition_penalty": 1.2,
            "no_repeat_ngram_size": 5,
            "length_penalty": 1.15,
        },
        "전자금융": {
            "max_new_tokens": 450,
            "temperature": 0.25,
            "top_p": 0.75,
            "repetition_penalty": 1.15,
            "no_repeat_ngram_size": 4,
        },
        "개인정보보호": {
            "max_new_tokens": 450,
            "temperature": 0.25,
            "top_p": 0.75,
            "repetition_penalty": 1.15,
            "no_repeat_ngram_size": 4,
        },
        "정보보안": {
            "max_new_tokens": 400,
            "temperature": 0.3,
            "top_p": 0.8,
            "repetition_penalty": 1.15,
            "no_repeat_ngram_size": 4,
        },
        "위험관리": {
            "max_new_tokens": 350,
            "temperature": 0.2,
            "top_p": 0.7,
            "repetition_penalty": 1.1,
            "no_repeat_ngram_size": 3,
        },
        "금융투자": {
            "max_new_tokens": 350,
            "temperature": 0.2,
            "top_p": 0.7,
            "repetition_penalty": 1.1,
            "no_repeat_ngram_size": 3,
        },
        "정보통신": {
            "max_new_tokens": 300,
            "temperature": 0.25,
            "top_p": 0.75,
            "repetition_penalty": 1.1,
            "no_repeat_ngram_size": 3,
        },
        "기타": {
            "max_new_tokens": 450,
            "temperature": 0.25,
            "top_p": 0.75,
            "repetition_penalty": 1.15,
            "no_repeat_ngram_size": 4,
        }
    }
}

OPTIMIZATION_CONFIG = {
    "intent_confidence_threshold": 0.7,
    "quality_threshold": 0.8,
    "korean_ratio_threshold": 0.7,
    "max_retry_attempts": 4,
    "template_preference": True,
    "adaptive_prompt": True,
    "mc_pattern_priority": True,
    "domain_specific_optimization": True,
    "institution_question_priority": True,
    "mc_context_weighting": True,
    "pkl_learning_enabled": True,
    "few_shot_enabled": True,
    "domain_adaptive_generation": True,
    "answer_diversity_check": True,
    "quality_based_selection": True,
    "context_aware_prompting": True,
    "domain_weight_boost": 2.2,
    "pattern_matching_sensitivity": 0.75,
    "fallback_strategy": "domain_specific",
    "positional_analysis_enabled": True,
    "complexity_adaptive_generation": True,
    "late_stage_enhancement": True,
}

KOREAN_REQUIREMENTS = {
    "min_korean_ratio": 0.35,
    "max_english_ratio": 0.4,
    "min_length": 12,
    "max_length": 650,
    "repetition_tolerance": 3,
    "critical_repetition_limit": 4,
    "sentence_count_range": (1, 10),
    "professional_term_bonus": 0.15,
}

MEMORY_CONFIG = {
    "gc_frequency": 20,
    "save_interval": 35,
    "pkl_save_frequency": 6,
    "max_learning_records": {
        "successful_answers": 4000,
        "failed_answers": 1500,
        "question_patterns": 3000,
        "domain_templates": 1200,
        "mc_patterns": 800,
        "performance_data": 2500,
        "positional_patterns": 1000,
        "complexity_analysis": 800,
    },
    "memory_threshold": 80,
    "cache_size_limit": 600,
}

TIME_LIMITS = {
    "total_inference_minutes": 270,
    "warmup_timeout": 30,
    "single_question_timeout": 25,
    "generation_timeout": 15,
    "pattern_matching_timeout": 4,
}

TEMPLATE_QUALITY_CRITERIA = {
    "length_range": (40, 580),
    "korean_ratio_min": 0.8,
    "structure_keywords": ["법", "규정", "조치", "관리", "절차", "기준", "원칙", "체계"],
    "intent_keywords": {
        "기관_묻기": ["위원회", "기관", "담당", "업무", "소속", "역할"],
        "특징_묻기": ["특징", "특성", "성질", "기능", "역할", "목적"],
        "지표_묻기": ["지표", "징후", "패턴", "탐지", "신호", "증상"],
        "방안_묻기": ["방안", "대책", "조치", "관리", "계획", "전략"],
        "절차_묻기": ["절차", "과정", "단계", "순서", "방법", "절차"],
        "조치_묻기": ["조치", "대응", "보안", "예방", "대책", "방어"],
        "비율_묻기": ["비율", "얼마", "기준", "퍼센트", "비중", "할당"],
    },
    "forbidden_patterns": [
        r"(.{2,8})\s*\1\s*\1\s*\1",
        r"(.{1,3})\s*(\1\s*){5,}",
        r"(같은|동일한|유사한)\s+(.+?)\s+\2",
    ],
    "quality_boost_keywords": ["법령", "규정", "기준", "원칙", "체계", "정책"],
    "domain_specific_weights": {
        "사이버보안": 1.4,
        "전자금융": 1.3,
        "개인정보보호": 1.3,
        "정보보안": 1.2,
        "위험관리": 1.1,
        "금융투자": 1.1,
        "정보통신": 1.1,
        "기타": 1.2,
    }
}

FILE_VALIDATION = {
    "required_files": ["test.csv", "sample_submission.csv"],
    "encoding": "utf-8-sig",
    "max_file_size_mb": 150,
    "backup_encoding": "utf-8",
}

DOMAIN_WEIGHTS = {
    "사이버보안": {
        "priority_boost": 1.6,
        "pattern_sensitivity": 0.8,
        "context_depth": "high",
        "position_adjustment": 1.2,
    },
    "전자금융": {
        "priority_boost": 1.5,
        "pattern_sensitivity": 0.85,
        "context_depth": "high",
        "position_adjustment": 1.1,
    },
    "개인정보보호": {
        "priority_boost": 1.4,
        "pattern_sensitivity": 0.75,
        "context_depth": "medium",
        "position_adjustment": 1.0,
    },
    "정보보안": {
        "priority_boost": 1.3,
        "pattern_sensitivity": 0.75,
        "context_depth": "medium",
        "position_adjustment": 1.1,
    },
    "위험관리": {
        "priority_boost": 1.2,
        "pattern_sensitivity": 0.7,
        "context_depth": "medium",
        "position_adjustment": 1.0,
    },
    "금융투자": {
        "priority_boost": 1.2,
        "pattern_sensitivity": 0.7,
        "context_depth": "medium",
        "position_adjustment": 1.0,
    },
    "정보통신": {
        "priority_boost": 1.1,
        "pattern_sensitivity": 0.65,
        "context_depth": "low",
        "position_adjustment": 1.0,
    },
    "기타": {
        "priority_boost": 1.3,
        "pattern_sensitivity": 0.7,
        "context_depth": "medium",
        "position_adjustment": 1.1,
    }
}

POSITIONAL_ANALYSIS = {
    "early_stage": {"start": 0, "end": 100, "complexity_threshold": 0.6},
    "middle_stage": {"start": 101, "end": 300, "complexity_threshold": 0.55},
    "late_stage": {"start": 301, "end": 514, "complexity_threshold": 0.65},
    "position_weight_factors": {
        "early": 1.0,
        "middle": 1.1,
        "late": 1.3,
    }
}

PERFORMANCE_CONFIG = {
    "batch_processing": False,
    "concurrent_workers": 1,
    "cache_enabled": True,
    "preload_patterns": True,
    "memory_efficient_mode": True,
    "gradient_checkpointing": False,
    "mixed_precision": True,
    "position_aware_processing": True,
}

def setup_environment():
    for key, value in OFFLINE_MODE.items():
        os.environ[key] = value
    
    os.environ["CURL_CA_BUNDLE"] = ""
    os.environ["REQUESTS_CA_BUNDLE"] = ""

def get_device():
    if DEVICE_AUTO_SELECT:
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        except ImportError:
            return "cpu"
    return "cpu"

def get_generation_config(question_type: str, domain: str = None) -> dict:
    base_config = GENERATION_CONFIG.get(question_type, GENERATION_CONFIG["subjective"])
    
    if domain and domain in GENERATION_CONFIG["domain_specific"]:
        domain_config = GENERATION_CONFIG["domain_specific"][domain]
        merged_config = base_config.copy()
        merged_config.update(domain_config)
        return merged_config
    
    return base_config

def get_domain_weight(domain: str) -> dict:
    return DOMAIN_WEIGHTS.get(domain, DOMAIN_WEIGHTS["기타"])

def get_positional_config(question_number: int) -> dict:
    if question_number <= 100:
        stage = "early_stage"
        weight = POSITIONAL_ANALYSIS["position_weight_factors"]["early"]
    elif question_number <= 300:
        stage = "middle_stage"
        weight = POSITIONAL_ANALYSIS["position_weight_factors"]["middle"]
    else:
        stage = "late_stage"
        weight = POSITIONAL_ANALYSIS["position_weight_factors"]["late"]
    
    stage_config = POSITIONAL_ANALYSIS[stage].copy()
    stage_config["position_weight"] = weight
    return stage_config

def ensure_directories():
    try:
        PKL_DIR.mkdir(exist_ok=True)
        
        test_file = PKL_DIR / "test_write.tmp"
        try:
            with open(test_file, 'w') as f:
                f.write("test")
            test_file.unlink()
        except PermissionError:
            print(f"디렉토리 쓰기 권한 없음: {PKL_DIR}")
            
    except Exception as e:
        print(f"디렉토리 생성 오류: {e}")
        sys.exit(1)

def validate_config():
    errors = []

    if not 0 <= KOREAN_REQUIREMENTS["min_korean_ratio"] <= 1:
        errors.append("min_korean_ratio는 0과 1 사이여야 합니다")

    if TIME_LIMITS["total_inference_minutes"] <= 0:
        errors.append("total_inference_minutes는 양수여야 합니다")

    if not 0 <= OPTIMIZATION_CONFIG["intent_confidence_threshold"] <= 1:
        errors.append("intent_confidence_threshold는 0과 1 사이여야 합니다")
        
    if not 0 <= OPTIMIZATION_CONFIG["quality_threshold"] <= 1:
        errors.append("quality_threshold는 0과 1 사이여야 합니다")

    for config_type, config in GENERATION_CONFIG.items():
        if isinstance(config, dict):
            if "temperature" in config and not 0 <= config["temperature"] <= 2:
                errors.append(f"{config_type} temperature는 0과 2 사이여야 합니다")
            if "top_p" in config and not 0 <= config["top_p"] <= 1:
                errors.append(f"{config_type} top_p는 0과 1 사이여야 합니다")

    if errors:
        raise ValueError(f"설정 오류: {'; '.join(errors)}")

    return True

def get_optimal_config_for_accuracy():
    return {
        "temperature_boost": 0.05,
        "top_p_boost": 0.03,
        "repetition_penalty_boost": 0.03,
        "max_retry_attempts": OPTIMIZATION_CONFIG["max_retry_attempts"] + 1,
        "quality_threshold": OPTIMIZATION_CONFIG["quality_threshold"] + 0.1,
        "korean_ratio_threshold": KOREAN_REQUIREMENTS["min_korean_ratio"] + 0.15,
        "position_weight_multiplier": 1.2,
    }

def initialize_system():
    try:
        setup_environment()
        ensure_directories()
        validate_config()

        if VERBOSE_MODE:
            print("시스템 설정 완료")
            print(f"기본 모델: {DEFAULT_MODEL_NAME}")
            print(f"디바이스: {get_device()}")
            print(f"위치 인식 처리: 활성화")
    except Exception as e:
        print(f"시스템 초기화 실패: {e}")
        sys.exit(1)

if __name__ != "__main__":
    try:
        initialize_system()
    except Exception as e:
        print(f"설정 초기화 중 오류: {e}")
        try:
            setup_environment()
            ensure_directories()
            print("기본 설정으로 시스템을 시작합니다")
        except Exception as fallback_error:
            print(f"기본 설정 로드 실패: {fallback_error}")
            sys.exit(1)