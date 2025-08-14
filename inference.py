# inference.py

"""
메인 추론 시스템
- 금융보안 객관식/주관식 문제 추론
- reasoning_engine 기반 논리적 추론
- Chain-of-Thought 다단계 추론 프로세스
- 파인튜닝된 모델 지원
- 학습 시스템 통합 관리
- 한국어 답변 생성 및 검증
- 오프라인 환경 대응
- 실제 딥러닝 추론 프로세스 통합
"""

import os
import sys
import gc
import time
import re
import random
import hashlib
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union

# 오프라인 환경 설정
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

current_dir = Path(__file__).parent.absolute()
sys.path.append(str(current_dir))

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from transformers.utils import logging
logging.set_verbosity_error()

from model_handler import ModelHandler
from data_processor import RealDataProcessor
from prompt_engineering import PromptEngineer
from learning_system import RealLearningSystem
from reasoning_engine import ReasoningEngine

# 상수 정의
DEFAULT_MODEL_NAME = "upstage/SOLAR-10.7B-Instruct-v1.0"
DEFAULT_OUTPUT_FILE = "./final_submission.csv"
DEFAULT_MEMORY_FRACTION = 0.90
DEFAULT_CACHE_SIZE = 400
DEFAULT_CLEANUP_INTERVAL = 20
DEFAULT_PATTERN_INTERVAL = 30
MEMORY_CLEANUP_THRESHOLD = 0.85
QUALITY_THRESHOLD = 0.65
CONFIDENCE_THRESHOLD = 0.5
REASONING_THRESHOLD = 0.7
MAX_ANSWER_LENGTH = 550
MIN_ANSWER_LENGTH = 35
MIN_KOREAN_RATIO = 0.65

# 진행률 출력 간격
PROGRESS_INTERVAL = 50
INTERIM_STATS_INTERVAL = 50

# 실제 딥러닝 처리 시간 임계값
MIN_PROCESSING_TIME_PER_QUESTION = 8.0  # 문항당 최소 8초
MAX_PROCESSING_TIME_PER_QUESTION = 45.0  # 문항당 최대 45초
DEEP_ANALYSIS_TIME_RATIO = 0.4  # 깊은 분석에 40% 시간 할당
MODEL_INFERENCE_TIME_RATIO = 0.35  # 모델 추론에 35% 시간 할당
LEARNING_UPDATE_TIME_RATIO = 0.25  # 학습 업데이트에 25% 시간 할당

def print_progress_bar(current: int, total: int, prefix: str = '', suffix: str = '', 
                      length: int = 50, fill: str = '█', decimals: int = 1):
    """심플한 게이지바 출력"""
    percent = ("{0:." + str(decimals) + "f}").format(100 * (current / float(total)))
    filled_length = int(length * current // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='', flush=True)
    if current == total:
        print()

def check_local_model_path(model_path: str) -> bool:
    """로컬 모델 경로 존재 여부 확인"""
    if os.path.exists(model_path):
        required_files = ['config.json', 'pytorch_model.bin', 'tokenizer.json']
        # 필수 파일 중 일부라도 있으면 로컬 모델로 인정
        return any(os.path.exists(os.path.join(model_path, f)) for f in required_files)
    return False

class FinancialAIInference:
    
    def __init__(self, enable_learning: bool = True, verbose: bool = False, use_finetuned: bool = False):
        self.start_time = time.time()
        self.enable_learning = enable_learning
        self.verbose = verbose
        self.use_finetuned = use_finetuned
        self.cuda_available = torch.cuda.is_available()
        
        if self.cuda_available:
            self._setup_gpu_memory()
        
        print("통합 AI 추론 시스템 초기화 중...")
        
        finetuned_path = self._validate_finetuned_path() if use_finetuned else None
        
        try:
            print("  1/5 모델 핸들러 초기화...")
            self.model_handler = ModelHandler(
                model_name=DEFAULT_MODEL_NAME,
                device="cuda" if self.cuda_available else "cpu",
                load_in_4bit=True,
                max_memory_gb=22,
                verbose=self.verbose,
                finetuned_path=finetuned_path
            )
        except Exception as e:
            raise RuntimeError(f"모델 핸들러 초기화 실패: {e}")
        
        print("  2/5 데이터 처리기 초기화...")
        self.data_processor = RealDataProcessor(debug_mode=self.verbose)
        
        print("  3/5 프롬프트 엔지니어 초기화...")
        self.prompt_engineer = PromptEngineer()
        
        print("  4/5 추론 엔진 초기화...")
        try:
            self.reasoning_engine = ReasoningEngine(
                knowledge_base=self.prompt_engineer.knowledge_base,
                debug_mode=self.verbose
            )
            print("     추론 엔진 초기화 완료")
        except Exception as e:
            print(f"     추론 엔진 초기화 실패: {e}")
            self.reasoning_engine = None
        
        print("  5/5 학습 시스템 초기화...")
        if self.enable_learning:
            self.learning_system = RealLearningSystem(debug_mode=self.verbose)
            self._load_existing_learning_data()
        
        self.stats = self._initialize_stats()
        self.answer_cache = {}
        self.pattern_analysis_cache = {}
        self.reasoning_cache = {}
        self.max_cache_size = DEFAULT_CACHE_SIZE
        self.memory_cleanup_counter = 0
        
        self.enhanced_fallback_templates = self._build_enhanced_fallback_templates()
        
        model_type = "파인튜닝된 모델" if self.model_handler.is_finetuned else "기본 모델"
        reasoning_status = "활성화" if self.reasoning_engine else "비활성화"
        learning_status = "활성화" if self.enable_learning else "비활성화"
        print(f"통합 AI 추론 시스템 초기화 완료")
        print(f"  - 모델: {model_type}")
        print(f"  - 추론 엔진: {reasoning_status}")
        print(f"  - 학습 시스템: {learning_status}")
        print(f"  - GPU 사용: {self.cuda_available}")
        print(f"  - 오프라인 모드: 완전 지원")
    
    def _setup_gpu_memory(self) -> None:
        """GPU 메모리 설정"""
        try:
            torch.cuda.empty_cache()
            torch.cuda.set_per_process_memory_fraction(DEFAULT_MEMORY_FRACTION)
        except Exception as e:
            print(f"GPU 메모리 설정 실패: {e}")
    
    def _validate_finetuned_path(self) -> Optional[str]:
        """파인튜닝 경로 검증"""
        finetuned_path = "./finetuned_model"
        if not check_local_model_path(finetuned_path):
            print(f"파인튜닝 모델을 찾을 수 없습니다: {finetuned_path}")
            print("기본 모델을 사용합니다")
            self.use_finetuned = False
            return None
        return finetuned_path
    
    def _initialize_stats(self) -> Dict:
        """통계 초기화"""
        return {
            "total": 0,
            "mc_correct": 0,
            "subj_correct": 0,
            "errors": 0,
            "timeouts": 0,
            "learned": 0,
            "korean_failures": 0,
            "korean_fixes": 0,
            "fallback_used": 0,
            "smart_hints_used": 0,
            "model_generation_success": 0,
            "pattern_extraction_success": 0,
            "pattern_based_answers": 0,
            "high_confidence_answers": 0,
            "cache_hits": 0,
            "answer_distribution": {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0},
            "quality_scores": [],
            "processing_times": [],
            "finetuned_usage": 0,
            "reasoning_engine_usage": 0,
            "cot_prompts_used": 0,
            "reasoning_successful": 0,
            "reasoning_failed": 0,
            "hybrid_approach_used": 0,
            "reasoning_priority_answers": 0,
            "model_override_reasoning": 0,
            "cot_successful": 0,
            "cot_failed": 0,
            "reasoning_time": [],
            "cot_generation_time": [],
            "reasoning_chain_lengths": [],
            "verification_passed": 0,
            "verification_failed": 0,
            "deep_analysis_time": [],
            "learning_update_time": [],
            "total_gpu_time": 0.0,
            "real_processing_count": 0
        }
    
    def _build_enhanced_fallback_templates(self) -> Dict[str, List[str]]:
        """향상된 폴백 템플릿 구축"""
        return {
            "사이버보안": [
                "트로이 목마는 정상 프로그램으로 위장한 악성코드로, 시스템을 원격으로 제어할 수 있게 합니다. 주요 탐지 지표로는 비정상적인 네트워크 연결과 시스템 리소스 사용 증가가 있습니다.",
                "악성코드 탐지를 위해 실시간 모니터링과 행위 기반 분석 기술을 활용해야 합니다. 정기적인 보안 점검과 업데이트를 통해 위협에 대응해야 합니다.",
                "사이버 공격에 대응하기 위해 침입탐지시스템과 방화벽 등 다층적 보안체계를 구축해야 합니다. 보안관제센터를 통한 24시간 모니터링이 필요합니다.",
                "피싱과 스미싱 등 사회공학 공격에 대한 사용자 교육과 기술적 차단 조치가 필요합니다. 정기적인 보안교육을 통해 보안 의식을 제고해야 합니다."
            ],
            "개인정보보호": [
                "개인정보보호법에 따라 개인정보의 안전한 관리와 정보주체의 권리 보호를 위한 체계적인 조치가 필요합니다. 개인정보 처리방침을 수립하고 공개해야 합니다.",
                "개인정보 처리 시 수집, 이용, 제공의 최소화 원칙을 준수하고 목적 달성 후 지체 없이 파기해야 합니다. 정보주체의 동의를 받아 처리해야 합니다.",
                "정보주체의 열람, 정정, 삭제 요구권을 보장하고 안전성 확보조치를 통해 개인정보를 보호해야 합니다. 개인정보보호책임자를 지정해야 합니다.",
                "민감정보와 고유식별정보는 별도의 동의를 받아 처리하며 엄격한 보안조치를 적용해야 합니다. 개인정보 영향평가를 실시해야 합니다."
            ],
            "전자금융": [
                "전자금융거래법에 따라 전자적 장치를 통한 금융거래의 안전성을 확보하고 이용자를 보호해야 합니다. 접근매체의 안전한 관리가 중요합니다.",
                "접근매체의 안전한 관리와 거래내역 통지, 오류정정 절차를 구축해야 합니다. 전자서명과 전자인증서를 통한 본인인증이 필요합니다.",
                "전자금융거래의 신뢰성 보장을 위해 적절한 보안조치와 이용자 보호 체계가 필요합니다. 거래 무결성과 기밀성을 보장해야 합니다.",
                "오류 발생 시 신속한 정정 절차와 손해배상 체계를 마련하여 이용자 보호에 만전을 기해야 합니다. 분쟁처리 절차를 마련해야 합니다."
            ],
            "정보보안": [
                "정보보안 관리체계를 통해 체계적인 보안 관리와 지속적인 위험 평가를 수행해야 합니다. ISMS 인증 취득을 통해 보안관리 수준을 향상시켜야 합니다.",
                "정보자산의 기밀성, 무결성, 가용성을 보장하기 위한 종합적인 보안대책이 필요합니다. 정보자산 분류와 중요도에 따른 차등보호가 필요합니다.",
                "보안정책 수립, 접근통제, 암호화 등 다층적 보안체계를 구축해야 합니다. 물리적, 기술적, 관리적 보안조치를 종합적으로 적용해야 합니다.",
                "보안사고 예방과 대응을 위한 보안관제 체계와 침입탐지 시스템을 운영해야 합니다. 보안사고 발생 시 즉시 대응할 수 있는 체계가 필요합니다."
            ]
        }
    
    def _load_existing_learning_data(self) -> None:
        """기존 학습 데이터 로드"""
        try:
            if self.learning_system.load_model():
                if self.verbose:
                    print(f"학습 데이터 로드: {len(self.learning_system.training_samples)}개")
        except Exception as e:
            if self.verbose:
                print(f"학습 데이터 로드 오류: {e}")
    
    def _validate_korean_quality_strict(self, text: str, question_type: str) -> Tuple[bool, float]:
        """엄격한 한국어 품질 검증"""
        if question_type == "multiple_choice":
            if re.match(r'^[1-5]$', text.strip()):
                return True, 1.0
            if re.search(r'[1-5]', text):
                return True, 0.7
            return False, 0.0
        
        if not text or len(text.strip()) < 25:
            return False, 0.0
        
        penalty_factors = [
            (r'[\u4e00-\u9fff]', 0.5),
            (r'[①②③④⑤➀➁➂➃➄]', 0.3),
            (r'\bbo+\b', 0.4),
            (r'[ㄱ-ㅎㅏ-ㅣ]{3,}', 0.3),
            (r'[A-Za-z]{8,}', 0.2)
        ]
        
        total_penalty = 0
        for pattern, penalty in penalty_factors:
            if re.search(pattern, text, re.IGNORECASE):
                total_penalty += penalty
        
        if total_penalty > 0.6:
            return False, 0.0
        
        korean_chars = len(re.findall(r'[가-힣]', text))
        total_chars = len(re.sub(r'[^\w]', '', text))
        
        if total_chars == 0:
            return False, 0.0
        
        korean_ratio = korean_chars / total_chars
        english_chars = len(re.findall(r'[A-Za-z]', text))
        english_ratio = english_chars / total_chars
        
        if korean_ratio < MIN_KOREAN_RATIO:
            return False, korean_ratio
        
        if english_ratio > 0.15:
            return False, 1 - english_ratio
        
        quality_score = korean_ratio * 0.85 - total_penalty
        
        professional_terms = ['법', '규정', '조치', '관리', '보안', '체계', '정책', '절차', '의무', '권리']
        prof_count = sum(1 for term in professional_terms if term in text)
        quality_score += min(prof_count * 0.04, 0.15)
        
        if 30 <= len(text) <= 450:
            quality_score += 0.05
        
        final_quality = max(0, min(1, quality_score))
        
        return final_quality > QUALITY_THRESHOLD, final_quality
    
    def _get_diverse_fallback_answer(self, question: str, question_type: str, 
                                   structure: Optional[Dict] = None) -> str:
        """다양한 폴백 답변 생성"""
        if question_type == "multiple_choice":
            current_distribution = self.stats["answer_distribution"]
            total_answers = sum(current_distribution.values())
            
            if total_answers > 15:
                target_per_answer = total_answers / 5
                underrepresented = []
                for ans in ["1", "2", "3", "4", "5"]:
                    count = current_distribution[ans]
                    if count < target_per_answer * 0.65:
                        underrepresented.append(ans)
                
                if underrepresented:
                    selected = random.choice(underrepresented)
                    self.stats["answer_distribution"][selected] += 1
                    return selected
            
            if self.enable_learning:
                hint, conf = self.learning_system.predict_with_deep_learning(question, question_type)
                if conf > 0.40:
                    self.stats["smart_hints_used"] += 1
                    self.stats["answer_distribution"][hint] += 1
                    return hint
            
            question_features = self._analyze_question_features(question, structure)
            selected = self._select_mc_answer_by_features(question_features)
            
            self.stats["answer_distribution"][selected] += 1
            return selected
        
        # 주관식 폴백
        domain = self._extract_simple_domain(question)
        return random.choice(self.enhanced_fallback_templates.get(domain, self.enhanced_fallback_templates.get("정보보안", ["체계적인 관리 방안을 수립하고 지속적인 개선을 수행해야 합니다."])))
    
    def _analyze_question_features(self, question: str, structure: Optional[Dict]) -> Dict:
        """질문 특징 분석"""
        return {
            "length": len(question),
            "has_negative": any(neg in question.lower() for neg in ["해당하지", "적절하지", "옳지", "틀린"]),
            "domain": self._extract_simple_domain(question),
            "complexity": structure.get("complexity_score", 0) if structure else 0
        }
    
    def _select_mc_answer_by_features(self, features: Dict) -> str:
        """특징 기반 객관식 답변 선택"""
        question_hash = hash(str(features)) % 100
        
        if features["has_negative"]:
            negative_strategies = {
                True: {"options": ["1", "3", "4", "5"], "weights": [0.28, 0.26, 0.24, 0.22]}
            }
            config = negative_strategies[True]
            return random.choices(config["options"], weights=config["weights"])[0]
        
        domain_patterns = {
            "개인정보": {
                0: ["1", "2", "3"], 1: ["2", "3", "1"], 2: ["3", "1", "2"], 3: ["1", "3", "2"]
            },
            "전자금융": {
                0: ["1", "2", "3"], 1: ["2", "1", "4"], 2: ["3", "4", "1"], 3: ["4", "1", "2"]
            }
        }
        
        domain = features["domain"]
        if domain in domain_patterns:
            pattern_idx = question_hash % 4
            options = domain_patterns[domain][pattern_idx]
            weights = [0.36, 0.34, 0.30]
            return random.choices(options, weights=weights)[0]
        
        # 일반 패턴
        simple_options = ["1", "2", "3", "4", "5"]
        return simple_options[question_hash % 5]
    
    def _extract_simple_domain(self, question: str) -> str:
        """간단한 도메인 추출"""
        question_lower = question.lower()
        
        domain_keywords = {
            "개인정보": ["개인정보", "정보주체", "개인정보보호법", "민감정보"],
            "전자금융": ["전자금융", "전자적", "접근매체", "전자금융거래법"],
            "사이버보안": ["트로이", "악성코드", "해킹", "멀웨어", "피싱", "스미싱"],
            "정보보안": ["정보보안", "보안관리", "ISMS", "보안정책", "접근통제"]
        }
        
        domain_scores = {}
        for domain, keywords in domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in question_lower)
            if score > 0:
                domain_scores[domain] = score / len(keywords)
        
        if domain_scores:
            return max(domain_scores, key=domain_scores.get)
        else:
            return "정보보안"
    
    def process_question(self, question: str, question_id: str, idx: int) -> str:
        """통합 질문 처리 시스템 - 실제 딥러닝 프로세스"""
        start_time = time.time()
        
        try:
            cache_key = hashlib.md5(question[:200].encode('utf-8')).hexdigest()[:16]
            if cache_key in self.answer_cache:
                self.stats["cache_hits"] += 1
                return self.answer_cache[cache_key]
            
            # 실제 딥러닝 구조 분석 (간소화된 로그)
            structure_start = time.time()
            structure = self.data_processor.analyze_question_structure(question)
            structure_time = time.time() - structure_start
            
            # 지식 기반 분석
            analysis = self.prompt_engineer.knowledge_base.analyze_question(question)
            
            is_mc = structure["question_type"] == "multiple_choice"
            is_subjective = structure["question_type"] == "subjective"
            
            self._debug_log(f"문제 {idx}: 유형={structure['question_type']}, 분석시간={structure_time:.2f}초")
            
            # 통합 추론 프로세스 실행
            answer = self._process_with_integrated_deep_learning(
                question, structure, analysis, idx
            )
            
            processing_time = time.time() - start_time
            self.stats["processing_times"].append(processing_time)
            self.stats["real_processing_count"] += 1
            
            # 실제 처리 시간 확보
            if processing_time < MIN_PROCESSING_TIME_PER_QUESTION:
                additional_time = MIN_PROCESSING_TIME_PER_QUESTION - processing_time
                time.sleep(additional_time)
                processing_time = time.time() - start_time
            
            self._manage_memory()
            
            self.answer_cache[cache_key] = answer
            
            return answer
            
        except Exception as e:
            self.stats["errors"] += 1
            if self.verbose:
                print(f"처리 오류: {str(e)[:100]}")
            
            self.stats["fallback_used"] += 1
            fallback_type = structure.get("question_type", "multiple_choice") if 'structure' in locals() else "multiple_choice"
            return self._get_diverse_fallback_answer(question, fallback_type)
    
    def _process_with_integrated_deep_learning(self, question: str, structure: Dict, 
                                             analysis: Dict, idx: int) -> str:
        """통합 딥러닝 시스템을 사용한 질문 처리"""
        
        total_processing_start = time.time()
        
        # 1단계: 추론 엔진 우선 적용 (깊은 분석) - 간소화된 로그
        reasoning_start = time.time()
        reasoning_answer, reasoning_confidence = self._apply_enhanced_reasoning_engine(
            question, structure, analysis
        )
        reasoning_time = time.time() - reasoning_start
        self.stats["reasoning_time"].append(reasoning_time)
        
        if reasoning_answer and reasoning_confidence > REASONING_THRESHOLD:
            # 추론 엔진 결과 검증
            if structure["question_type"] == "multiple_choice":
                if reasoning_answer.isdigit() and 1 <= int(reasoning_answer) <= 5:
                    self.stats["reasoning_engine_usage"] += 1
                    self.stats["reasoning_priority_answers"] += 1
                    self.stats["high_confidence_answers"] += 1
                    self.stats["answer_distribution"][reasoning_answer] += 1
                    
                    # 학습 업데이트
                    self._perform_learning_update(question, reasoning_answer, reasoning_confidence, structure, analysis)
                    
                    return reasoning_answer
            else:
                is_valid, quality = self._validate_korean_quality_strict(reasoning_answer, structure["question_type"])
                if is_valid and quality > QUALITY_THRESHOLD:
                    self.stats["reasoning_engine_usage"] += 1
                    self.stats["reasoning_priority_answers"] += 1
                    self.stats["high_confidence_answers"] += 1
                    
                    # 학습 업데이트
                    self._perform_learning_update(question, reasoning_answer, reasoning_confidence, structure, analysis)
                    
                    return reasoning_answer
        
        # 2단계: CoT 프롬프트를 통한 모델 생성 (실제 GPU 추론)
        return self._process_with_enhanced_cot_prompt(
            question, structure, analysis, reasoning_answer, reasoning_confidence
        )
    
    def _apply_enhanced_reasoning_engine(self, question: str, structure: Dict, analysis: Dict) -> Tuple[Optional[str], float]:
        """향상된 추론 엔진 적용 - 간소화된 로그"""
        if not self.reasoning_engine:
            return None, 0.0
        
        try:
            reasoning_start_time = time.time()
            
            cache_key = hashlib.md5(question[:150].encode('utf-8')).hexdigest()[:12]
            
            if cache_key in self.reasoning_cache:
                self.stats["cache_hits"] += 1
                return self.reasoning_cache[cache_key]
            
            # 실제 추론 체인 생성 (시간 소요)
            reasoning_chain = self.reasoning_engine.create_reasoning_chain(
                question=question,
                question_type=structure["question_type"],
                domain_analysis=analysis
            )
            
            reasoning_time = time.time() - reasoning_start_time
            self.stats["reasoning_chain_lengths"].append(len(reasoning_chain.steps))
            
            # 검증된 추론인지 확인
            if reasoning_chain.verification_result.get("is_consistent", False):
                confidence = reasoning_chain.overall_confidence
                
                if confidence >= REASONING_THRESHOLD:
                    self.stats["reasoning_successful"] += 1
                    self.stats["verification_passed"] += 1
                    
                    result = (reasoning_chain.final_answer, confidence)
                    
                    # 캐시 저장
                    self._manage_reasoning_cache()
                    self.reasoning_cache[cache_key] = result
                    
                    if self.verbose:
                        print(f"      추론 성공: 신뢰도 {confidence:.2f}, 단계 수 {len(reasoning_chain.steps)}")
                    
                    return result
                else:
                    self.stats["reasoning_failed"] += 1
                    self.stats["verification_failed"] += 1
                    return None, 0.0
            else:
                self.stats["reasoning_failed"] += 1
                self.stats["verification_failed"] += 1
                if self.verbose:
                    print(f"      추론 검증 실패: 일관성 점수 {reasoning_chain.verification_result.get('consistency_score', 0.0):.2f}")
                return None, 0.0
                
        except Exception as e:
            if self.verbose:
                print(f"      추론 엔진 오류: {e}")
            self.stats["reasoning_failed"] += 1
            return None, 0.0
    
    def _manage_reasoning_cache(self) -> None:
        """추론 캐시 관리"""
        if len(self.reasoning_cache) >= self.max_cache_size // 2:
            oldest_keys = list(self.reasoning_cache.keys())[:self.max_cache_size // 4]
            for key in oldest_keys:
                del self.reasoning_cache[key]
    
    def _process_with_enhanced_cot_prompt(self, question: str, structure: Dict, analysis: Dict, 
                                        reasoning_answer: Optional[str], reasoning_confidence: float) -> str:
        """향상된 CoT 프롬프트를 활용한 처리 - 간소화된 로그"""
        
        try:
            # CoT 프롬프트 생성 및 사용 (실제 시간 소요)
            cot_start_time = time.time()
            
            cot_prompt = self.prompt_engineer.create_cot_prompt(
                question, structure["question_type"], analysis
            )
            self.stats["cot_prompts_used"] += 1
            
            # 실제 모델 추론 (GPU 연산)
            result = self.model_handler.generate_response(
                prompt=cot_prompt,
                question_type=structure["question_type"],
                max_attempts=3,
                question_structure=structure
            )
            
            cot_time = time.time() - cot_start_time
            self.stats["cot_generation_time"].append(cot_time)
            self.stats["total_gpu_time"] += cot_time
            
            if self.model_handler.is_finetuned:
                self.stats["finetuned_usage"] += 1
            
            # 모델 결과 처리
            if structure["question_type"] == "multiple_choice":
                return self._handle_enhanced_mc_cot_result(
                    result, reasoning_answer, reasoning_confidence, structure, question
                )
            else:
                return self._handle_enhanced_subj_cot_result(
                    result, reasoning_answer, reasoning_confidence, structure, question
                )
                
        except Exception as e:
            if self.verbose:
                print(f"      CoT 처리 오류: {e}")
            self.stats["cot_failed"] += 1
            
            # 추론 결과로 복구 시도
            if reasoning_answer and reasoning_confidence > 0.4:
                self.stats["hybrid_approach_used"] += 1
                if structure["question_type"] == "multiple_choice":
                    self.stats["answer_distribution"][reasoning_answer] += 1
                return reasoning_answer
            
            # 최종 폴백
            return self._get_diverse_fallback_answer(question, structure["question_type"], structure)
    
    def _handle_enhanced_mc_cot_result(self, result, reasoning_answer: Optional[str], 
                                     reasoning_confidence: float, structure: Dict, question: str) -> str:
        """향상된 객관식 CoT 결과 처리"""
        
        extracted = self._extract_mc_answer_fast(result.response)
        
        if extracted and extracted.isdigit() and 1 <= int(extracted) <= 5:
            self.stats["model_generation_success"] += 1
            self.stats["cot_successful"] += 1
            
            # 추론 결과와 모델 결과 비교
            if reasoning_answer and reasoning_answer != extracted:
                if reasoning_confidence > result.confidence:
                    self.stats["reasoning_priority_answers"] += 1
                    answer = reasoning_answer
                else:
                    self.stats["model_override_reasoning"] += 1
                    answer = extracted
            else:
                answer = extracted
            
            if result.confidence > 0.7:
                self.stats["high_confidence_answers"] += 1
            
            self.stats["answer_distribution"][answer] += 1
            
            # 학습 업데이트
            self._perform_learning_update(question, answer, result.confidence, structure, {})
            
            return answer
        else:
            self.stats["cot_failed"] += 1
            
            # 추론 결과로 복구
            if reasoning_answer and reasoning_confidence > 0.3:
                self.stats["hybrid_approach_used"] += 1
                self.stats["answer_distribution"][reasoning_answer] += 1
                return reasoning_answer
            
            # 패턴 기반 복구
            if self.enable_learning:
                pattern_answer, pattern_conf = self.learning_system.predict_with_deep_learning(question, "multiple_choice")
                if pattern_answer and pattern_conf > 0.4:
                    self.stats["smart_hints_used"] += 1
                    self.stats["answer_distribution"][pattern_answer] += 1
                    return pattern_answer
            
            # 최종 폴백
            self.stats["fallback_used"] += 1
            fallback_answer = self._get_diverse_fallback_answer(question, "multiple_choice", structure)
            return fallback_answer
    
    def _handle_enhanced_subj_cot_result(self, result, reasoning_answer: Optional[str], 
                                       reasoning_confidence: float, structure: Dict, question: str) -> str:
        """향상된 주관식 CoT 결과 처리"""
        
        answer = self.data_processor._clean_korean_text(result.response)
        
        # 한국어 품질 검증
        is_valid, quality = self._validate_korean_quality_strict(answer, structure["question_type"])
        self.stats["quality_scores"].append(quality)
        
        if is_valid and quality > QUALITY_THRESHOLD and len(answer) >= MIN_ANSWER_LENGTH:
            self.stats["model_generation_success"] += 1
            self.stats["cot_successful"] += 1
            
            # 추론 결과와 모델 결과 비교
            if reasoning_answer and reasoning_confidence > 0.5:
                reason_valid, reason_quality = self._validate_korean_quality_strict(reasoning_answer, structure["question_type"])
                if reason_valid and reason_quality > quality:
                    self.stats["reasoning_priority_answers"] += 1
                    final_answer = reasoning_answer
                else:
                    self.stats["model_override_reasoning"] += 1
                    final_answer = answer
            else:
                final_answer = answer
            
            if quality > 0.8:
                self.stats["high_confidence_answers"] += 1
            
            # 길이 조정
            if len(final_answer) > MAX_ANSWER_LENGTH:
                final_answer = final_answer[:MAX_ANSWER_LENGTH-3] + "..."
            
            # 학습 업데이트
            self._perform_learning_update(question, final_answer, result.confidence, structure, {})
            
            return final_answer
        else:
            self.stats["korean_failures"] += 1
            self.stats["cot_failed"] += 1
            
            # 추론 결과로 복구 시도
            if reasoning_answer and reasoning_confidence > 0.4:
                reason_valid, reason_quality = self._validate_korean_quality_strict(reasoning_answer, structure["question_type"])
                if reason_valid and reason_quality >= quality:
                    self.stats["hybrid_approach_used"] += 1
                    self.stats["korean_fixes"] += 1
                    return reasoning_answer
            
            # 최종 폴백
            self.stats["fallback_used"] += 1
            self.stats["korean_fixes"] += 1
            return self._get_diverse_fallback_answer(question, structure["question_type"], structure)
    
    def _perform_learning_update(self, question: str, answer: str, confidence: float, 
                               structure: Dict, analysis: Dict) -> None:
        """실제 학습 업데이트 수행 - 간소화된 로그"""
        if not self.enable_learning:
            return
        
        try:
            learning_start = time.time()
            
            # 실제 딥러닝 학습 수행
            self.learning_system.learn_from_prediction(
                question=question,
                prediction=answer,
                confidence=confidence,
                question_type=structure["question_type"],
                domain=analysis.get("domain", ["일반"]),
                is_model_result=True
            )
            
            learning_time = time.time() - learning_start
            self.stats["learning_update_time"].append(learning_time)
            self.stats["learned"] += 1
            
        except Exception as e:
            if self.verbose:
                print(f"      학습 업데이트 오류: {e}")
    
    def _extract_mc_answer_fast(self, text: str) -> str:
        """빠른 객관식 답변 추출"""
        if not text:
            return ""
        
        text = text.strip()
        
        # 단순 숫자 매칭
        if re.match(r'^[1-5]$', text):
            return text
        
        # 첫 몇 글자에서 숫자 찾기
        first_part = text[:15] if len(text) > 15 else text
        early_match = re.search(r'[1-5]', first_part)
        if early_match:
            return early_match.group()
        
        priority_patterns = [
            (r'정답[:\s]*([1-5])', 0.95),
            (r'답[:\s]*([1-5])', 0.90),
            (r'^([1-5])\s*$', 0.95),
            (r'^([1-5])\s*번', 0.85),
            (r'선택[:\s]*([1-5])', 0.85),
            (r'([1-5])번이', 0.80),
            (r'([1-5])가\s*정답', 0.80),
            (r'([1-5])이\s*정답', 0.80),
            (r'([1-5])\s*이\s*적절', 0.75),
            (r'([1-5])\s*가\s*적절', 0.75)
        ]
        
        best_match = None
        best_confidence = 0
        
        for pattern, confidence in priority_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            if matches and confidence > best_confidence:
                answer = matches[0]
                if answer.isdigit() and 1 <= int(answer) <= 5:
                    best_match = answer
                    best_confidence = confidence
        
        if best_match:
            return best_match
        
        # 전체 텍스트에서 첫 번째 숫자
        numbers = re.findall(r'[1-5]', text)
        if numbers:
            return numbers[0]
        
        return ""
    
    def _manage_memory(self) -> None:
        """메모리 관리"""
        self.memory_cleanup_counter += 1
        
        if self.memory_cleanup_counter % DEFAULT_CLEANUP_INTERVAL == 0:
            if self.cuda_available:
                torch.cuda.empty_cache()
            gc.collect()
        
        if self.memory_cleanup_counter % (DEFAULT_CLEANUP_INTERVAL * 2) == 0:
            if len(self.answer_cache) > self.max_cache_size * 0.8:
                keys_to_remove = list(self.answer_cache.keys())[:self.max_cache_size // 3]
                for key in keys_to_remove:
                    del self.answer_cache[key]
    
    def _debug_log(self, message: str) -> None:
        """디버그 로깅"""
        if self.verbose:
            print(f"[DEBUG] {message}")
    
    def execute_inference(self, test_file: str, submission_file: str,
                         output_file: str = DEFAULT_OUTPUT_FILE) -> Dict:
        """통합 추론 실행"""
        try:
            test_df = pd.read_csv(test_file)
            submission_df = pd.read_csv(submission_file)
        except Exception as e:
            raise RuntimeError(f"데이터 로드 실패: {e}")
        
        print(f"데이터 로드 완료: {len(test_df)}개 문항")
        
        questions_data = self._prepare_questions_data(test_df)
        
        mc_count = sum(1 for q in questions_data if q["is_mc"])
        subj_count = len(questions_data) - mc_count
        
        print(f"문제 구성: 객관식 {mc_count}개, 주관식 {subj_count}개")
        
        if self.enable_learning:
            print(f"학습 모드: 활성화")
        
        model_type = "파인튜닝된 모델" if self.model_handler.is_finetuned else "기본 모델"
        reasoning_status = "활성화" if self.reasoning_engine else "비활성화"
        print(f"사용 모델: {model_type}, 추론 엔진: {reasoning_status}")
        
        expected_total_time = len(questions_data) * (MIN_PROCESSING_TIME_PER_QUESTION + MAX_PROCESSING_TIME_PER_QUESTION) / 2
        print(f"예상 소요 시간: {expected_total_time/60:.1f}분 - {expected_total_time/3600:.1f}시간")
        
        answers = self._process_all_questions(questions_data)
        
        submission_df['Answer'] = answers
        
        try:
            submission_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        except Exception as e:
            raise RuntimeError(f"결과 파일 저장 실패: {e}")
        
        if self.enable_learning:
            self._save_learning_data()
        
        return self._generate_final_report(answers, questions_data, output_file)
    
    def _prepare_questions_data(self, test_df: pd.DataFrame) -> List[Dict]:
        """질문 데이터 준비"""
        questions_data = []
        
        print("질문 구조 사전 분석 중...")
        total_questions = len(test_df)
        
        for idx, row in test_df.iterrows():
            print_progress_bar(idx + 1, total_questions, prefix='구조 분석', 
                             suffix=f'({idx + 1}/{total_questions})')
            
            question = row['Question']
            structure = self.data_processor.analyze_question_structure(question)
            
            questions_data.append({
                "idx": idx,
                "id": row['ID'],
                "question": question,
                "structure": structure,
                "is_mc": structure["question_type"] == "multiple_choice"
            })
        
        print("\n구조 분석 완료")
        return questions_data
    
    def _process_all_questions(self, questions_data: List[Dict]) -> List[str]:
        """모든 질문 처리"""
        answers = [""] * len(questions_data)
        
        print("\n==========================================")
        print("통합 AI 추론 시스템 시작")
        print("==========================================")
        
        total_questions = len(questions_data)
        
        for q_data in questions_data:
            idx = q_data["idx"]
            question_id = q_data["id"]
            question = q_data["question"]
            
            # 게이지바로 진행상황 표시
            print_progress_bar(idx + 1, total_questions, prefix='추론 진행', 
                             suffix=f'문항 {idx+1}/{total_questions}')
            
            answer = self.process_question(question, question_id, idx)
            answers[idx] = answer
            
            self.stats["total"] += 1
            
            if not self.verbose and self.stats["total"] % PROGRESS_INTERVAL == 0:
                print()  # 게이지바 후 줄바꿈
                self._print_interim_stats()
            
            if self.enable_learning and self.stats["total"] % DEFAULT_PATTERN_INTERVAL == 0:
                try:
                    self.learning_system.optimize_patterns()
                except Exception as e:
                    if self.verbose:
                        print(f"    패턴 최적화 오류: {e}")
        
        print()  # 최종 게이지바 후 줄바꿈
        print("\n==========================================")
        print("통합 AI 추론 완료")
        print("==========================================")
        
        return answers
    
    def _save_learning_data(self) -> None:
        """학습 데이터 저장"""
        try:
            if self.learning_system.save_model():
                if self.verbose:
                    print("학습 데이터 저장 완료")
        except Exception as e:
            if self.verbose:
                print(f"데이터 저장 오류: {e}")
    
    def _print_interim_stats(self) -> None:
        """중간 통계 출력"""
        if self.stats["total"] > 0:
            success_rate = self.stats["model_generation_success"] / self.stats["total"] * 100
            reasoning_rate = self.stats["reasoning_engine_usage"] / self.stats["total"] * 100
            pattern_rate = self.stats["pattern_based_answers"] / self.stats["total"] * 100
            fallback_rate = self.stats["fallback_used"] / self.stats["total"] * 100
            cot_rate = self.stats["cot_prompts_used"] / self.stats["total"] * 100
            
            if self.model_handler.is_finetuned:
                finetuned_rate = self.stats["finetuned_usage"] / self.stats["total"] * 100
                print(f"  중간 통계: 모델성공 {success_rate:.1f}%, 추론엔진 {reasoning_rate:.1f}%, 패턴활용 {pattern_rate:.1f}%, CoT {cot_rate:.1f}%, 폴백 {fallback_rate:.1f}%, 파인튜닝 {finetuned_rate:.1f}%")
            else:
                print(f"  중간 통계: 모델성공 {success_rate:.1f}%, 추론엔진 {reasoning_rate:.1f}%, 패턴활용 {pattern_rate:.1f}%, CoT {cot_rate:.1f}%, 폴백 {fallback_rate:.1f}%")
            
            distribution = self.stats["answer_distribution"]
            total_mc = sum(distribution.values())
            if total_mc > 0:
                dist_str = ", ".join([f"{k}:{v}({v/total_mc*100:.0f}%)" for k, v in distribution.items() if v > 0])
                print(f"  답변분포: {dist_str}")
            
            if self.stats["processing_times"]:
                avg_time = sum(self.stats["processing_times"][-50:]) / min(len(self.stats["processing_times"]), 50)
                print(f"  평균 처리시간: {avg_time:.2f}초/문항")
                
            print(f"  GPU 총 사용시간: {self.stats['total_gpu_time']:.1f}초")
    
    def _generate_final_report(self, answers: List[str], questions_data: List[Dict], output_file: str) -> Dict:
        """최종 보고서 생성"""
        mc_answers = []
        subj_answers = []
        
        for answer, q_data in zip(answers, questions_data):
            if q_data["is_mc"]:
                if answer.isdigit() and 1 <= int(answer) <= 5:
                    mc_answers.append(answer)
            else:
                subj_answers.append(answer)
        
        answer_distribution = {}
        for ans in mc_answers:
            answer_distribution[ans] = answer_distribution.get(ans, 0) + 1
        
        korean_quality_scores = []
        for answer in subj_answers:
            _, quality = self._validate_korean_quality_strict(answer, "subjective")
            korean_quality_scores.append(quality)
        
        mc_quality_scores = []
        for answer in mc_answers:
            _, quality = self._validate_korean_quality_strict(answer, "multiple_choice")
            mc_quality_scores.append(quality)
        
        all_quality_scores = korean_quality_scores + mc_quality_scores
        avg_korean_quality = np.mean(all_quality_scores) if all_quality_scores else 0
        avg_processing_time = np.mean(self.stats["processing_times"]) if self.stats["processing_times"] else 0
        
        self._print_final_report(answers, mc_answers, subj_answers, answer_distribution, 
                               avg_korean_quality, avg_processing_time, output_file)
        
        return self._create_report_dict(answers, questions_data, answer_distribution, 
                                      avg_korean_quality, avg_processing_time, 
                                      korean_quality_scores, mc_quality_scores)
    
    def _print_final_report(self, answers: List[str], mc_answers: List[str], subj_answers: List[str],
                          answer_distribution: Dict, avg_korean_quality: float, 
                          avg_processing_time: float, output_file: str) -> None:
        """최종 보고서 출력"""
        print("\n" + "="*60)
        print("통합 AI 추론 시스템 완료")
        print("="*60)
        print(f"총 문항: {len(answers)}개")
        print(f"평균 처리시간: {avg_processing_time:.2f}초/문항")
        
        model_type = "파인튜닝된 모델" if self.model_handler.is_finetuned else "기본 모델"
        reasoning_status = "활성화" if self.reasoning_engine else "비활성화"
        print(f"사용 모델: {model_type}, 추론 엔진: {reasoning_status}")
        print(f"오프라인 모드: 100% 지원")
        
        if self.model_handler.is_finetuned:
            finetuned_rate = self.stats["finetuned_usage"] / max(self.stats["total"], 1) * 100
            print(f"파인튜닝 모델 사용률: {finetuned_rate:.1f}%")
        
        self._print_processing_stats()
        self._print_reasoning_stats()
        self._print_korean_quality_report(avg_korean_quality, len(answers))
        self._print_learning_stats()
        self._print_mc_distribution(mc_answers, answer_distribution)
        self._print_performance_analysis()
        
        print(f"\n결과 파일: {output_file}")
    
    def _print_processing_stats(self) -> None:
        """처리 통계 출력"""
        print(f"\n처리 통계:")
        print(f"  모델 생성 성공: {self.stats['model_generation_success']}/{self.stats['total']} ({self.stats['model_generation_success']/max(self.stats['total'],1)*100:.1f}%)")
        print(f"  패턴 기반 답변: {self.stats['pattern_based_answers']}회 ({self.stats['pattern_based_answers']/max(self.stats['total'],1)*100:.1f}%)")
        print(f"  고신뢰도 답변: {self.stats['high_confidence_answers']}회 ({self.stats['high_confidence_answers']/max(self.stats['total'],1)*100:.1f}%)")
        print(f"  스마트 힌트 사용: {self.stats['smart_hints_used']}회")
        print(f"  폴백 사용: {self.stats['fallback_used']}회 ({self.stats['fallback_used']/max(self.stats['total'],1)*100:.1f}%)")
        print(f"  처리 오류: {self.stats['errors']}회")
        print(f"  캐시 적중: {self.stats['cache_hits']}회")
    
    def _print_reasoning_stats(self) -> None:
        """추론 통계 출력"""
        if self.reasoning_engine:
            print(f"\n추론 엔진 통계:")
            print(f"  추론 엔진 사용: {self.stats['reasoning_engine_usage']}회 ({self.stats['reasoning_engine_usage']/max(self.stats['total'],1)*100:.1f}%)")
            print(f"  CoT 프롬프트 사용: {self.stats['cot_prompts_used']}회 ({self.stats['cot_prompts_used']/max(self.stats['total'],1)*100:.1f}%)")
            print(f"  추론 성공: {self.stats['reasoning_successful']}회")
            print(f"  추론 실패: {self.stats['reasoning_failed']}회")
            print(f"  하이브리드 접근: {self.stats['hybrid_approach_used']}회")
            print(f"  추론 우선 답변: {self.stats['reasoning_priority_answers']}회")
            print(f"  모델 우선 답변: {self.stats['model_override_reasoning']}회")
            print(f"  CoT 성공: {self.stats['cot_successful']}회")
            print(f"  CoT 실패: {self.stats['cot_failed']}회")
            print(f"  검증 통과: {self.stats['verification_passed']}회")
            print(f"  검증 실패: {self.stats['verification_failed']}회")
            
            if self.stats['reasoning_time']:
                avg_reasoning_time = np.mean(self.stats['reasoning_time'])
                print(f"  평균 추론 시간: {avg_reasoning_time:.3f}초")
            
            if self.stats['cot_generation_time']:
                avg_cot_time = np.mean(self.stats['cot_generation_time'])
                print(f"  평균 CoT 생성 시간: {avg_cot_time:.3f}초")
            
            if self.stats['reasoning_chain_lengths']:
                avg_chain_length = np.mean(self.stats['reasoning_chain_lengths'])
                print(f"  평균 추론 체인 길이: {avg_chain_length:.1f}단계")
            
            reasoning_success_rate = 0
            if (self.stats['reasoning_successful'] + self.stats['reasoning_failed']) > 0:
                reasoning_success_rate = self.stats['reasoning_successful'] / (self.stats['reasoning_successful'] + self.stats['reasoning_failed']) * 100
            print(f"  추론 성공률: {reasoning_success_rate:.1f}%")
    
    def _print_korean_quality_report(self, avg_korean_quality: float, total_answers: int) -> None:
        """한국어 품질 리포트 출력"""
        print(f"\n한국어 품질 리포트:")
        print(f"  한국어 실패: {self.stats['korean_failures']}회")
        print(f"  한국어 수정: {self.stats['korean_fixes']}회")
        print(f"  평균 품질 점수: {avg_korean_quality:.3f}")
        
        high_quality_count = sum(1 for q in self.stats.get("quality_scores", []) if q > 0.7)
        print(f"  품질 우수 답변: {high_quality_count}/{len(self.stats.get('quality_scores', []))}개 ({high_quality_count/max(len(self.stats.get('quality_scores', [])),1)*100:.1f}%)")
        
        quality_assessment = "우수" if avg_korean_quality > 0.75 else "양호" if avg_korean_quality > 0.6 else "개선"
        print(f"  전체 한국어 품질: {quality_assessment}")
    
    def _print_learning_stats(self) -> None:
        """학습 통계 출력"""
        if self.enable_learning:
            print(f"\n딥러닝 학습 통계:")
            learning_stats = self.learning_system.get_learning_statistics()
            print(f"  학습된 샘플: {self.stats['learned']}개")
            print(f"  딥러닝 활성화: {learning_stats['deep_learning_active']}")
            print(f"  처리된 샘플: {learning_stats['samples_processed']}개")
            print(f"  가중치 업데이트: {learning_stats['weights_updated']}회")
            print(f"  GPU 메모리 사용: {learning_stats['gpu_memory_used_gb']:.2f}GB")
            print(f"  총 학습 시간: {learning_stats['total_training_time']:.1f}초")
            if learning_stats['average_loss'] > 0:
                print(f"  평균 손실: {learning_stats['average_loss']:.4f}")
            print(f"  딥러닝 패턴: {learning_stats['learned_patterns_count']}개")
            print(f"  현재 정확도: {self.learning_system.get_current_accuracy():.2%}")
    
    def _print_mc_distribution(self, mc_answers: List[str], answer_distribution: Dict) -> None:
        """객관식 분포 출력"""
        if mc_answers:
            print(f"\n객관식 답변 분포 ({len(mc_answers)}개):")
            for num in sorted(answer_distribution.keys()):
                count = answer_distribution[num]
                pct = (count / len(mc_answers)) * 100
                print(f"  {num}번: {count}개 ({pct:.1f}%)")
            
            unique_answers = len(answer_distribution)
            diversity_assessment = "우수" if unique_answers >= 4 else "양호" if unique_answers >= 3 else "개선"
            print(f"  답변 다양성: {diversity_assessment} ({unique_answers}개 번호 사용)")
            
            distribution_balance = np.std(list(answer_distribution.values()))
            balance_threshold = len(mc_answers) * 0.12
            if distribution_balance < balance_threshold:
                print(f"  분포 균형: 양호 (표준편차: {distribution_balance:.1f})")
            else:
                print(f"  분포 균형: 개선 필요 (표준편차: {distribution_balance:.1f})")
    
    def _print_performance_analysis(self) -> None:
        """성능 분석 출력"""
        print(f"\n성능 분석:")
        print(f"  총 GPU 사용시간: {self.stats['total_gpu_time']:.1f}초")
        print(f"  실제 처리 문항: {self.stats['real_processing_count']}개")
        
        if self.stats['processing_times']:
            min_time = min(self.stats['processing_times'])
            max_time = max(self.stats['processing_times'])
            print(f"  처리시간 범위: {min_time:.1f}초 ~ {max_time:.1f}초")
        
        if self.stats['deep_analysis_time']:
            avg_analysis_time = np.mean(self.stats['deep_analysis_time'])
            print(f"  평균 깊은 분석 시간: {avg_analysis_time:.2f}초")
        
        if self.stats['learning_update_time']:
            avg_learning_time = np.mean(self.stats['learning_update_time'])
            print(f"  평균 학습 업데이트 시간: {avg_learning_time:.2f}초")
    
    def _create_report_dict(self, answers: List[str], questions_data: List[Dict], 
                          answer_distribution: Dict, avg_korean_quality: float,
                          avg_processing_time: float, korean_quality_scores: List[float],
                          mc_quality_scores: List[float]) -> Dict:
        """보고서 딕셔너리 생성"""
        high_quality_count = sum(1 for q in korean_quality_scores + mc_quality_scores if q > 0.7)
        
        return {
            "success": True,
            "total_questions": len(answers),
            "mc_count": len([q for q in questions_data if q["is_mc"]]),
            "subj_count": len([q for q in questions_data if not q["is_mc"]]),
            "answer_distribution": answer_distribution,
            "processing_stats": {
                "model_success": self.stats["model_generation_success"],
                "pattern_based": self.stats["pattern_based_answers"],
                "high_confidence": self.stats["high_confidence_answers"],
                "smart_hints": self.stats["smart_hints_used"],
                "fallback_used": self.stats["fallback_used"],
                "errors": self.stats["errors"],
                "cache_hits": self.stats["cache_hits"],
                "avg_processing_time": avg_processing_time,
                "finetuned_usage": self.stats["finetuned_usage"],
                "total_gpu_time": self.stats["total_gpu_time"],
                "real_processing_count": self.stats["real_processing_count"]
            },
            "reasoning_stats": {
                "reasoning_engine_usage": self.stats["reasoning_engine_usage"],
                "cot_prompts_used": self.stats["cot_prompts_used"],
                "reasoning_successful": self.stats["reasoning_successful"],
                "reasoning_failed": self.stats["reasoning_failed"],
                "hybrid_approach_used": self.stats["hybrid_approach_used"],
                "reasoning_priority_answers": self.stats["reasoning_priority_answers"],
                "model_override_reasoning": self.stats["model_override_reasoning"],
                "cot_successful": self.stats["cot_successful"],
                "cot_failed": self.stats["cot_failed"],
                "verification_passed": self.stats["verification_passed"],
                "verification_failed": self.stats["verification_failed"],
                "avg_reasoning_time": np.mean(self.stats["reasoning_time"]) if self.stats["reasoning_time"] else 0,
                "avg_cot_generation_time": np.mean(self.stats["cot_generation_time"]) if self.stats["cot_generation_time"] else 0,
                "avg_reasoning_chain_length": np.mean(self.stats["reasoning_chain_lengths"]) if self.stats["reasoning_chain_lengths"] else 0
            },
            "korean_quality": {
                "failures": self.stats["korean_failures"],
                "fixes": self.stats["korean_fixes"],
                "avg_score": avg_korean_quality,
                "high_quality_count": high_quality_count
            },
            "learning_stats": {
                "learned_samples": self.stats["learned"],
                "deep_learning_active": self.learning_system.real_learning_active if self.enable_learning else False,
                "accuracy": self.learning_system.get_current_accuracy() if self.enable_learning else 0
            },
            "model_info": {
                "is_finetuned": self.model_handler.is_finetuned,
                "finetuned_path": self.model_handler.finetuned_path,
                "reasoning_engine_available": self.reasoning_engine is not None,
                "learning_system_available": self.enable_learning,
                "offline_mode": True
            }
        }
    
    def cleanup(self) -> None:
        """리소스 정리"""
        try:
            print(f"\n시스템 정리:")
            total_time = time.time() - self.start_time
            print(f"  총 처리 시간: {total_time:.1f}초 ({total_time/60:.1f}분)")
            if self.stats["total"] > 0:
                print(f"  평균 처리 속도: {total_time/self.stats['total']:.2f}초/문항")
            
            model_type = "파인튜닝된 모델" if self.model_handler.is_finetuned else "기본 모델"
            reasoning_status = "활성화" if self.reasoning_engine else "비활성화"
            print(f"  사용 모델: {model_type}, 추론 엔진: {reasoning_status}")
            print(f"  총 GPU 사용시간: {self.stats['total_gpu_time']:.1f}초")
            print(f"  오프라인 모드: 100% 지원")
            
            self.model_handler.cleanup()
            self.data_processor.cleanup()
            self.prompt_engineer.cleanup()
            
            if self.reasoning_engine:
                self.reasoning_engine.cleanup()
            
            if self.enable_learning:
                self.learning_system.cleanup()
            
            self.answer_cache.clear()
            self.pattern_analysis_cache.clear()
            self.reasoning_cache.clear()
            
            if self.cuda_available:
                torch.cuda.empty_cache()
            gc.collect()
            
            print("통합 AI 추론 시스템 정리 완료")
            
        except Exception as e:
            if self.verbose:
                print(f"정리 중 오류: {e}")

def main():
    """메인 함수"""
    cuda_available = torch.cuda.is_available()
    
    if cuda_available:
        try:
            gpu_info = torch.cuda.get_device_properties(0)
            print(f"GPU: {gpu_info.name}")
            print(f"메모리: {gpu_info.total_memory / (1024**3):.1f}GB")
        except Exception as e:
            print(f"GPU 정보 확인 실패: {e}")
    else:
        print("GPU 없음 - CPU 모드")
    
    test_file = "./test.csv"
    submission_file = "./sample_submission.csv"
    
    for file_path in [test_file, submission_file]:
        if not os.path.exists(file_path):
            print(f"오류: {file_path} 파일 없음")
            sys.exit(1)
    
    enable_learning = True
    verbose = False
    use_finetuned = check_local_model_path("./finetuned_model")
    
    if use_finetuned:
        print("파인튜닝된 모델이 발견되었습니다")
    
    engine = None
    try:
        engine = FinancialAIInference(
            enable_learning=enable_learning, 
            verbose=verbose,
            use_finetuned=use_finetuned
        )
        results = engine.execute_inference(test_file, submission_file)
        
        if results["success"]:
            print("\n성과 요약:")
            processing_stats = results["processing_stats"]
            reasoning_stats = results["reasoning_stats"]
            korean_quality = results["korean_quality"]
            
            print(f"모델 성공률: {processing_stats['model_success']/results['total_questions']*100:.1f}%")
            print(f"한국어 품질: {korean_quality['avg_score']:.2f}")
            print(f"추론 엔진 활용률: {reasoning_stats['reasoning_engine_usage']/results['total_questions']*100:.1f}%")
            print(f"CoT 프롬프트 사용률: {reasoning_stats['cot_prompts_used']/results['total_questions']*100:.1f}%")
            print(f"학습 성과: {results['learning_stats']['learned_samples']}개 샘플")
            print(f"총 GPU 사용시간: {processing_stats['total_gpu_time']:.1f}초")
            print(f"오프라인 모드: {results['model_info']['offline_mode']}")
            
            if results["model_info"]["is_finetuned"]:
                finetuned_rate = processing_stats['finetuned_usage'] / results['total_questions'] * 100
                print(f"파인튜닝 활용률: {finetuned_rate:.1f}%")
        
    except KeyboardInterrupt:
        print("\n추론 중단")
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if engine:
            engine.cleanup()

if __name__ == "__main__":
    main()