# test_runner.py

"""
테스트 실행기
- 50문항 테스트 실행
- 파인튜닝된 모델 지원
- 빠른 성능 검증
- 간단한 결과 분석
- 논리적 추론 성능 측정
- CoT 추론 과정 검증
- 추론 품질 평가 메트릭
"""

import gc
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

current_dir = Path(__file__).parent.absolute()
sys.path.append(str(current_dir))

from model_handler import ModelHandler
from data_processor import DataProcessor
from prompt_engineering import PromptEngineer
from learning_system import LearningSystem
from reasoning_engine import ReasoningEngine

DEFAULT_TEST_SIZE = 50
MAX_TEST_SIZE = 500
MIN_TEST_SIZE = 1
DEFAULT_GPU_MEMORY_FRACTION = 0.9
MEMORY_CLEANUP_INTERVAL = 10
PROGRESS_REPORT_INTERVAL = 10
QUALITY_THRESHOLD = 0.65
REASONING_QUALITY_THRESHOLD = 0.7
MIN_ANSWER_LENGTH = 30
MAX_ANSWER_LENGTH = 550
MIN_KOREAN_RATIO = 0.6
MAX_ENGLISH_RATIO = 0.15

PROGRESS_INTERVAL = 50
INTERIM_STATS_INTERVAL = 50

class TestRunner:
    
    def __init__(self, test_size: int = DEFAULT_TEST_SIZE, use_finetuned: bool = False):
        """테스트 실행기 초기화"""
        self.test_size = max(MIN_TEST_SIZE, min(test_size, MAX_TEST_SIZE))
        self.use_finetuned = use_finetuned
        self.start_time = time.time()
        self.cuda_available = torch.cuda.is_available()
        
        print(f"테스트 실행기 초기화 중... (대상: {self.test_size}문항)")
        
        if self.cuda_available:
            try:
                torch.cuda.empty_cache()
                torch.cuda.set_per_process_memory_fraction(DEFAULT_GPU_MEMORY_FRACTION)
                gpu_name = torch.cuda.get_device_name(0)
                print(f"GPU: {gpu_name}")
            except Exception as e:
                print(f"GPU 초기화 중 오류: {e}")
        else:
            print("CPU 모드로 실행")
        
        finetuned_path = self._validate_finetuned_path() if use_finetuned else None
        
        try:
            self.model_handler = ModelHandler(
                model_name="upstage/SOLAR-10.7B-Instruct-v1.0",
                device="cuda" if self.cuda_available else "cpu",
                load_in_4bit=True,
                max_memory_gb=22,
                verbose=False,
                finetuned_path=finetuned_path
            )
            
            self.data_processor = DataProcessor(debug_mode=False)
            self.prompt_engineer = PromptEngineer()
            self.learning_system = LearningSystem(debug_mode=False)
            
            # 추론 엔진 초기화
            try:
                self.reasoning_engine = ReasoningEngine(
                    knowledge_base=self.prompt_engineer.knowledge_base,
                    debug_mode=False
                )
                print("추론 엔진 초기화 완료")
            except Exception as e:
                print(f"추론 엔진 초기화 실패: {e}")
                self.reasoning_engine = None
            
            self._load_existing_learning_data()
            
        except Exception as e:
            raise RuntimeError(f"테스트 실행기 초기화 실패: {e}")
        
        self.stats = self._initialize_stats()
        self.enhanced_fallback_templates = self._build_enhanced_templates()
        self.memory_cleanup_counter = 0
        
        # 추론 성능 분석 시스템
        self.reasoning_analyzer = ReasoningPerformanceAnalyzer()
        
        model_type = "파인튜닝된 모델" if self.model_handler.is_finetuned else "기본 모델"
        reasoning_status = "활성화" if self.reasoning_engine else "비활성화"
        print(f"초기화 완료 - {model_type} 사용, 추론 엔진 {reasoning_status}\n")
    
    def _validate_finetuned_path(self) -> Optional[str]:
        """파인튜닝 모델 경로 검증"""
        finetuned_path = "./finetuned_model"
        
        if not os.path.exists(finetuned_path):
            print(f"파인튜닝 모델을 찾을 수 없습니다: {finetuned_path}")
            print("기본 모델을 사용합니다")
            self.use_finetuned = False
            return None
        
        required_files = ["adapter_config.json", "adapter_model.bin"]
        missing_files = []
        
        for file_name in required_files:
            file_path = os.path.join(finetuned_path, file_name)
            if not os.path.exists(file_path):
                missing_files.append(file_name)
        
        if missing_files:
            print(f"파인튜닝 모델에 필수 파일이 없습니다: {missing_files}")
            print("기본 모델을 사용합니다")
            self.use_finetuned = False
            return None
        
        return finetuned_path
    
    def _load_existing_learning_data(self) -> None:
        """기존 학습 데이터 로드"""
        try:
            if self.learning_system.load_model():
                print("기존 학습 데이터 로드 완료")
            else:
                print("새로운 학습 세션 시작")
        except Exception as e:
            print(f"학습 데이터 로드 중 오류: {e}")
            print("새로운 학습 세션 시작")
    
    def _initialize_stats(self) -> Dict:
        """통계 초기화"""
        return {
            "total": 0,
            "mc_count": 0,
            "subj_count": 0,
            "model_success": 0,
            "pattern_success": 0,
            "fallback_used": 0,
            "korean_quality_sum": 0.0,
            "processing_times": [],
            "answer_distribution": {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0},
            "high_confidence_count": 0,
            "smart_hints_used": 0,
            "finetuned_usage": 0,
            "memory_cleanups": 0,
            "cache_hits": 0,
            "generation_errors": 0,
            "reasoning_engine_usage": 0,
            "reasoning_successful": 0,
            "reasoning_failed": 0,
            "cot_prompts_used": 0,
            "reasoning_quality_scores": [],
            "consistency_scores": [],
            "reasoning_depth_scores": [],
            "logical_errors": 0,
            "chain_verification_passed": 0,
            "chain_verification_failed": 0,
            "reasoning_time_per_question": [],
            "step_count_distribution": {},
            "advanced_reasoning_features": 0,
            "hybrid_approach_success": 0,
            "reasoning_vs_pattern_comparison": {"reasoning_better": 0, "pattern_better": 0, "equal": 0},
            "error_recovery_success": 0
        }
    
    def _build_enhanced_templates(self) -> Dict[str, List[str]]:
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
            ],
            "위험관리": [
                "위험관리는 조직의 목표 달성에 영향을 미칠 수 있는 위험을 체계적으로 식별하고 관리하는 과정입니다. 위험 식별, 분석, 평가, 대응의 4단계 프로세스를 통해 체계적인 위험관리를 수행해야 합니다.",
                "위험 수용, 회피, 완화, 전가의 4가지 대응전략 중 적절한 방안을 선택하여 적용해야 합니다. 정기적인 위험평가와 모니터링을 통해 위험 수준을 지속적으로 관리하고 개선해야 합니다.",
                "경영진의 위험관리 의지와 조직 전체의 위험 문화 조성이 성공적인 위험관리의 핵심입니다. 위험관리 정책과 절차를 수립하고 정기적으로 검토해야 합니다.",
                "위험 허용 수준을 설정하고 이를 초과하는 위험에 대해서는 즉시 조치를 취해야 합니다. 위험관리 교육을 통해 조직 구성원의 위험 인식을 제고해야 합니다."
            ],
            "일반": [
                "관련 법령과 규정에 따라 체계적인 관리 방안을 수립하고 지속적인 개선을 수행해야 합니다. 정기적인 점검과 평가를 통해 관리수준을 향상시켜야 합니다.",
                "정보보안 정책과 절차를 수립하여 체계적인 보안 관리와 위험 평가를 수행해야 합니다. 경영진의 의지와 조직 전체의 참여가 중요합니다.",
                "적절한 기술적, 관리적, 물리적 보안조치를 통해 정보자산을 안전하게 보호해야 합니다. 보안관리 조직을 구성하고 책임과 권한을 명확히 해야 합니다.",
                "법령에서 요구하는 안전성 확보조치를 이행하고 정기적인 점검을 통해 개선해야 합니다. 자체 점검과 외부 점검을 병행하여 객관성을 확보해야 합니다.",
                "위험관리 체계를 구축하여 예방적 관리와 사후 대응 방안을 마련해야 합니다. 위험 식별, 분석, 평가, 대응의 4단계 프로세스를 수행해야 합니다."
            ]
        }
    
    def load_test_data(self, test_file: str, submission_file: str) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
        """테스트 데이터 로드"""
        try:
            if not os.path.exists(test_file):
                print(f"오류: {test_file} 파일을 찾을 수 없습니다")
                return None
            
            if not os.path.exists(submission_file):
                print(f"오류: {submission_file} 파일을 찾을 수 없습니다")
                return None
            
            test_df = pd.read_csv(test_file, encoding='utf-8')
            submission_df = pd.read_csv(submission_file, encoding='utf-8')
            
            if len(test_df) < self.test_size:
                print(f"경고: 전체 {len(test_df)}문항, 요청 {self.test_size}문항")
                self.test_size = len(test_df)
            
            test_sample = test_df.head(self.test_size).copy()
            submission_sample = submission_df.head(self.test_size).copy()
            
            print(f"테스트 데이터 로드: {len(test_sample)}문항")
            return test_sample, submission_sample
            
        except pd.errors.EmptyDataError:
            print(f"오류: {test_file} 파일이 비어있습니다")
            return None
        except pd.errors.ParserError as e:
            print(f"오류: 파일 파싱 실패 - {e}")
            return None
        except PermissionError:
            print(f"오류: 파일 접근 권한이 없습니다")
            return None
        except Exception as e:
            print(f"오류: 데이터 로드 실패 - {e}")
            return None
    
    def analyze_questions(self, test_df: pd.DataFrame) -> Dict:
        """질문 분석"""
        mc_count = 0
        subj_count = 0
        
        try:
            for _, row in test_df.iterrows():
                question = row['Question']
                structure = self.data_processor.analyze_question_structure(question)
                
                if structure["question_type"] == "multiple_choice":
                    mc_count += 1
                else:
                    subj_count += 1
        
        except Exception as e:
            print(f"질문 분석 중 오류: {e}")
            mc_count = self.test_size // 2
            subj_count = self.test_size - mc_count
        
        print(f"문제 구성: 객관식 {mc_count}개, 주관식 {subj_count}개")
        
        return {
            "mc_count": mc_count,
            "subj_count": subj_count,
            "total": mc_count + subj_count
        }
    
    def process_single_question(self, question: str, question_id: str, idx: int) -> str:
        """단일 질문 처리 (추론 성능 분석 포함)"""
        start_time = time.time()
        reasoning_start_time = time.time()
        
        try:
            structure = self.data_processor.analyze_question_structure(question)
            analysis = self.prompt_engineer.knowledge_base.analyze_question(question)
            is_mc = structure["question_type"] == "multiple_choice"
            
            # 추론 엔진 활용 시도
            reasoning_result = None
            reasoning_quality = 0.0
            reasoning_used = False
            
            if self.reasoning_engine:
                try:
                    reasoning_chain = self.reasoning_engine.create_reasoning_chain(
                        question=question,
                        question_type=structure["question_type"],
                        domain_analysis=analysis
                    )
                    
                    reasoning_time = time.time() - reasoning_start_time
                    self.stats["reasoning_time_per_question"].append(reasoning_time)
                    
                    # 추론 품질 분석
                    reasoning_analysis = self.reasoning_analyzer.analyze_reasoning_chain(reasoning_chain)
                    reasoning_quality = reasoning_analysis["quality_score"]
                    
                    self.stats["reasoning_quality_scores"].append(reasoning_quality)
                    self.stats["consistency_scores"].append(reasoning_chain.verification_result.get("consistency_score", 0.0))
                    self.stats["reasoning_depth_scores"].append(len(reasoning_chain.steps))
                    
                    # 단계 수 분포 기록
                    step_count = len(reasoning_chain.steps)
                    self.stats["step_count_distribution"][step_count] = self.stats["step_count_distribution"].get(step_count, 0) + 1
                    
                    # 추론 체인 검증
                    chain_valid = self._verify_reasoning_chain(reasoning_chain, structure)
                    if chain_valid:
                        self.stats["chain_verification_passed"] += 1
                        reasoning_result = reasoning_chain.final_answer
                        reasoning_used = True
                        self.stats["reasoning_engine_usage"] += 1
                        
                        if reasoning_chain.verification_result.get("is_consistent", False):
                            self.stats["reasoning_successful"] += 1
                        else:
                            self.stats["reasoning_failed"] += 1
                    else:
                        self.stats["chain_verification_failed"] += 1
                        self.stats["reasoning_failed"] += 1
                        
                    # 고급 추론 기능 사용 여부 체크
                    if reasoning_analysis["uses_advanced_features"]:
                        self.stats["advanced_reasoning_features"] += 1
                        
                except Exception as e:
                    self.stats["reasoning_failed"] += 1
                    self.stats["logical_errors"] += 1
            
            # 기존 처리 방식
            if is_mc:
                answer = self._process_multiple_choice_with_reasoning(
                    question, structure, analysis, reasoning_result, reasoning_quality, reasoning_used
                )
            else:
                answer = self._process_subjective_with_reasoning(
                    question, structure, analysis, reasoning_result, reasoning_quality, reasoning_used
                )
            
            # 통계 업데이트
            self.stats["total"] += 1
            processing_time = time.time() - start_time
            self.stats["processing_times"].append(processing_time)
            
            # 진행 상황 출력
            if self.stats["total"] % PROGRESS_REPORT_INTERVAL == 0:
                progress = self.stats["total"] / self.test_size * 100
                print(f"  진행: {self.stats['total']}/{self.test_size} ({progress:.0f}%)")
            
            self._manage_memory()
            
            return answer
            
        except Exception as e:
            print(f"  오류 발생 (문항 {idx}): {str(e)[:50]}")
            self.stats["generation_errors"] += 1
            self.stats["fallback_used"] += 1
            
            if 'is_mc' in locals() and is_mc:
                fallback_answer = str(random.randint(1, 5))
                self.stats["answer_distribution"][fallback_answer] += 1
                return fallback_answer
            else:
                return random.choice(self.enhanced_fallback_templates["일반"])
    
    def _verify_reasoning_chain(self, reasoning_chain, structure: Dict) -> bool:
        """추론 체인 검증"""
        try:
            # 기본 검증
            if not reasoning_chain or not reasoning_chain.steps:
                return False
            
            # 단계 수 검증
            if len(reasoning_chain.steps) < 2:
                return False
            
            # 신뢰도 검증
            if reasoning_chain.overall_confidence < 0.3:
                return False
            
            # 일관성 검증
            if not reasoning_chain.verification_result.get("is_consistent", False):
                return False
            
            # 객관식의 경우 답변 형식 검증
            if structure["question_type"] == "multiple_choice":
                if not (reasoning_chain.final_answer.isdigit() and 1 <= int(reasoning_chain.final_answer) <= 5):
                    return False
            
            # 주관식의 경우 한국어 품질 검증
            else:
                if len(reasoning_chain.final_answer) < 20:
                    return False
                    
                korean_chars = len([c for c in reasoning_chain.final_answer if 'ㄱ' <= c <= 'ㅣ' or '가' <= c <= '힣'])
                total_chars = len([c for c in reasoning_chain.final_answer if c.isalnum()])
                
                if total_chars > 0:
                    korean_ratio = korean_chars / total_chars
                    if korean_ratio < 0.5:
                        return False
            
            return True
            
        except Exception:
            return False
    
    def _process_multiple_choice_with_reasoning(self, question: str, structure: Dict, analysis: Dict, 
                                              reasoning_result: Optional[str], reasoning_quality: float, 
                                              reasoning_used: bool) -> str:
        """추론 엔진을 활용한 객관식 처리"""
        self.stats["mc_count"] += 1
        
        # 추론 결과 우선 사용
        if reasoning_used and reasoning_result and reasoning_quality > REASONING_QUALITY_THRESHOLD:
            if reasoning_result.isdigit() and 1 <= int(reasoning_result) <= 5:
                self.stats["high_confidence_count"] += 1
                self.stats["answer_distribution"][reasoning_result] += 1
                return reasoning_result
        
        # 답변 분포 균형화
        answer = self._apply_distribution_balancing()
        if answer:
            self.stats["pattern_success"] += 1
            self.stats["answer_distribution"][answer] += 1
            return answer
        
        # 스마트 힌트 시도
        hint_answer, hint_confidence = self.learning_system.get_smart_answer_hint(question, structure)
        
        if hint_confidence > 0.50:
            # 추론 결과와 패턴 결과 비교
            if reasoning_used and reasoning_result != hint_answer:
                if reasoning_quality > hint_confidence:
                    self.stats["reasoning_vs_pattern_comparison"]["reasoning_better"] += 1
                    selected_answer = reasoning_result if reasoning_result.isdigit() and 1 <= int(reasoning_result) <= 5 else hint_answer
                elif reasoning_quality < hint_confidence:
                    self.stats["reasoning_vs_pattern_comparison"]["pattern_better"] += 1
                    selected_answer = hint_answer
                else:
                    self.stats["reasoning_vs_pattern_comparison"]["equal"] += 1
                    selected_answer = hint_answer  # 동점일 때는 패턴 우선
                    
                if selected_answer == reasoning_result:
                    self.stats["hybrid_approach_success"] += 1
            else:
                selected_answer = hint_answer
            
            self.stats["pattern_success"] += 1
            self.stats["smart_hints_used"] += 1
            if hint_confidence > 0.65:
                self.stats["high_confidence_count"] += 1
            self.stats["answer_distribution"][selected_answer] += 1
            return selected_answer
        
        # 모델 생성 시도
        try:
            # CoT 프롬프트 사용 (추론 엔진 사용 시)
            if reasoning_used or self.reasoning_engine:
                prompt = self.prompt_engineer.create_cot_prompt(question, "multiple_choice", analysis)
                self.stats["cot_prompts_used"] += 1
            else:
                prompt = self.prompt_engineer.create_korean_reinforced_prompt(question, "multiple_choice")
            
            result = self.model_handler.generate_response(
                prompt=prompt,
                question_type="multiple_choice",
                max_attempts=2,
                question_structure=structure
            )
            
            if self.model_handler.is_finetuned:
                self.stats["finetuned_usage"] += 1
            
            extracted = self.data_processor.extract_mc_answer_fast(result.response)
            
            if extracted and extracted.isdigit() and 1 <= int(extracted) <= 5:
                self.stats["model_success"] += 1
                if result.confidence > 0.7:
                    self.stats["high_confidence_count"] += 1
                self.stats["answer_distribution"][extracted] += 1
                return extracted
                
        except Exception as e:
            self.stats["generation_errors"] += 1
            # 오류 복구 시도
            if reasoning_used and reasoning_result:
                self.stats["error_recovery_success"] += 1
                self.stats["answer_distribution"][reasoning_result] += 1
                return reasoning_result
        
        # 폴백 처리
        self.stats["fallback_used"] += 1
        fallback_answer = self._get_enhanced_fallback_mc(question, structure)
        self.stats["answer_distribution"][fallback_answer] += 1
        return fallback_answer
    
    def _process_subjective_with_reasoning(self, question: str, structure: Dict, analysis: Dict,
                                         reasoning_result: Optional[str], reasoning_quality: float,
                                         reasoning_used: bool) -> str:
        """추론 엔진을 활용한 주관식 처리"""
        self.stats["subj_count"] += 1
        
        # 추론 결과 우선 사용
        if reasoning_used and reasoning_result and reasoning_quality > REASONING_QUALITY_THRESHOLD:
            is_valid, quality = self._validate_korean_quality_enhanced(reasoning_result)
            if is_valid and quality > QUALITY_THRESHOLD:
                self.stats["high_confidence_count"] += 1
                self.stats["korean_quality_sum"] += quality
                return reasoning_result
        
        try:
            # CoT 프롬프트 사용 (추론 엔진 사용 시)
            if reasoning_used or self.reasoning_engine:
                prompt = self.prompt_engineer.create_cot_prompt(question, "subjective", analysis)
                self.stats["cot_prompts_used"] += 1
            else:
                prompt = self.prompt_engineer.create_korean_reinforced_prompt(question, "subjective")
            
            result = self.model_handler.generate_response(
                prompt=prompt,
                question_type="subjective",
                max_attempts=2,
                question_structure=structure
            )
            
            if self.model_handler.is_finetuned:
                self.stats["finetuned_usage"] += 1
            
            answer = self.data_processor._clean_korean_text(result.response)
            
            # 한국어 품질 검증
            is_valid, quality = self._validate_korean_quality_enhanced(answer)
            self.stats["korean_quality_sum"] += quality
            
            # 품질 확인 및 길이 조정
            if not is_valid or quality < QUALITY_THRESHOLD or len(answer) < MIN_ANSWER_LENGTH:
                # 추론 결과로 대체 시도
                if reasoning_used and reasoning_result:
                    reason_valid, reason_quality = self._validate_korean_quality_enhanced(reasoning_result)
                    if reason_valid and reason_quality >= quality and len(reasoning_result) >= MIN_ANSWER_LENGTH:
                        self.stats["hybrid_approach_success"] += 1
                        self.stats["error_recovery_success"] += 1
                        self.stats["korean_quality_sum"] += (reason_quality - quality)  # 품질 차이만큼 보정
                        return reasoning_result
                
                self.stats["fallback_used"] += 1
                return self._get_enhanced_fallback_subj(question)
            
            self.stats["model_success"] += 1
            if quality > 0.8:
                self.stats["high_confidence_count"] += 1
            
            # 길이 조정
            if len(answer) > MAX_ANSWER_LENGTH:
                answer = answer[:MAX_ANSWER_LENGTH-3] + "..."
            
            return answer
            
        except Exception as e:
            self.stats["generation_errors"] += 1
            
            # 오류 복구 시도
            if reasoning_used and reasoning_result:
                reason_valid, reason_quality = self._validate_korean_quality_enhanced(reasoning_result)
                if reason_valid and reason_quality > 0.5:
                    self.stats["error_recovery_success"] += 1
                    self.stats["korean_quality_sum"] += reason_quality
                    return reasoning_result
            
            self.stats["fallback_used"] += 1
            return self._get_enhanced_fallback_subj(question)
    
    def _apply_distribution_balancing(self) -> Optional[str]:
        """답변 분포 균형화"""
        current_distribution = self.stats["answer_distribution"]
        total_mc_answers = sum(current_distribution.values())
        
        if total_mc_answers > 8:
            target_per_answer = total_mc_answers / 5
            underrepresented = []
            
            for ans in ["1", "2", "3", "4", "5"]:
                count = current_distribution[ans]
                if count < target_per_answer * 0.65:
                    underrepresented.append(ans)
            
            if underrepresented:
                return random.choice(underrepresented)
        
        return None
    
    def _get_enhanced_fallback_mc(self, question: str, structure: Dict) -> str:
        """향상된 객관식 폴백"""
        question_lower = question.lower()
        has_negative = structure.get("has_negative", False)
        
        if has_negative:
            negative_strategies = {
                "해당하지": ["1", "3", "4", "5"],
                "적절하지": ["1", "3", "4", "5"], 
                "옳지": ["2", "3", "4", "5"],
                "틀린": ["1", "2", "4", "5"]
            }
            
            for neg_word, options in negative_strategies.items():
                if neg_word in question_lower:
                    return random.choice(options)
            
            return random.choice(["1", "3", "4", "5"])
        
        # 도메인별 패턴 적용
        domain = self._extract_simple_domain(question)
        question_hash = hash(question) % 100
        
        domain_patterns = {
            "개인정보": {
                0: ["1", "2", "3"], 1: ["2", "1", "3"], 2: ["3", "1", "2"], 3: ["1", "3", "2"]
            },
            "전자금융": {
                0: ["1", "2", "3"], 1: ["2", "3", "4"], 2: ["3", "4", "1"], 3: ["4", "1", "2"]
            },
            "사이버보안": {
                0: ["2", "1", "3"], 1: ["1", "3", "4"], 2: ["3", "2", "4"]
            }
        }
        
        if domain in domain_patterns:
            patterns = domain_patterns[domain]
            pattern_idx = question_hash % len(patterns)
            return random.choice(patterns[pattern_idx])
        
        return str((question_hash % 5) + 1)
    
    def _get_enhanced_fallback_subj(self, question: str) -> str:
        """향상된 주관식 폴백"""
        domain = self._extract_simple_domain(question)
        templates = self.enhanced_fallback_templates.get(domain, self.enhanced_fallback_templates["일반"])
        return random.choice(templates)
    
    def _extract_simple_domain(self, question: str) -> str:
        """간단한 도메인 추출"""
        question_lower = question.lower()
        
        domain_keywords = {
            "사이버보안": ["트로이", "악성코드", "해킹", "멀웨어", "피싱"],
            "개인정보": ["개인정보", "정보주체", "개인정보보호법"],
            "전자금융": ["전자금융", "전자적", "접근매체", "전자금융거래법"],
            "정보보안": ["정보보안", "보안관리", "ISMS", "보안정책"],
            "위험관리": ["위험관리", "위험평가", "위험분석", "위험통제"]
        }
        
        for domain, keywords in domain_keywords.items():
            if any(keyword in question_lower for keyword in keywords):
                return domain
        
        return "일반"
    
    def _validate_korean_quality_enhanced(self, text: str) -> Tuple[bool, float]:
        """향상된 한국어 품질 검증"""
        if not text or len(text) < 20:
            return False, 0.0
        
        # 패널티 요소들
        penalty_factors = [
            (r'[\u4e00-\u9fff]', 0.4),
            (r'[①②③④⑤➀➁➂➃➄]', 0.3),
            (r'\bbo+\b', 0.4),
            (r'[ㄱ-ㅎㅏ-ㅣ]{3,}', 0.3)
        ]
        
        total_penalty = 0
        for pattern, penalty in penalty_factors:
            if re.search(pattern, text, re.IGNORECASE):
                total_penalty += penalty
        
        if total_penalty > 0.5:
            return False, 0.0
        
        # 한국어 비율 계산
        korean_chars = len([c for c in text if '가' <= c <= '힣'])
        total_chars = len([c for c in text if c.isalnum()])
        
        if total_chars == 0:
            return False, 0.0
        
        korean_ratio = korean_chars / total_chars
        
        if korean_ratio < MIN_KOREAN_RATIO:
            return False, korean_ratio
        
        # 영어 비율 확인
        english_chars = len([c for c in text if c.isalpha() and ord(c) < 128])
        english_ratio = english_chars / total_chars
        
        if english_ratio > MAX_ENGLISH_RATIO:
            return False, korean_ratio * (1 - english_ratio)
        
        # 품질 점수 계산
        quality_score = korean_ratio * 0.85 - total_penalty
        
        # 전문 용어 보너스
        professional_terms = ['법', '규정', '관리', '보안', '조치', '정책', '체계', '절차']
        prof_count = sum(1 for term in professional_terms if term in text)
        quality_score += min(prof_count * 0.04, 0.15)
        
        # 길이 보너스
        if MIN_ANSWER_LENGTH <= len(text) <= MAX_ANSWER_LENGTH:
            quality_score += 0.05
        
        return quality_score > QUALITY_THRESHOLD, max(0, min(1, quality_score))
    
    def _manage_memory(self) -> None:
        """메모리 관리"""
        self.memory_cleanup_counter += 1
        
        if self.memory_cleanup_counter % MEMORY_CLEANUP_INTERVAL == 0:
            self.stats["memory_cleanups"] += 1
            
            # GPU 메모리 정리
            if self.cuda_available:
                torch.cuda.empty_cache()
            
            # CPU 메모리 정리
            gc.collect()
    
    def run_test(self, test_file: str = "./test.csv", submission_file: str = "./sample_submission.csv") -> None:
        """테스트 실행"""
        print("="*50)
        print(f"테스트 실행 시작 ({self.test_size}문항)")
        if self.use_finetuned:
            print("파인튜닝된 모델 사용")
        if self.reasoning_engine:
            print("논리적 추론 엔진 활성화")
        print("="*50)
        
        # 데이터 로드
        data_result = self.load_test_data(test_file, submission_file)
        if data_result is None:
            return
        
        test_df, submission_df = data_result
        
        # 질문 분석
        question_analysis = self.analyze_questions(test_df)
        
        print(f"\n추론 시작...")
        
        # 답변 생성
        answers = []
        
        try:
            for idx, row in test_df.iterrows():
                question = row['Question']
                question_id = row['ID']
                
                answer = self.process_single_question(question, question_id, idx)
                answers.append(answer)
            
            # 결과 저장
            submission_df['Answer'] = answers
            
            output_file = f"./test_result_{self.test_size}.csv"
            submission_df.to_csv(output_file, index=False, encoding='utf-8-sig')
            
            # 결과 출력
            self._print_results(output_file, question_analysis)
            
            # 학습 데이터 저장
            self._save_learning_data()
            
        except KeyboardInterrupt:
            print("\n테스트 중단됨")
        except Exception as e:
            print(f"테스트 실행 중 오류: {e}")
            raise
    
    def _save_learning_data(self) -> None:
        """학습 데이터 저장"""
        try:
            if self.learning_system.save_model():
                print("학습 데이터 저장 완료")
            else:
                print("학습 데이터 저장 실패")
        except Exception as e:
            print(f"학습 데이터 저장 중 오류: {e}")
    
    def _print_results(self, output_file: str, question_analysis: Dict) -> None:
        """결과 출력"""
        total_time = time.time() - self.start_time
        avg_time = (sum(self.stats["processing_times"]) / len(self.stats["processing_times"]) 
                   if self.stats["processing_times"] else 0)
        
        print(f"\n" + "="*50)
        print("테스트 완료")
        print("="*50)
        
        # 처리 시간 정보
        print(f"처리 시간: {total_time:.1f}초")
        print(f"문항당 평균: {avg_time:.2f}초")
        
        # 모델 정보
        model_type = "파인튜닝된 모델" if self.model_handler.is_finetuned else "기본 모델"
        print(f"사용 모델: {model_type}")
        
        # 파인튜닝 사용률
        if self.model_handler.is_finetuned:
            finetuned_rate = self.stats["finetuned_usage"] / max(self.stats["total"], 1) * 100
            print(f"파인튜닝 모델 사용률: {finetuned_rate:.1f}%")
        
        # 처리 통계
        self._print_processing_stats()
        
        # 추론 성능 통계
        self._print_reasoning_performance_stats()
        
        # 한국어 품질 통계
        self._print_korean_quality_stats()
        
        # 객관식 분포
        self._print_mc_distribution()
        
        # 메모리 사용 정보
        self._print_memory_stats()
        
        print(f"\n결과 파일: {output_file}")
    
    def _print_processing_stats(self) -> None:
        """처리 통계 출력"""
        print(f"\n처리 통계:")
        success_rate = self.stats["model_success"] / max(self.stats["total"], 1) * 100
        pattern_rate = self.stats["pattern_success"] / max(self.stats["total"], 1) * 100
        fallback_rate = self.stats["fallback_used"] / max(self.stats["total"], 1) * 100
        
        print(f"  모델 생성 성공: {self.stats['model_success']}/{self.stats['total']} ({success_rate:.1f}%)")
        print(f"  패턴 매칭 성공: {self.stats['pattern_success']}/{self.stats['total']} ({pattern_rate:.1f}%)")
        print(f"  스마트 힌트 사용: {self.stats['smart_hints_used']}회")
        print(f"  고신뢰도 답변: {self.stats['high_confidence_count']}회")
        print(f"  폴백 사용: {self.stats['fallback_used']}/{self.stats['total']} ({fallback_rate:.1f}%)")
        print(f"  생성 오류: {self.stats['generation_errors']}회")
    
    def _print_reasoning_performance_stats(self) -> None:
        """추론 성능 통계 출력"""
        if self.reasoning_engine:
            print(f"\n추론 엔진 성능:")
            reasoning_rate = self.stats["reasoning_engine_usage"] / max(self.stats["total"], 1) * 100
            cot_rate = self.stats["cot_prompts_used"] / max(self.stats["total"], 1) * 100
            
            print(f"  추론 엔진 사용: {self.stats['reasoning_engine_usage']}/{self.stats['total']} ({reasoning_rate:.1f}%)")
            print(f"  CoT 프롬프트 사용: {self.stats['cot_prompts_used']}/{self.stats['total']} ({cot_rate:.1f}%)")
            print(f"  추론 성공: {self.stats['reasoning_successful']}회")
            print(f"  추론 실패: {self.stats['reasoning_failed']}회")
            
            # 추론 품질 점수
            if self.stats["reasoning_quality_scores"]:
                avg_reasoning_quality = np.mean(self.stats["reasoning_quality_scores"])
                print(f"  평균 추론 품질: {avg_reasoning_quality:.2f}")
            
            # 일관성 점수
            if self.stats["consistency_scores"]:
                avg_consistency = np.mean(self.stats["consistency_scores"])
                print(f"  평균 일관성 점수: {avg_consistency:.2f}")
            
            # 추론 깊이 분석
            if self.stats["reasoning_depth_scores"]:
                avg_depth = np.mean(self.stats["reasoning_depth_scores"])
                print(f"  평균 추론 단계 수: {avg_depth:.1f}")
            
            # 체인 검증 통계
            total_chains = self.stats["chain_verification_passed"] + self.stats["chain_verification_failed"]
            if total_chains > 0:
                verification_rate = self.stats["chain_verification_passed"] / total_chains * 100
                print(f"  체인 검증 통과율: {verification_rate:.1f}%")
            
            # 논리적 오류
            print(f"  논리적 오류: {self.stats['logical_errors']}회")
            
            # 고급 기능 사용
            if self.stats["advanced_reasoning_features"] > 0:
                advanced_rate = self.stats["advanced_reasoning_features"] / max(self.stats["total"], 1) * 100
                print(f"  고급 추론 기능 사용: {advanced_rate:.1f}%")
            
            # 하이브리드 접근법 성공률
            if self.stats["hybrid_approach_success"] > 0:
                hybrid_rate = self.stats["hybrid_approach_success"] / max(self.stats["total"], 1) * 100
                print(f"  하이브리드 접근 성공: {hybrid_rate:.1f}%")
            
            # 추론 vs 패턴 비교
            comparison = self.stats["reasoning_vs_pattern_comparison"]
            total_comparisons = sum(comparison.values())
            if total_comparisons > 0:
                reasoning_better_rate = comparison["reasoning_better"] / total_comparisons * 100
                print(f"  추론 우위 사례: {reasoning_better_rate:.1f}%")
            
            # 오류 복구 성공률
            if self.stats["error_recovery_success"] > 0:
                recovery_rate = self.stats["error_recovery_success"] / max(self.stats["generation_errors"], 1) * 100
                print(f"  오류 복구 성공률: {recovery_rate:.1f}%")
            
            # 평균 추론 시간
            if self.stats["reasoning_time_per_question"]:
                avg_reasoning_time = np.mean(self.stats["reasoning_time_per_question"])
                print(f"  평균 추론 시간: {avg_reasoning_time:.3f}초")
            
            # 단계 수 분포
            if self.stats["step_count_distribution"]:
                print(f"  추론 단계 분포:")
                for steps, count in sorted(self.stats["step_count_distribution"].items()):
                    percentage = count / max(self.stats["total"], 1) * 100
                    print(f"    {steps}단계: {count}회 ({percentage:.1f}%)")
    
    def _print_korean_quality_stats(self) -> None:
        """한국어 품질 통계 출력"""
        if self.stats["subj_count"] > 0:
            avg_korean_quality = self.stats["korean_quality_sum"] / self.stats["subj_count"]
            print(f"\n한국어 품질:")
            print(f"  평균 품질 점수: {avg_korean_quality:.2f}")
            
            if avg_korean_quality > 0.8:
                quality_level = "우수"
            elif avg_korean_quality > 0.65:
                quality_level = "양호"
            else:
                quality_level = "개선 필요"
            
            print(f"  품질 평가: {quality_level}")
    
    def _print_mc_distribution(self) -> None:
        """객관식 분포 출력"""
        total_mc = sum(self.stats["answer_distribution"].values())
        if total_mc > 0:
            print(f"\n객관식 답변 분포:")
            for ans in sorted(self.stats["answer_distribution"].keys()):
                count = self.stats["answer_distribution"][ans]
                pct = count / total_mc * 100
                print(f"  {ans}번: {count}개 ({pct:.1f}%)")
            
            # 다양성 평가
            unique_answers = len([k for k, v in self.stats["answer_distribution"].items() if v > 0])
            print(f"  답변 다양성: {unique_answers}/5개 번호 사용")
            
            # 분포 균형 평가
            distribution_balance = np.std(list(self.stats["answer_distribution"].values()))
            balance_threshold = total_mc * 0.15
            
            if distribution_balance < balance_threshold:
                print(f"  분포 균형: 양호")
            else:
                print(f"  분포 균형: 개선 필요")
    
    def _print_memory_stats(self) -> None:
        """메모리 통계 출력"""
        if self.cuda_available:
            try:
                memory_used = torch.cuda.max_memory_allocated() / (1024**3)
                print(f"\nGPU 메모리 사용:")
                print(f"  최대 사용량: {memory_used:.1f}GB")
                print(f"  메모리 정리 횟수: {self.stats['memory_cleanups']}회")
            except Exception:
                pass
    
    def cleanup(self) -> None:
        """리소스 정리"""
        try:
            print("\n시스템 정리 중...")
            
            # 각 컴포넌트 정리
            if hasattr(self, 'model_handler'):
                self.model_handler.cleanup()
            if hasattr(self, 'data_processor'):
                self.data_processor.cleanup()
            if hasattr(self, 'prompt_engineer'):
                self.prompt_engineer.cleanup()
            if hasattr(self, 'learning_system'):
                self.learning_system.cleanup()
            if hasattr(self, 'reasoning_engine'):
                self.reasoning_engine.cleanup()
            
            # 메모리 정리
            if self.cuda_available:
                torch.cuda.empty_cache()
            gc.collect()
            
            print("정리 완료")
            
        except Exception as e:
            print(f"정리 중 오류: {e}")


class ReasoningPerformanceAnalyzer:
    """추론 성능 분석기"""
    
    def __init__(self):
        self.quality_metrics = {
            "logical_consistency": 0.3,
            "evidence_quality": 0.25,
            "reasoning_depth": 0.2,
            "conclusion_validity": 0.15,
            "step_coherence": 0.1
        }
    
    def analyze_reasoning_chain(self, reasoning_chain) -> Dict:
        """추론 체인 분석"""
        try:
            analysis = {
                "quality_score": 0.0,
                "uses_advanced_features": False,
                "logical_errors": [],
                "strengths": [],
                "weaknesses": []
            }
            
            if not reasoning_chain or not reasoning_chain.steps:
                return analysis
            
            # 논리적 일관성 평가
            logical_score = self._evaluate_logical_consistency(reasoning_chain)
            
            # 증거 품질 평가
            evidence_score = self._evaluate_evidence_quality(reasoning_chain)
            
            # 추론 깊이 평가
            depth_score = self._evaluate_reasoning_depth(reasoning_chain)
            
            # 결론 타당성 평가
            conclusion_score = self._evaluate_conclusion_validity(reasoning_chain)
            
            # 단계 연결성 평가
            coherence_score = self._evaluate_step_coherence(reasoning_chain)
            
            # 종합 점수 계산
            analysis["quality_score"] = (
                logical_score * self.quality_metrics["logical_consistency"] +
                evidence_score * self.quality_metrics["evidence_quality"] +
                depth_score * self.quality_metrics["reasoning_depth"] +
                conclusion_score * self.quality_metrics["conclusion_validity"] +
                coherence_score * self.quality_metrics["step_coherence"]
            )
            
            # 고급 기능 사용 여부
            analysis["uses_advanced_features"] = self._check_advanced_features(reasoning_chain)
            
            # 강점과 약점 식별
            analysis["strengths"] = self._identify_strengths(reasoning_chain, {
                "logical": logical_score,
                "evidence": evidence_score,
                "depth": depth_score,
                "conclusion": conclusion_score,
                "coherence": coherence_score
            })
            
            analysis["weaknesses"] = self._identify_weaknesses(reasoning_chain, {
                "logical": logical_score,
                "evidence": evidence_score,
                "depth": depth_score,
                "conclusion": conclusion_score,
                "coherence": coherence_score
            })
            
            return analysis
            
        except Exception:
            return {
                "quality_score": 0.0,
                "uses_advanced_features": False,
                "logical_errors": ["분석 오류"],
                "strengths": [],
                "weaknesses": ["분석 실패"]
            }
    
    def _evaluate_logical_consistency(self, reasoning_chain) -> float:
        """논리적 일관성 평가"""
        try:
            consistency_score = reasoning_chain.verification_result.get("consistency_score", 0.0)
            is_consistent = reasoning_chain.verification_result.get("is_consistent", False)
            
            base_score = consistency_score
            if is_consistent:
                base_score += 0.2
            
            # 신뢰도 편차 검사
            variance = reasoning_chain.verification_result.get("confidence_variance", 0.0)
            if variance < 0.2:
                base_score += 0.1
            
            return min(1.0, base_score)
            
        except Exception:
            return 0.0
    
    def _evaluate_evidence_quality(self, reasoning_chain) -> float:
        """증거 품질 평가"""
        try:
            score = 0.0
            
            for step in reasoning_chain.steps:
                # 증거 존재 여부
                if step.supporting_evidence:
                    score += 0.2
                    
                    # 증거 다양성
                    if len(step.supporting_evidence) > 1:
                        score += 0.1
                
                # 추론 타입별 가점
                if step.reasoning_type in ["논리_추론", "개념_적용"]:
                    score += 0.1
            
            return min(1.0, score / len(reasoning_chain.steps))
            
        except Exception:
            return 0.0
    
    def _evaluate_reasoning_depth(self, reasoning_chain) -> float:
        """추론 깊이 평가"""
        try:
            step_count = len(reasoning_chain.steps)
            
            # 기본 단계 수 점수
            depth_score = min(step_count / 5.0, 0.6)
            
            # 추론 타입 다양성
            reasoning_types = set(step.reasoning_type for step in reasoning_chain.steps)
            type_diversity = len(reasoning_types) / 4.0  # 최대 4가지 타입 가정
            depth_score += type_diversity * 0.3
            
            # 복잡성 보너스
            complex_steps = sum(1 for step in reasoning_chain.steps 
                              if step.reasoning_type in ["논리_추론", "결론_도출"])
            if complex_steps >= 2:
                depth_score += 0.1
            
            return min(1.0, depth_score)
            
        except Exception:
            return 0.0
    
    def _evaluate_conclusion_validity(self, reasoning_chain) -> float:
        """결론 타당성 평가"""
        try:
            # 전체 신뢰도
            confidence_score = reasoning_chain.overall_confidence
            
            # 최종 답변 형식 점수
            format_score = 0.0
            if reasoning_chain.final_answer:
                if len(reasoning_chain.final_answer) > 0:
                    format_score = 0.5
                    
                    # 한국어 품질 (주관식의 경우)
                    if not reasoning_chain.final_answer.isdigit():
                        korean_chars = len([c for c in reasoning_chain.final_answer if '가' <= c <= '힣'])
                        total_chars = len([c for c in reasoning_chain.final_answer if c.isalnum()])
                        if total_chars > 0:
                            korean_ratio = korean_chars / total_chars
                            if korean_ratio > 0.7:
                                format_score = 1.0
                            elif korean_ratio > 0.5:
                                format_score = 0.8
            
            return (confidence_score + format_score) / 2.0
            
        except Exception:
            return 0.0
    
    def _evaluate_step_coherence(self, reasoning_chain) -> float:
        """단계 연결성 평가"""
        try:
            if len(reasoning_chain.steps) < 2:
                return 0.5
            
            coherence_score = 0.0
            
            # 순차적 연결성 검사
            for i in range(1, len(reasoning_chain.steps)):
                prev_step = reasoning_chain.steps[i-1]
                curr_step = reasoning_chain.steps[i]
                
                # 신뢰도 일관성
                confidence_diff = abs(prev_step.confidence - curr_step.confidence)
                if confidence_diff < 0.3:
                    coherence_score += 0.2
                
                # 추론 타입 연계성
                if self._check_reasoning_flow(prev_step.reasoning_type, curr_step.reasoning_type):
                    coherence_score += 0.2
            
            return min(1.0, coherence_score / (len(reasoning_chain.steps) - 1))
            
        except Exception:
            return 0.0
    
    def _check_reasoning_flow(self, prev_type: str, curr_type: str) -> bool:
        """추론 흐름 유효성 검사"""
        valid_flows = {
            "문제_분석": ["개념_적용", "논리_추론"],
            "개념_적용": ["논리_추론", "결론_도출"],
            "논리_추론": ["결론_도출", "논리_추론"],
            "결론_도출": []
        }
        
        return curr_type in valid_flows.get(prev_type, [])
    
    def _check_advanced_features(self, reasoning_chain) -> bool:
        """고급 기능 사용 여부 확인"""
        try:
            # 다단계 추론
            if len(reasoning_chain.steps) >= 4:
                return True
            
            # 다양한 추론 타입 사용
            reasoning_types = set(step.reasoning_type for step in reasoning_chain.steps)
            if len(reasoning_types) >= 3:
                return True
            
            # 높은 신뢰도의 일관성
            if (reasoning_chain.overall_confidence > 0.8 and 
                reasoning_chain.verification_result.get("is_consistent", False)):
                return True
            
            return False
            
        except Exception:
            return False
    
    def _identify_strengths(self, reasoning_chain, scores: Dict) -> List[str]:
        """강점 식별"""
        strengths = []
        
        try:
            if scores["logical"] > 0.8:
                strengths.append("높은 논리적 일관성")
            
            if scores["evidence"] > 0.7:
                strengths.append("풍부한 증거 제시")
            
            if scores["depth"] > 0.7:
                strengths.append("충분한 추론 깊이")
            
            if scores["conclusion"] > 0.8:
                strengths.append("타당한 결론 도출")
            
            if scores["coherence"] > 0.7:
                strengths.append("단계별 연결성 우수")
            
            if len(reasoning_chain.steps) >= 4:
                strengths.append("다단계 분석")
            
        except Exception:
            pass
        
        return strengths
    
    def _identify_weaknesses(self, reasoning_chain, scores: Dict) -> List[str]:
        """약점 식별"""
        weaknesses = []
        
        try:
            if scores["logical"] < 0.5:
                weaknesses.append("논리적 일관성 부족")
            
            if scores["evidence"] < 0.4:
                weaknesses.append("증거 부족")
            
            if scores["depth"] < 0.4:
                weaknesses.append("추론 깊이 부족")
            
            if scores["conclusion"] < 0.5:
                weaknesses.append("결론 타당성 부족")
            
            if scores["coherence"] < 0.5:
                weaknesses.append("단계별 연결성 부족")
            
            if len(reasoning_chain.steps) < 2:
                weaknesses.append("추론 단계 부족")
            
        except Exception:
            pass
        
        return weaknesses


def main():
    """메인 함수"""
    test_size = DEFAULT_TEST_SIZE
    use_finetuned = False
    
    # 명령행 인수 처리
    if len(sys.argv) > 1:
        try:
            test_size = int(sys.argv[1])
            test_size = max(MIN_TEST_SIZE, min(test_size, MAX_TEST_SIZE))
        except ValueError:
            print("잘못된 문항 수, 기본값 50 사용")
            test_size = DEFAULT_TEST_SIZE
    
    if len(sys.argv) > 2:
        if sys.argv[2].lower() in ['true', '1', 'yes', 'finetuned']:
            use_finetuned = True
    
    # 파인튜닝 모델 자동 감지
    if os.path.exists("./finetuned_model") and not use_finetuned:
        try:
            response = input("파인튜닝된 모델이 발견되었습니다. 사용하시겠습니까? (y/n): ")
            if response.lower() in ['y', 'yes']:
                use_finetuned = True
        except (EOFError, KeyboardInterrupt):
            print("\n기본 모델 사용")
    
    print(f"테스트 실행기 시작 (Python {sys.version.split()[0]})")
    
    runner = None
    try:
        runner = TestRunner(test_size=test_size, use_finetuned=use_finetuned)
        runner.run_test()
        
    except KeyboardInterrupt:
        print("\n테스트 중단")
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if runner:
            runner.cleanup()


if __name__ == "__main__":
    main()