# inference.py

"""
메인 추론 시스템
- 금융보안 객관식/주관식 문제 추론
- 학습 시스템 통합 관리
- 한국어 답변 생성 및 검증
- 오프라인 환경 대응
"""

import os
import sys
import time
import re
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
import warnings
import gc
from typing import List, Dict, Tuple, Optional
import numpy as np
warnings.filterwarnings("ignore")

current_dir = Path(__file__).parent.absolute()
sys.path.append(str(current_dir))

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from transformers.utils import logging
logging.set_verbosity_error()

from model_handler import ModelHandler
from data_processor import DataProcessor
from prompt_engineering import PromptEngineer
from learning_system import LearningSystem

class FinancialAIInference:
    
    def __init__(self, enable_learning: bool = True, verbose: bool = False):
        self.start_time = time.time()
        self.enable_learning = enable_learning
        self.verbose = verbose
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.set_per_process_memory_fraction(0.95)
        
        print("시스템 초기화 중...")
        
        self.model_handler = ModelHandler(
            model_name="upstage/SOLAR-10.7B-Instruct-v1.0",
            device="cuda" if torch.cuda.is_available() else "cpu",
            load_in_4bit=True,
            max_memory_gb=22,
            verbose=self.verbose
        )
        
        self.data_processor = DataProcessor()
        self.prompt_engineer = PromptEngineer()
        
        if self.enable_learning:
            self.learning_system = LearningSystem(debug_mode=self.verbose)
            self._load_existing_learning_data()
        
        self.stats = {
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
            "memory_optimizations": 0,
            "adaptive_processing": 0,
            "quality_improvements": 0
        }
        
        self.answer_cache = {}
        self.pattern_analysis_cache = {}
        self.structure_cache = {}
        self.max_cache_size = 1000
        
        self.processing_modes = {
            "lightning": {"max_attempts": 1, "cache_priority": True, "pattern_focus": True},
            "fast": {"max_attempts": 1, "cache_priority": True, "pattern_focus": False},
            "normal": {"max_attempts": 2, "cache_priority": False, "pattern_focus": False},
            "careful": {"max_attempts": 2, "cache_priority": False, "pattern_focus": False},
            "deep": {"max_attempts": 3, "cache_priority": False, "pattern_focus": False}
        }
        
        self.memory_management = {
            "cleanup_frequency": 25,
            "cache_optimization_frequency": 50,
            "deep_cleanup_frequency": 100,
            "processed_count": 0
        }
        
        if torch.cuda.is_available():
            self.gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        else:
            self.gpu_memory_total = 0
        
        print("초기화 완료")
    
    def _load_existing_learning_data(self) -> None:
        try:
            if self.learning_system.load_model():
                if self.verbose:
                    print(f"학습 데이터 로드: {len(self.learning_system.learning_history)}개")
        except Exception as e:
            if self.verbose:
                print(f"학습 데이터 로드 오류: {e}")
    
    def _validate_korean_quality_enhanced(self, text: str, question_type: str) -> Tuple[bool, float]:
        if question_type == "multiple_choice":
            if re.match(r'^[1-5]$', text.strip()):
                return True, 1.0
            if re.search(r'[1-5]', text):
                return True, 0.8
            return False, 0.0
        
        if not text or len(text.strip()) < 15:
            return False, 0.0
        
        quality_checks = {
            "no_chinese": not bool(re.search(r'[\u4e00-\u9fff]', text)),
            "no_japanese": not bool(re.search(r'[\u3040-\u309f\u30a0-\u30ff]', text)),
            "no_russian": not bool(re.search(r'[а-яё]', text.lower())),
            "has_korean": bool(re.search(r'[가-힣]', text)),
            "appropriate_length": 20 <= len(text) <= 800
        }
        
        if not all([quality_checks["no_chinese"], quality_checks["no_japanese"], 
                   quality_checks["no_russian"], quality_checks["has_korean"]]):
            return False, 0.1
        
        korean_chars = len(re.findall(r'[가-힣]', text))
        total_chars = len(re.sub(r'[^\w]', '', text))
        
        if total_chars == 0:
            return False, 0.0
        
        korean_ratio = korean_chars / total_chars
        english_chars = len(re.findall(r'[A-Za-z]', text))
        english_ratio = english_chars / total_chars
        
        if korean_ratio < 0.4:
            return False, korean_ratio
        
        if english_ratio > 0.3:
            return False, 1 - english_ratio
        
        quality_score = korean_ratio * 0.7
        
        professional_terms = [
            '법', '규정', '조치', '관리', '보안', '체계', '정책', '절차', '방안', '시스템',
            '개인정보', '전자금융', '위험관리', '암호화', '접근통제', '정보보호'
        ]
        prof_count = sum(1 for term in professional_terms if term in text)
        quality_score += min(prof_count * 0.06, 0.25)
        
        if quality_checks["appropriate_length"]:
            quality_score += 0.15
        
        structure_markers = ['첫째', '둘째', '따라서', '그러므로', '결론적으로', '또한']
        if any(marker in text for marker in structure_markers):
            quality_score += 0.1
        
        final_quality = max(0, min(1, quality_score))
        
        return final_quality > 0.5, final_quality
    
    def _get_domain_specific_fallback_enhanced(self, question: str, question_type: str, 
                                             structure: Dict = None) -> str:
        if question_type == "multiple_choice":
            if self.enable_learning:
                hint, conf = self.learning_system.get_smart_answer_hint(question, structure or {})
                
                if conf > 0.4:
                    self.stats["smart_hints_used"] += 1
                    return hint
            
            question_hash = hash(question) % 5 + 1
            base_answers = ["2", "3", "1", "4", "5"]
            return str(base_answers[question_hash % 5])
        
        question_lower = question.lower()
        
        domain_responses = {
            "사이버보안": {
                "keywords": ["트로이", "악성코드", "RAT", "원격", "탐지"],
                "response": "트로이 목마는 정상 프로그램으로 위장한 악성코드로, 원격 접근 트로이 목마는 공격자가 감염된 시스템을 원격으로 제어할 수 있게 합니다. 주요 탐지 지표로는 비정상적인 네트워크 연결, 시스템 리소스 사용 증가, 알 수 없는 프로세스 실행, 방화벽 규칙 변경, 레지스트리 변경, 이상한 파일 생성 등이 있습니다."
            },
            "개인정보보호": {
                "keywords": ["개인정보", "정보주체", "유출", "보호법"],
                "response": "개인정보보호법에 따라 개인정보의 안전한 관리와 정보주체의 권리 보호를 위한 체계적인 조치가 필요합니다. 개인정보 처리 시 수집, 이용, 제공의 최소화 원칙을 준수하고 안전성 확보조치를 통해 기술적, 관리적, 물리적 보호조치를 구현해야 합니다."
            },
            "전자금융": {
                "keywords": ["전자금융", "전자적", "접근매체", "거래법"],
                "response": "전자금융거래법에 따라 전자적 장치를 통한 금융거래의 안전성을 확보하고 이용자를 보호해야 합니다. 접근매체의 안전한 관리, 거래내역 통지, 오류정정 절차, 손해배상 체계를 구축하여 전자금융거래의 신뢰성을 보장해야 합니다."
            },
            "정보보안": {
                "keywords": ["정보보안", "관리체계", "ISMS", "보안정책"],
                "response": "정보보안 관리체계를 통해 체계적인 보안 관리와 지속적인 위험 평가를 수행해야 합니다. 정보보안 정책 수립, 조직 구성, 자산 관리, 접근 통제, 시스템 보안, 사고 대응 등 전 영역에 걸친 통합적 보안 관리가 필요합니다."
            },
            "위험관리": {
                "keywords": ["위험", "관리", "평가", "대응"],
                "response": "위험관리는 조직의 목표 달성에 영향을 미칠 수 있는 위험을 체계적으로 식별, 분석, 평가하고 적절한 대응방안을 수립하여 관리하는 과정입니다. 위험 식별, 분석, 평가, 대응, 모니터링의 단계별 활동을 통해 지속적인 위험 관리 체계를 구축해야 합니다."
            },
            "관리체계": {
                "keywords": ["관리체계", "정책", "수립", "운영", "경영진"],
                "response": "관리체계 수립과 운영에서는 최고경영진의 참여와 지원이 가장 중요하며, 명확한 정책 수립과 책임자 지정, 적절한 자원 할당이 필요합니다. 정보보호 및 개인정보보호 정책의 제정과 개정을 통해 체계적인 관리 기반을 마련하고 지속적인 개선을 수행해야 합니다."
            },
            "재해복구": {
                "keywords": ["재해", "복구", "비상", "백업", "BCP"],
                "response": "재해복구계획은 재해 발생 시 핵심 업무를 신속하게 복구하기 위한 체계적인 계획입니다. 복구목표시간과 복구목표시점을 설정하고, 백업 및 복구 절차를 수립하며, 정기적인 모의훈련을 통해 실효성을 검증해야 합니다."
            },
            "암호화": {
                "keywords": ["암호화", "복호화", "키", "해시", "인증서"],
                "response": "암호화는 정보의 기밀성과 무결성을 보장하기 위한 핵심 보안 기술입니다. 대칭키 암호화와 공개키 암호화를 적절히 활용하고, 안전한 키 관리 체계를 구축해야 합니다. 중요 정보는 전송 구간과 저장 시 모두 암호화하여 보호해야 합니다."
            }
        }
        
        for domain, config in domain_responses.items():
            if any(keyword in question_lower for keyword in config["keywords"]):
                return config["response"]
        
        return "관련 법령과 규정에 따라 체계적인 보안 관리 방안을 수립하고, 지속적인 모니터링과 개선을 통해 안전성을 확보해야 합니다."
    
    def _apply_pattern_based_answer_enhanced(self, question: str, structure: Dict) -> Tuple[Optional[str], float]:
        cache_key = hash(question[:100])
        if cache_key in self.pattern_analysis_cache:
            self.stats["cache_hits"] += 1
            return self.pattern_analysis_cache[cache_key]
        
        if not self.enable_learning:
            return None, 0
        
        hint_answer, hint_confidence = self.learning_system.get_smart_answer_hint(question, structure)
        
        difficulty = self.learning_system.evaluate_question_difficulty(question, structure)
        confidence_threshold = 0.65 if difficulty.score < 0.5 else 0.55
        
        if hint_confidence > confidence_threshold:
            self.stats["pattern_based_answers"] += 1
            result = (hint_answer, hint_confidence)
        else:
            result = (None, 0)
        
        if len(self.pattern_analysis_cache) >= self.max_cache_size // 2:
            oldest_keys = list(self.pattern_analysis_cache.keys())[:self.max_cache_size // 4]
            for key in oldest_keys:
                del self.pattern_analysis_cache[key]
        
        self.pattern_analysis_cache[cache_key] = result
        return result
    
    def _select_processing_mode(self, structure: Dict, current_load: int) -> str:
        complexity = structure.get("complexity_score", 0.5)
        choice_count = structure.get("choice_count", 0)
        
        if current_load > 400:
            return "lightning"
        elif current_load > 300:
            return "fast"
        elif complexity < 0.3 and choice_count >= 3:
            return "fast"
        elif complexity > 0.7 or structure.get("has_negative", False):
            return "careful"
        elif len(structure.get("technical_terms", [])) > 3:
            return "deep"
        else:
            return "normal"
    
    def process_question(self, question: str, question_id: str, idx: int) -> str:
        try:
            cache_key = hash(question[:200])
            if cache_key in self.answer_cache:
                self.stats["cache_hits"] += 1
                return self.answer_cache[cache_key]
            
            structure = self.data_processor.analyze_question_structure(question)
            analysis = self.prompt_engineer.knowledge_base.analyze_question(question)
            
            is_mc = structure["question_type"] == "multiple_choice"
            is_subjective = structure["question_type"] == "subjective"
            
            self._debug_log(f"문제 {idx}: 유형={structure['question_type']}, 객관식={is_mc}, 주관식={is_subjective}")
            
            processing_mode = self._select_processing_mode(structure, self.stats["total"])
            mode_config = self.processing_modes[processing_mode]
            
            if processing_mode in ["lightning", "fast"]:
                self.stats["adaptive_processing"] += 1
            
            difficulty = self.learning_system.evaluate_question_difficulty(question, structure) if self.enable_learning else None
            
            if self.enable_learning and mode_config.get("pattern_focus", False):
                learned_answer, learned_confidence = self.learning_system.predict_with_patterns(
                    question, structure["question_type"]
                )
                
                if learned_confidence > 0.75:
                    is_valid, quality = self._validate_korean_quality_enhanced(learned_answer, structure["question_type"])
                    if is_valid:
                        self.answer_cache[cache_key] = learned_answer
                        self.stats["high_confidence_answers"] += 1
                        return learned_answer
            
            if is_mc:
                pattern_answer, pattern_conf = self._apply_pattern_based_answer_enhanced(question, structure)
                
                if pattern_answer and pattern_conf > 0.65:
                    self.stats["smart_hints_used"] += 1
                    self.stats["high_confidence_answers"] += 1
                    self.answer_cache[cache_key] = pattern_answer
                    return pattern_answer
                
                if processing_mode in ["lightning", "fast"] and pattern_answer and pattern_conf > 0.45:
                    self.stats["smart_hints_used"] += 1
                    self.answer_cache[cache_key] = pattern_answer
                    return pattern_answer
                
                prompt = self.prompt_engineer.create_korean_reinforced_prompt(
                    question, structure["question_type"]
                )
                
                max_attempts = mode_config["max_attempts"]
                
                result = self.model_handler.generate_response(
                    prompt=prompt,
                    question_type=structure["question_type"],
                    max_attempts=max_attempts
                )
                
                extracted = self.data_processor.extract_mc_answer_fast(result.response)
                
                if extracted and extracted.isdigit() and 1 <= int(extracted) <= 5:
                    self.stats["model_generation_success"] += 1
                    self.stats["pattern_extraction_success"] += 1
                    answer = extracted
                    
                    if result.confidence > 0.7:
                        self.stats["high_confidence_answers"] += 1
                else:
                    if pattern_answer and pattern_conf > 0.3:
                        self.stats["smart_hints_used"] += 1
                        answer = pattern_answer
                    else:
                        self.stats["fallback_used"] += 1
                        answer = self._get_domain_specific_fallback_enhanced(question, "multiple_choice", structure)
            
            elif is_subjective:
                prompt = self.prompt_engineer.create_korean_reinforced_prompt(
                    question, structure["question_type"]
                )
                
                max_attempts = mode_config["max_attempts"]
                
                result = self.model_handler.generate_response(
                    prompt=prompt,
                    question_type=structure["question_type"],
                    max_attempts=max_attempts
                )
                
                answer = self.data_processor._clean_korean_text(result.response)
                
                is_valid, quality = self._validate_korean_quality_enhanced(answer, structure["question_type"])
                
                if not is_valid or quality < 0.5:
                    self.stats["korean_failures"] += 1
                    original_answer = answer
                    answer = self._get_domain_specific_fallback_enhanced(question, structure["question_type"], structure)
                    self.stats["korean_fixes"] += 1
                    self.stats["fallback_used"] += 1
                    
                    if quality > 0.3:
                        self.stats["quality_improvements"] += 1
                else:
                    self.stats["model_generation_success"] += 1
                    if quality > 0.8:
                        self.stats["high_confidence_answers"] += 1
                
                if len(answer) < 30:
                    answer = self._get_domain_specific_fallback_enhanced(question, structure["question_type"], structure)
                    self.stats["fallback_used"] += 1
                elif len(answer) > 1000:
                    answer = answer[:997] + "..."
            
            else:
                self.stats["fallback_used"] += 1
                answer = self._get_domain_specific_fallback_enhanced(question, "multiple_choice", structure)
            
            if self.enable_learning and 'result' in locals() and result.confidence > 0.45:
                self.learning_system.learn_from_prediction(
                    question, answer, result.confidence,
                    structure["question_type"], analysis.get("domain", ["일반"])
                )
                self.stats["learned"] += 1
            
            self._manage_memory_efficiently()
            
            self.answer_cache[cache_key] = answer
            return answer
            
        except Exception as e:
            self.stats["errors"] += 1
            if self.verbose:
                print(f"처리 오류: {e}")
            
            self.stats["fallback_used"] += 1
            fallback_type = structure.get("question_type", "multiple_choice") if 'structure' in locals() else "multiple_choice"
            return self._get_domain_specific_fallback_enhanced(question, fallback_type)
    
    def _manage_memory_efficiently(self):
        self.memory_management["processed_count"] += 1
        count = self.memory_management["processed_count"]
        
        if count % self.memory_management["cleanup_frequency"] == 0:
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            gc.collect()
            self.stats["memory_optimizations"] += 1
        
        if count % self.memory_management["cache_optimization_frequency"] == 0:
            self._optimize_caches()
        
        if count % self.memory_management["deep_cleanup_frequency"] == 0:
            self._deep_memory_cleanup()
    
    def _optimize_caches(self):
        if len(self.answer_cache) > self.max_cache_size * 0.8:
            keys_to_remove = list(self.answer_cache.keys())[: self.max_cache_size // 3]
            for key in keys_to_remove:
                del self.answer_cache[key]
        
        if len(self.pattern_analysis_cache) > self.max_cache_size // 2:
            keys_to_remove = list(self.pattern_analysis_cache.keys())[: self.max_cache_size // 4]
            for key in keys_to_remove:
                del self.pattern_analysis_cache[key]
        
        if hasattr(self.data_processor, 'structure_cache'):
            if len(self.data_processor.structure_cache) > 600:
                keys_to_remove = list(self.data_processor.structure_cache.keys())[:200]
                for key in keys_to_remove:
                    del self.data_processor.structure_cache[key]
    
    def _deep_memory_cleanup(self):
        self.answer_cache.clear()
        self.pattern_analysis_cache.clear()
        
        if hasattr(self.data_processor, 'structure_cache'):
            self.data_processor.structure_cache.clear()
        
        if hasattr(self.prompt_engineer, 'prompt_cache'):
            self.prompt_engineer.prompt_cache.clear()
        
        if hasattr(self.model_handler, 'response_cache'):
            cache_size_before = len(self.model_handler.response_cache)
            if cache_size_before > 400:
                keys_to_keep = list(self.model_handler.response_cache.keys())[-200:]
                new_cache = {k: self.model_handler.response_cache[k] for k in keys_to_keep}
                self.model_handler.response_cache = new_cache
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def _debug_log(self, message: str):
        if self.verbose:
            print(f"[DEBUG] {message}")
    
    def execute_inference(self, test_file: str, submission_file: str,
                         output_file: str = "./final_submission.csv") -> Dict:
        test_df = pd.read_csv(test_file)
        submission_df = pd.read_csv(submission_file)
        
        print(f"데이터 로드 완료: {len(test_df)}개 문항")
        
        questions_data = []
        
        for idx, row in test_df.iterrows():
            question = row['Question']
            structure = self.data_processor.analyze_question_structure(question)
            
            questions_data.append({
                "idx": idx,
                "id": row['ID'],
                "question": question,
                "structure": structure,
                "is_mc": structure["question_type"] == "multiple_choice"
            })
        
        mc_count = sum(1 for q in questions_data if q["is_mc"])
        subj_count = len(questions_data) - mc_count
        
        print(f"문제 구성: 객관식 {mc_count}개, 주관식 {subj_count}개")
        
        if self.enable_learning:
            print(f"학습 모드: 활성화 (적응형 처리 포함)")
        
        answers = [""] * len(test_df)
        
        print("추론 시작...")
        progress_bar = tqdm(questions_data, desc="추론", ncols=80, disable=not self.verbose)
        
        batch_size = 25
        batch_answers = []
        
        for i, q_data in enumerate(progress_bar):
            idx = q_data["idx"]
            question_id = q_data["id"]
            question = q_data["question"]
            
            answer = self.process_question(question, question_id, idx)
            answers[idx] = answer
            batch_answers.append(answer)
            
            self.stats["total"] += 1
            
            if len(batch_answers) >= batch_size or i == len(questions_data) - 1:
                if self.enable_learning and len(batch_answers) >= 5:
                    self.learning_system.optimize_patterns()
                batch_answers.clear()
            
            if not self.verbose and self.stats["total"] % 50 == 0:
                print(f"진행률: {self.stats['total']}/{len(test_df)} ({self.stats['total']/len(test_df)*100:.1f}%)")
            
            if self.stats["total"] % 100 == 0:
                self._print_progress_stats()
        
        submission_df['Answer'] = answers
        submission_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        if self.enable_learning:
            try:
                if self.learning_system.save_model():
                    if self.verbose:
                        print("학습 데이터 저장 완료")
            except Exception as e:
                if self.verbose:
                    print(f"데이터 저장 오류: {e}")
        
        return self._generate_final_report(answers, questions_data, output_file)
    
    def _print_progress_stats(self):
        if self.stats["total"] > 0:
            success_rate = self.stats["model_generation_success"] / self.stats["total"] * 100
            pattern_rate = self.stats["pattern_based_answers"] / self.stats["total"] * 100
            cache_rate = self.stats["cache_hits"] / self.stats["total"] * 100
            print(f"  생성 성공률: {success_rate:.1f}%, 패턴 활용률: {pattern_rate:.1f}%, 캐시 적중률: {cache_rate:.1f}%")
    
    def _generate_final_report(self, answers: List[str], questions_data: List[Dict], output_file: str) -> Dict:
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
            _, quality = self._validate_korean_quality_enhanced(answer, "subjective")
            korean_quality_scores.append(quality)
        
        mc_quality_scores = []
        for answer in mc_answers:
            _, quality = self._validate_korean_quality_enhanced(answer, "multiple_choice")
            mc_quality_scores.append(quality)
        
        all_quality_scores = korean_quality_scores + mc_quality_scores
        avg_korean_quality = np.mean(all_quality_scores) if all_quality_scores else 0
        
        print("\n" + "="*60)
        print("추론 완료")
        print("="*60)
        print(f"총 문항: {len(answers)}개")
        
        print(f"\n처리 통계:")
        print(f"  모델 생성 성공: {self.stats['model_generation_success']}/{self.stats['total']} ({self.stats['model_generation_success']/max(self.stats['total'],1)*100:.1f}%)")
        print(f"  패턴 기반 답변: {self.stats['pattern_based_answers']}회 ({self.stats['pattern_based_answers']/max(self.stats['total'],1)*100:.1f}%)")
        print(f"  고신뢰도 답변: {self.stats['high_confidence_answers']}회 ({self.stats['high_confidence_answers']/max(self.stats['total'],1)*100:.1f}%)")
        print(f"  스마트 힌트 사용: {self.stats['smart_hints_used']}회")
        print(f"  적응형 처리: {self.stats['adaptive_processing']}회")
        print(f"  폴백 사용: {self.stats['fallback_used']}회")
        print(f"  처리 오류: {self.stats['errors']}회")
        
        print(f"\n메모리 및 캐시 통계:")
        print(f"  캐시 적중: {self.stats['cache_hits']}회")
        print(f"  메모리 최적화: {self.stats['memory_optimizations']}회")
        print(f"  품질 개선: {self.stats['quality_improvements']}회")
        
        print(f"\n한국어 품질 리포트:")
        print(f"  한국어 실패: {self.stats['korean_failures']}회")
        print(f"  한국어 수정: {self.stats['korean_fixes']}회")
        print(f"  평균 품질 점수: {avg_korean_quality:.3f}")
        
        high_quality_count = sum(1 for q in all_quality_scores if q > 0.7)
        print(f"  품질 우수 답변: {high_quality_count}/{len(all_quality_scores)}개 ({high_quality_count/max(len(all_quality_scores),1)*100:.1f}%)")
        
        if len(all_quality_scores) > 0:
            print(f"  평균 한국어 비율: {avg_korean_quality:.1%}")
        
        quality_assessment = "우수" if avg_korean_quality > 0.7 else "양호" if avg_korean_quality > 0.5 else "개선 필요"
        print(f"  전체 한국어 품질: {quality_assessment}")
        
        if self.enable_learning:
            print(f"\n학습 통계:")
            print(f"  학습된 샘플: {self.stats['learned']}개")
            print(f"  정적 패턴: {len(self.learning_system.pattern_weights)}개")
            print(f"  동적 패턴: {len(self.learning_system.dynamic_patterns)}개")
            print(f"  현재 정확도: {self.learning_system.get_current_accuracy():.2%}")
        
        if mc_answers:
            print(f"\n객관식 답변 분포 ({len(mc_answers)}개):")
            for num in sorted(answer_distribution.keys()):
                count = answer_distribution[num]
                pct = (count / len(mc_answers)) * 100
                print(f"  {num}번: {count}개 ({pct:.1f}%)")
            
            unique_answers = len(answer_distribution)
            diversity_assessment = "우수" if unique_answers >= 4 else "양호" if unique_answers >= 3 else "보통" if unique_answers >= 2 else "부족"
            print(f"  답변 다양성: {diversity_assessment}")
        
        efficiency_stats = self._calculate_efficiency_metrics()
        print(f"\n효율성 지표:")
        print(f"  전체 처리 시간: {efficiency_stats['total_time']:.1f}초")
        print(f"  문항당 평균 시간: {efficiency_stats['avg_time_per_question']:.2f}초")
        print(f"  캐시 효율성: {efficiency_stats['cache_efficiency']:.1%}")
        print(f"  메모리 사용 최적화: {efficiency_stats['memory_optimization']:.1%}")
        
        print(f"\n결과 파일: {output_file}")
        
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
                "adaptive_processing": self.stats["adaptive_processing"],
                "fallback_used": self.stats["fallback_used"],
                "errors": self.stats["errors"],
                "cache_hits": self.stats["cache_hits"]
            },
            "korean_quality": {
                "failures": self.stats["korean_failures"],
                "fixes": self.stats["korean_fixes"],
                "avg_score": avg_korean_quality,
                "high_quality_count": high_quality_count,
                "quality_improvements": self.stats["quality_improvements"]
            },
            "learning_stats": {
                "learned_samples": self.stats["learned"],
                "static_patterns": len(self.learning_system.pattern_weights) if self.enable_learning else 0,
                "dynamic_patterns": len(self.learning_system.dynamic_patterns) if self.enable_learning else 0,
                "accuracy": self.learning_system.get_current_accuracy() if self.enable_learning else 0
            },
            "efficiency_metrics": efficiency_stats
        }
    
    def _calculate_efficiency_metrics(self) -> Dict:
        total_time = time.time() - self.start_time
        total_requests = max(self.stats["total"], 1)
        
        return {
            "total_time": total_time,
            "avg_time_per_question": total_time / total_requests,
            "cache_efficiency": self.stats["cache_hits"] / total_requests,
            "memory_optimization": self.stats["memory_optimizations"] / max(total_requests // 25, 1),
            "processing_efficiency": (self.stats["model_generation_success"] + self.stats["pattern_based_answers"]) / total_requests
        }
    
    def cleanup(self):
        try:
            efficiency_stats = self._calculate_efficiency_metrics()
            
            print(f"\n시스템 정리:")
            print(f"  총 처리 시간: {efficiency_stats['total_time']:.1f}초")
            print(f"  평균 처리 속도: {efficiency_stats['avg_time_per_question']:.2f}초/문항")
            print(f"  캐시 효율성: {efficiency_stats['cache_efficiency']:.1%}")
            
            self.model_handler.cleanup()
            self.data_processor.cleanup()
            self.prompt_engineer.cleanup()
            
            if self.enable_learning:
                self.learning_system.cleanup()
            
            self.answer_cache.clear()
            self.pattern_analysis_cache.clear()
            
            torch.cuda.empty_cache()
            gc.collect()
            
        except Exception as e:
            if self.verbose:
                print(f"정리 중 오류: {e}")

def main():
    if torch.cuda.is_available():
        gpu_info = torch.cuda.get_device_properties(0)
        print(f"GPU: {gpu_info.name}")
        print(f"메모리: {gpu_info.total_memory / (1024**3):.1f}GB")
        print(f"CUDA 버전: {torch.version.cuda}")
    else:
        print("GPU 없음 - CPU 모드")
    
    test_file = "./test.csv"
    submission_file = "./sample_submission.csv"
    
    if not os.path.exists(test_file):
        print(f"오류: {test_file} 파일 없음")
        sys.exit(1)
    
    if not os.path.exists(submission_file):
        print(f"오류: {submission_file} 파일 없음")
        sys.exit(1)
    
    enable_learning = True
    verbose = False
    
    engine = None
    try:
        engine = FinancialAIInference(enable_learning=enable_learning, verbose=verbose)
        results = engine.execute_inference(test_file, submission_file)
        
        if results["success"]:
            print("\n최종 성과 요약:")
            processing_stats = results["processing_stats"]
            efficiency_metrics = results["efficiency_metrics"]
            
            print(f"처리 효율성: {efficiency_metrics['processing_efficiency']:.1%}")
            print(f"한국어 품질: {results['korean_quality']['avg_score']:.2f}")
            print(f"학습 성과: {results['learning_stats']['learned_samples']}개 샘플")
            
            if results['learning_stats']['dynamic_patterns'] > 0:
                print(f"동적 패턴 발견: {results['learning_stats']['dynamic_patterns']}개")
        
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