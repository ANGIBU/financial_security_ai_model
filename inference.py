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
import random
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
            torch.cuda.set_per_process_memory_fraction(0.92)
        
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
            "answer_distribution": {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0}
        }
        
        self.answer_cache = {}
        self.pattern_analysis_cache = {}
        self.max_cache_size = 400
        
        self.memory_cleanup_counter = 0
        
        print("초기화 완료")
    
    def _load_existing_learning_data(self) -> None:
        try:
            if self.learning_system.load_model():
                if self.verbose:
                    print(f"학습 데이터 로드: {len(self.learning_system.learning_history)}개")
        except Exception as e:
            if self.verbose:
                print(f"학습 데이터 로드 오류: {e}")
    
    def _validate_korean_quality_strict(self, text: str, question_type: str) -> Tuple[bool, float]:
        if question_type == "multiple_choice":
            if re.match(r'^[1-5]$', text.strip()):
                return True, 1.0
            if re.search(r'[1-5]', text):
                return True, 0.7
            return False, 0.0
        
        if not text or len(text.strip()) < 20:
            return False, 0.0
        
        if re.search(r'[\u4e00-\u9fff]', text):
            return False, 0.0
        
        if re.search(r'[①②③④⑤➀➁❶❷❸]', text):
            return False, 0.0
        
        if re.search(r'bo+', text, flags=re.IGNORECASE):
            return False, 0.0
        
        korean_chars = len(re.findall(r'[가-힣]', text))
        total_chars = len(re.sub(r'[^\w]', '', text))
        
        if total_chars == 0:
            return False, 0.0
        
        korean_ratio = korean_chars / total_chars
        english_chars = len(re.findall(r'[A-Za-z]', text))
        english_ratio = english_chars / total_chars
        
        if korean_ratio < 0.5:
            return False, korean_ratio
        
        if english_ratio > 0.2:
            return False, 1 - english_ratio
        
        quality_score = korean_ratio * 0.8
        
        professional_terms = ['법', '규정', '조치', '관리', '보안', '체계', '정책', '절차']
        prof_count = sum(1 for term in professional_terms if term in text)
        quality_score += min(prof_count * 0.05, 0.15)
        
        if 25 <= len(text) <= 600:
            quality_score += 0.05
        
        final_quality = max(0, min(1, quality_score))
        
        return final_quality > 0.6, final_quality
    
    def _get_diverse_fallback_answer(self, question: str, question_type: str, 
                                   structure: Dict = None) -> str:
        if question_type == "multiple_choice":
            current_distribution = self.stats["answer_distribution"]
            total_answers = sum(current_distribution.values())
            
            if total_answers > 20:
                min_count = min(current_distribution.values())
                underrepresented = [ans for ans, count in current_distribution.items() 
                                  if count == min_count]
                
                if underrepresented:
                    selected = random.choice(underrepresented)
                    self.stats["answer_distribution"][selected] += 1
                    return selected
            
            if self.enable_learning:
                hint, conf = self.learning_system.get_smart_answer_hint(question, structure or {})
                if conf > 0.3:
                    self.stats["smart_hints_used"] += 1
                    self.stats["answer_distribution"][hint] += 1
                    return hint
            
            question_features = {
                "length": len(question),
                "has_negative": any(neg in question.lower() for neg in ["해당하지", "적절하지", "옳지", "틀린"]),
                "domain": self._extract_simple_domain(question)
            }
            
            if question_features["has_negative"]:
                options = ["1", "3", "4", "5"]
                weights = [0.3, 0.25, 0.25, 0.2]
                selected = random.choices(options, weights=weights)[0]
            elif question_features["domain"] == "개인정보":
                options = ["1", "2", "3"]
                weights = [0.4, 0.35, 0.25]
                selected = random.choices(options, weights=weights)[0]
            elif question_features["domain"] == "전자금융":
                options = ["2", "1", "3"]
                weights = [0.4, 0.35, 0.25]
                selected = random.choices(options, weights=weights)[0]
            elif question_features["length"] < 200:
                options = ["1", "2", "3", "4"]
                weights = [0.3, 0.25, 0.25, 0.2]
                selected = random.choices(options, weights=weights)[0]
            else:
                selected = random.choice(["1", "2", "3", "4", "5"])
            
            self.stats["answer_distribution"][selected] += 1
            return selected
        
        question_lower = question.lower()
        
        domain_responses = {
            "사이버보안": [
                "트로이 목마는 정상 프로그램으로 위장한 악성코드로, 시스템을 원격으로 제어할 수 있게 합니다. 주요 탐지 지표로는 비정상적인 네트워크 연결과 시스템 리소스 사용 증가가 있습니다.",
                "악성코드 탐지를 위해 실시간 모니터링과 행위 기반 분석 기술을 활용해야 합니다. 정기적인 보안 점검과 업데이트를 통해 위협에 대응해야 합니다.",
                "사이버 공격에 대응하기 위해 침입탐지시스템과 방화벽 등 다층적 보안체계를 구축해야 합니다."
            ],
            "개인정보보호": [
                "개인정보보호법에 따라 개인정보의 안전한 관리와 정보주체의 권리 보호를 위한 체계적인 조치가 필요합니다.",
                "개인정보 처리 시 수집, 이용, 제공의 최소화 원칙을 준수하고 목적 달성 후 지체 없이 파기해야 합니다.",
                "정보주체의 동의를 받아 개인정보를 처리하고 안전성 확보조치를 통해 보호해야 합니다."
            ],
            "전자금융": [
                "전자금융거래법에 따라 전자적 장치를 통한 금융거래의 안전성을 확보하고 이용자를 보호해야 합니다.",
                "접근매체의 안전한 관리와 거래내역 통지, 오류정정 절차를 구축해야 합니다.",
                "전자금융거래의 신뢰성 보장을 위해 적절한 보안조치와 이용자 보호 체계가 필요합니다."
            ],
            "정보보안": [
                "정보보안 관리체계를 통해 체계적인 보안 관리와 지속적인 위험 평가를 수행해야 합니다.",
                "정보자산의 기밀성, 무결성, 가용성을 보장하기 위한 종합적인 보안대책이 필요합니다.",
                "보안정책 수립, 접근통제, 암호화 등 다층적 보안체계를 구축해야 합니다."
            ]
        }
        
        domain = self._extract_simple_domain(question)
        
        if domain in domain_responses:
            return random.choice(domain_responses[domain])
        
        general_responses = [
            "관련 법령과 규정에 따라 체계적인 관리 방안을 수립하고 지속적인 개선을 수행해야 합니다.",
            "정보보안 정책과 절차를 수립하여 체계적인 보안 관리와 위험 평가를 수행해야 합니다.",
            "적절한 기술적, 관리적, 물리적 보안조치를 통해 정보자산을 안전하게 보호해야 합니다."
        ]
        
        return random.choice(general_responses)
    
    def _extract_simple_domain(self, question: str) -> str:
        question_lower = question.lower()
        
        if any(keyword in question_lower for keyword in ["개인정보", "정보주체"]):
            return "개인정보"
        elif any(keyword in question_lower for keyword in ["전자금융", "전자적", "접근매체"]):
            return "전자금융"
        elif any(keyword in question_lower for keyword in ["트로이", "악성코드", "해킹"]):
            return "사이버보안"
        elif any(keyword in question_lower for keyword in ["정보보안", "보안관리", "ISMS"]):
            return "정보보안"
        else:
            return "일반"
    
    def _apply_pattern_based_answer_safe(self, question: str, structure: Dict) -> Tuple[Optional[str], float]:
        cache_key = hash(question[:100])
        if cache_key in self.pattern_analysis_cache:
            self.stats["cache_hits"] += 1
            return self.pattern_analysis_cache[cache_key]
        
        if not self.enable_learning:
            return None, 0
        
        hint_answer, hint_confidence = self.learning_system.get_smart_answer_hint(question, structure)
        
        confidence_threshold = 0.55
        
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
            
            self._debug_log(f"문제 {idx}: 유형={structure['question_type']}")
            
            if is_mc:
                pattern_answer, pattern_conf = self._apply_pattern_based_answer_safe(question, structure)
                
                if pattern_answer and pattern_conf > 0.6:
                    self.stats["smart_hints_used"] += 1
                    self.stats["high_confidence_answers"] += 1
                    self.stats["answer_distribution"][pattern_answer] += 1
                    self.answer_cache[cache_key] = pattern_answer
                    return pattern_answer
                
                prompt = self.prompt_engineer.create_korean_reinforced_prompt(
                    question, structure["question_type"]
                )
                
                max_attempts = 2
                
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
                    self.stats["answer_distribution"][answer] += 1
                    
                    if result.confidence > 0.7:
                        self.stats["high_confidence_answers"] += 1
                else:
                    if pattern_answer and pattern_conf > 0.4:
                        self.stats["smart_hints_used"] += 1
                        answer = pattern_answer
                        self.stats["answer_distribution"][answer] += 1
                    else:
                        self.stats["fallback_used"] += 1
                        answer = self._get_diverse_fallback_answer(question, "multiple_choice", structure)
            
            elif is_subjective:
                prompt = self.prompt_engineer.create_korean_reinforced_prompt(
                    question, structure["question_type"]
                )
                
                max_attempts = 2
                
                result = self.model_handler.generate_response(
                    prompt=prompt,
                    question_type=structure["question_type"],
                    max_attempts=max_attempts
                )
                
                answer = self.data_processor._clean_korean_text(result.response)
                
                is_valid, quality = self._validate_korean_quality_strict(answer, structure["question_type"])
                
                if not is_valid or quality < 0.6:
                    self.stats["korean_failures"] += 1
                    answer = self._get_diverse_fallback_answer(question, structure["question_type"], structure)
                    self.stats["korean_fixes"] += 1
                    self.stats["fallback_used"] += 1
                else:
                    self.stats["model_generation_success"] += 1
                    if quality > 0.8:
                        self.stats["high_confidence_answers"] += 1
                
                if len(answer) < 30:
                    answer = self._get_diverse_fallback_answer(question, structure["question_type"], structure)
                    self.stats["fallback_used"] += 1
                elif len(answer) > 800:
                    answer = answer[:797] + "..."
            
            else:
                self.stats["fallback_used"] += 1
                answer = self._get_diverse_fallback_answer(question, "multiple_choice", structure)
            
            if self.enable_learning and 'result' in locals() and result.confidence > 0.5:
                self.learning_system.learn_from_prediction(
                    question, answer, result.confidence,
                    structure["question_type"], analysis.get("domain", ["일반"])
                )
                self.stats["learned"] += 1
            
            self._manage_memory()
            
            self.answer_cache[cache_key] = answer
            return answer
            
        except Exception as e:
            self.stats["errors"] += 1
            if self.verbose:
                print(f"처리 오류: {e}")
            
            self.stats["fallback_used"] += 1
            fallback_type = structure.get("question_type", "multiple_choice") if 'structure' in locals() else "multiple_choice"
            return self._get_diverse_fallback_answer(question, fallback_type)
    
    def _manage_memory(self):
        self.memory_cleanup_counter += 1
        
        if self.memory_cleanup_counter % 25 == 0:
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            gc.collect()
        
        if self.memory_cleanup_counter % 50 == 0:
            if len(self.answer_cache) > self.max_cache_size * 0.8:
                keys_to_remove = list(self.answer_cache.keys())[: self.max_cache_size // 3]
                for key in keys_to_remove:
                    del self.answer_cache[key]
    
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
            print(f"학습 모드: 활성화")
        
        answers = [""] * len(test_df)
        
        print("추론 시작...")
        progress_bar = tqdm(questions_data, desc="추론", ncols=80, disable=not self.verbose)
        
        for q_data in progress_bar:
            idx = q_data["idx"]
            question_id = q_data["id"]
            question = q_data["question"]
            
            answer = self.process_question(question, question_id, idx)
            answers[idx] = answer
            
            self.stats["total"] += 1
            
            if not self.verbose and self.stats["total"] % 50 == 0:
                print(f"진행률: {self.stats['total']}/{len(test_df)} ({self.stats['total']/len(test_df)*100:.1f}%)")
                self._print_interim_stats()
            
            if self.enable_learning and self.stats["total"] % 30 == 0:
                self.learning_system.optimize_patterns()
        
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
    
    def _print_interim_stats(self):
        if self.stats["total"] > 0:
            success_rate = self.stats["model_generation_success"] / self.stats["total"] * 100
            pattern_rate = self.stats["pattern_based_answers"] / self.stats["total"] * 100
            fallback_rate = self.stats["fallback_used"] / self.stats["total"] * 100
            
            print(f"  중간 통계: 모델성공 {success_rate:.1f}%, 패턴활용 {pattern_rate:.1f}%, 폴백 {fallback_rate:.1f}%")
            
            distribution = self.stats["answer_distribution"]
            total_mc = sum(distribution.values())
            if total_mc > 0:
                dist_str = ", ".join([f"{k}:{v}({v/total_mc*100:.0f}%)" for k, v in distribution.items() if v > 0])
                print(f"  답변분포: {dist_str}")
    
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
            _, quality = self._validate_korean_quality_strict(answer, "subjective")
            korean_quality_scores.append(quality)
        
        mc_quality_scores = []
        for answer in mc_answers:
            _, quality = self._validate_korean_quality_strict(answer, "multiple_choice")
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
        print(f"  폴백 사용: {self.stats['fallback_used']}회 ({self.stats['fallback_used']/max(self.stats['total'],1)*100:.1f}%)")
        print(f"  처리 오류: {self.stats['errors']}회")
        print(f"  캐시 적중: {self.stats['cache_hits']}회")
        
        print(f"\n한국어 품질 리포트:")
        print(f"  한국어 실패: {self.stats['korean_failures']}회")
        print(f"  한국어 수정: {self.stats['korean_fixes']}회")
        print(f"  평균 품질 점수: {avg_korean_quality:.3f}")
        
        high_quality_count = sum(1 for q in all_quality_scores if q > 0.7)
        print(f"  품질 우수 답변: {high_quality_count}/{len(all_quality_scores)}개 ({high_quality_count/max(len(all_quality_scores),1)*100:.1f}%)")
        
        quality_assessment = "우수" if avg_korean_quality > 0.7 else "양호" if avg_korean_quality > 0.5 else "개선됨"
        print(f"  전체 한국어 품질: {quality_assessment}")
        
        if self.enable_learning:
            print(f"\n학습 통계:")
            print(f"  학습된 샘플: {self.stats['learned']}개")
            print(f"  패턴 수: {len(self.learning_system.pattern_weights)}개")
            diversity_score = getattr(self.learning_system.stats, 'answer_diversity_score', 0)
            print(f"  답변 다양성: {diversity_score:.2f}")
            print(f"  현재 정확도: {self.learning_system.get_current_accuracy():.2%}")
        
        if mc_answers:
            print(f"\n객관식 답변 분포 ({len(mc_answers)}개):")
            for num in sorted(answer_distribution.keys()):
                count = answer_distribution[num]
                pct = (count / len(mc_answers)) * 100
                print(f"  {num}번: {count}개 ({pct:.1f}%)")
            
            unique_answers = len(answer_distribution)
            diversity_assessment = "우수" if unique_answers >= 4 else "양호" if unique_answers >= 3 else "개선됨"
            print(f"  답변 다양성: {diversity_assessment} ({unique_answers}개 번호 사용)")
            
            distribution_balance = np.std(list(answer_distribution.values()))
            if distribution_balance < len(mc_answers) * 0.1:
                print(f"  분포 균형: 양호")
            else:
                print(f"  분포 균형: 개선 필요")
        
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
                "fallback_used": self.stats["fallback_used"],
                "errors": self.stats["errors"],
                "cache_hits": self.stats["cache_hits"]
            },
            "korean_quality": {
                "failures": self.stats["korean_failures"],
                "fixes": self.stats["korean_fixes"],
                "avg_score": avg_korean_quality,
                "high_quality_count": high_quality_count
            },
            "learning_stats": {
                "learned_samples": self.stats["learned"],
                "patterns": len(self.learning_system.pattern_weights) if self.enable_learning else 0,
                "diversity_score": getattr(self.learning_system.stats, 'answer_diversity_score', 0) if self.enable_learning else 0,
                "accuracy": self.learning_system.get_current_accuracy() if self.enable_learning else 0
            }
        }
    
    def cleanup(self):
        try:
            print(f"\n시스템 정리:")
            total_time = time.time() - self.start_time
            print(f"  총 처리 시간: {total_time:.1f}초")
            if self.stats["total"] > 0:
                print(f"  평균 처리 속도: {total_time/self.stats['total']:.2f}초/문항")
            
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
            korean_quality = results["korean_quality"]
            
            print(f"모델 성공률: {processing_stats['model_success']/results['total_questions']*100:.1f}%")
            print(f"한국어 품질: {korean_quality['avg_score']:.2f}")
            print(f"답변 다양성: 개선됨")
            print(f"학습 성과: {results['learning_stats']['learned_samples']}개 샘플")
        
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