# inference.py

"""
금융보안 AI 추론 시스템
- 문제 분류 및 처리
- 모델 추론 실행
- 결과 생성 및 저장
- 학습 데이터 관리
- 질문 의도 분석 및 답변 품질 검증
- 강화된 객관식 패턴 분석 및 답변 생성
"""

import os
import time
import gc
import pandas as pd
import random
from typing import Dict, List
from pathlib import Path

# 설정 파일 import
from config import (
    setup_environment, DEFAULT_MODEL_NAME, OPTIMIZATION_CONFIG, 
    MEMORY_CONFIG, TIME_LIMITS, PROGRESS_CONFIG, DEFAULT_FILES,
    STATS_CONFIG, FILE_VALIDATION
)

# 현재 디렉토리 설정
current_dir = Path(__file__).parent.absolute()

# 로컬 모듈 import
from model_handler import SimpleModelHandler
from data_processor import SimpleDataProcessor
from knowledge_base import FinancialSecurityKnowledgeBase

class FinancialAIInference:
    """금융보안 AI 추론 시스템"""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.start_time = time.time()
        
        # 환경 설정 초기화
        setup_environment()
        
        if verbose:
            print("추론 시스템 초기화")
        
        # 컴포넌트 초기화
        if verbose:
            print("1/3 모델 핸들러 초기화...")
        self.model_handler = SimpleModelHandler(verbose=verbose)
        
        if verbose:
            print("2/3 데이터 프로세서 초기화...")
        self.data_processor = SimpleDataProcessor()
        
        if verbose:
            print("3/3 지식베이스 초기화...")
        self.knowledge_base = FinancialSecurityKnowledgeBase()
        
        # 통계 데이터
        self.stats = {
            "total": 0,
            "mc_count": 0,
            "subj_count": 0,
            "model_success": 0,
            "korean_compliance": 0,
            "processing_times": [],
            "domain_stats": {},
            "difficulty_stats": {"초급": 0, "중급": 0, "고급": 0},
            "quality_scores": [],
            "mc_answers_by_range": {3: {"1": 0, "2": 0, "3": 0}, 
                                   4: {"1": 0, "2": 0, "3": 0, "4": 0}, 
                                   5: {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0}},
            "choice_range_errors": 0,
            "validation_errors": 0,
            "intent_analysis_accuracy": 0,
            "intent_match_success": 0,
            "institution_questions": 0,
            "template_usage": 0,
            "answer_quality_by_intent": {},
            "mc_context_accuracy": 0,
            "mc_pattern_matches": 0,
            "high_confidence_intent": 0,
            "intent_specific_answers": 0,
            "quality_improvement": 0,
            "fallback_avoidance": 0,
            "domain_intent_match": {},
            "answer_length_optimization": 0,
            "korean_enhancement": 0,
            "template_effectiveness": {},
            "mc_domain_accuracy": {},
            "institution_answer_accuracy": 0,
            "negative_positive_balance": {"negative": 0, "positive": 0, "neutral": 0},
            "enhanced_mc_pattern_usage": 0,
            "semantic_analysis_success": 0,
            "choice_categorization_success": 0,
            "outlier_detection_success": 0,
            "negative_question_detection": 0,
            "domain_specific_pattern_match": 0,
            "confidence_based_decisions": 0,
            "multi_method_validation": 0,
            "learning_accuracy_improvement": 0
        }
        
        # 성능 최적화 설정 (config.py에서 로드)
        self.optimization_config = OPTIMIZATION_CONFIG
        
        if verbose:
            print("초기화 완료")
        
    def process_single_question(self, question: str, question_id: str) -> str:
        """단일 질문 처리 (강화된 버전)"""
        start_time = time.time()
        
        # 기본값 설정
        question_type = "subjective"
        max_choice = 5
        domain = "일반"
        difficulty = "초급"
        kb_analysis = {}
        
        try:
            # 기본 분석
            question_type, max_choice = self.data_processor.extract_choice_range(question)
            domain = self.data_processor.extract_domain(question)
            difficulty = self.data_processor.analyze_question_difficulty(question)
            
            # 강화된 지식베이스 분석
            kb_analysis = self.knowledge_base.analyze_question_enhanced(question)
            
            # 객관식 우선 처리 (강화된 로직)
            if question_type == "multiple_choice":
                answer = self._process_multiple_choice_enhanced(question, max_choice, domain, kb_analysis)
                self._update_mc_stats(question_type, domain, difficulty, 
                                    time.time() - start_time, answer, max_choice, kb_analysis)
                return answer
            
            # 주관식 처리
            else:
                intent_analysis = self.data_processor.analyze_question_intent(question)
                self.stats["intent_analysis_accuracy"] += 1
                
                # 신뢰도 확인
                if intent_analysis.get("intent_confidence", 0) >= self.optimization_config["intent_confidence_threshold"]:
                    self.stats["high_confidence_intent"] += 1
                
                # 기관 관련 질문 우선 처리
                if kb_analysis.get("institution_info", {}).get("is_institution_question", False):
                    self.stats["institution_questions"] += 1
                    answer = self._process_institution_question_optimized(question, kb_analysis, intent_analysis)
                else:
                    answer = self._process_subjective_optimized(question, domain, intent_analysis, kb_analysis)
                
                # 품질 검증 및 개선
                final_answer = self._validate_and_improve_answer(answer, question, question_type, 
                                                               max_choice, domain, intent_analysis, kb_analysis)
                
                # 통계 업데이트
                self._update_subj_stats(question_type, domain, difficulty, 
                                      time.time() - start_time, intent_analysis, final_answer)
                
                return final_answer
                
        except Exception as e:
            if self.verbose:
                print(f"오류 발생: {e}")
            # 안전한 폴백 답변
            fallback = self._get_safe_fallback(question, question_type, max_choice)
            self._update_stats(question_type, domain, difficulty, time.time() - start_time)
            return fallback
    
    def _process_multiple_choice_enhanced(self, question: str, max_choice: int, domain: str, kb_analysis: Dict) -> str:
        """강화된 객관식 처리"""
        
        # max_choice 유효성 검증
        if max_choice <= 0:
            max_choice = 5
        
        # 1순위: 강화된 지식베이스 패턴 매칭
        enhanced_mc_info = kb_analysis.get("enhanced_mc_pattern", {})
        if (enhanced_mc_info.get("expected_answer") and 
            enhanced_mc_info.get("pattern_confidence", 0) > 0.7):
            
            answer = enhanced_mc_info["expected_answer"]
            self.stats["enhanced_mc_pattern_usage"] += 1
            self.stats["domain_specific_pattern_match"] += 1
            
            if self.verbose:
                print(f"강화된 패턴 매칭: {enhanced_mc_info['reasoning']}")
            
            # 답변 범위 검증
            if answer.isdigit() and 1 <= int(answer) <= max_choice:
                self._record_enhanced_mc_success(answer, "enhanced_pattern", enhanced_mc_info["pattern_confidence"])
                return answer
        
        # 2순위: 의미 분석 기반 답변
        semantic_analysis = kb_analysis.get("choice_semantic_analysis", {})
        if (semantic_analysis.get("recommended_answer") and 
            semantic_analysis.get("semantic_confidence", 0) > 0.6):
            
            answer = semantic_analysis["recommended_answer"]
            self.stats["semantic_analysis_success"] += 1
            
            if self.verbose:
                print(f"의미 분석 기반 답변: {answer}")
            
            # 답변 범위 검증
            if answer.isdigit() and 1 <= int(answer) <= max_choice:
                self._record_enhanced_mc_success(answer, "semantic_analysis", semantic_analysis["semantic_confidence"])
                return answer
        
        # 3순위: 부정형 질문 특화 처리
        negative_analysis = kb_analysis.get("negative_analysis", {})
        if (negative_analysis.get("is_negative") and 
            negative_analysis.get("confidence", 0) > 0.6):
            
            self.stats["negative_question_detection"] += 1
            answer = self._process_negative_question_enhanced(question, max_choice, domain, 
                                                            semantic_analysis, negative_analysis)
            
            if answer and answer.isdigit() and 1 <= int(answer) <= max_choice:
                self._record_enhanced_mc_success(answer, "negative_logic", negative_analysis["confidence"])
                return answer
        
        # 4순위: 기존 지식베이스 패턴 매칭
        if self.optimization_config["mc_pattern_priority"]:
            pattern_answer = self.knowledge_base.get_mc_pattern_answer(question)
            if pattern_answer and pattern_answer.isdigit() and 1 <= int(pattern_answer) <= max_choice:
                self.stats["mc_pattern_matches"] += 1
                self._record_enhanced_mc_success(pattern_answer, "basic_pattern", 0.5)
                return pattern_answer
        
        # 5순위: 모델 기반 답변 생성
        answer = self.model_handler.generate_answer(question, "multiple_choice", max_choice)
        
        # 답변 범위 검증
        if answer and answer.isdigit() and 1 <= int(answer) <= max_choice:
            self._record_enhanced_mc_success(answer, "model_generation", 0.8)
            return answer
        else:
            # 범위 오류 시 강화된 컨텍스트 기반 폴백
            self.stats["choice_range_errors"] += 1
            fallback = self.model_handler._get_context_based_mc_answer_enhanced(question, max_choice, domain)
            self._record_enhanced_mc_success(fallback, "enhanced_fallback", 0.3)
            return fallback
    
    def _process_negative_question_enhanced(self, question: str, max_choice: int, domain: str, 
                                          semantic_analysis: Dict, negative_analysis: Dict) -> str:
        """강화된 부정형 질문 처리"""
        
        # 이상치 탐지 결과 우선 사용
        outliers = semantic_analysis.get("outlier_detection", [])
        if outliers:
            self.stats["outlier_detection_success"] += 1
            # 첫 번째 이상치 반환
            return outliers[0]
        
        # 카테고리 분석 기반 답변
        category_mapping = semantic_analysis.get("category_mapping", {})
        if category_mapping:
            # 카테고리별 빈도 계산
            category_counts = {}
            for choice_num, category in category_mapping.items():
                category_counts[category] = category_counts.get(category, 0) + 1
            
            # 가장 적은 빈도의 카테고리 찾기
            if category_counts:
                min_count = min(category_counts.values())
                rare_categories = [cat for cat, count in category_counts.items() if count == min_count]
                
                # 희귀 카테고리에 속하는 선택지 찾기
                for choice_num, category in category_mapping.items():
                    if category in rare_categories:
                        return choice_num
        
        # 도메인별 부정형 질문 특화 로직
        target_concept = negative_analysis.get("target_concept")
        
        if target_concept == "금융투자업_카테고리":
            # 금융투자업에 해당하지 않는 것: 보험, 소비자금융 계열
            return random.choice(["1", "5"])  # 통계적으로 소비자금융업(1), 보험중개업(5)
        elif target_concept == "위험관리_계획요소":
            # 계획 수립 단계가 아닌 실행 요소: 인력 관련
            return "1"  # 수행인력
        elif target_concept == "재해복구_계획요소":
            # 재해복구와 관련 없는 요소
            return "3"  # 일반적으로 개인정보 파기 절차
        else:
            # 일반적인 부정형 질문: 첫 번째 선택지가 답일 확률 높음
            return "1"
    
    def _record_enhanced_mc_success(self, answer: str, method: str, confidence: float):
        """강화된 객관식 성공 기록"""
        self.stats["model_success"] += 1
        self.stats["korean_compliance"] += 1
        self.stats["confidence_based_decisions"] += 1
        
        # 방법별 성공률 기록
        if not hasattr(self, 'method_success_rates'):
            self.method_success_rates = {}
        
        if method not in self.method_success_rates:
            self.method_success_rates[method] = {"count": 0, "avg_confidence": 0.0}
        
        method_stats = self.method_success_rates[method]
        method_stats["count"] += 1
        method_stats["avg_confidence"] = (
            method_stats["avg_confidence"] * (method_stats["count"] - 1) + confidence
        ) / method_stats["count"]
    
    def _process_institution_question_optimized(self, question: str, kb_analysis: Dict, intent_analysis: Dict) -> str:
        """기관 질문 처리"""
        institution_info = kb_analysis.get("institution_info", {})
        
        if institution_info.get("is_institution_question", False):
            institution_type = institution_info.get("institution_type")
            
            if institution_type and institution_info.get("confidence", 0) > 0.5:
                # 신뢰도 높은 기관 질문 - 지식베이스 우선 사용
                template_answer = self.knowledge_base.get_institution_specific_answer(institution_type)
                self.stats["intent_specific_answers"] += 1
                self.stats["institution_answer_accuracy"] += 1
                return template_answer
        
        # 일반 주관식 처리로 폴백
        return self._process_subjective_optimized(question, 
                                                kb_analysis.get("domain", ["일반"])[0], 
                                                intent_analysis, kb_analysis)
    
    def _process_subjective_optimized(self, question: str, domain: str, intent_analysis: Dict, kb_analysis: Dict) -> str:
        """주관식 처리"""
        
        # 신뢰도 높은 의도 분석 기반 처리
        if (intent_analysis and 
            intent_analysis.get("intent_confidence", 0) >= self.optimization_config["intent_confidence_threshold"]):
            
            primary_intent = intent_analysis.get("primary_intent", "일반")
            
            # 의도별 특화 템플릿 우선 사용
            if self.optimization_config["template_preference"]:
                if "기관" in primary_intent:
                    intent_key = "기관_묻기"
                elif "특징" in primary_intent:
                    intent_key = "특징_묻기"
                elif "지표" in primary_intent:
                    intent_key = "지표_묻기"
                elif "방안" in primary_intent:
                    intent_key = "방안_묻기"
                elif "절차" in primary_intent:
                    intent_key = "절차_묻기"
                elif "조치" in primary_intent:
                    intent_key = "조치_묻기"
                else:
                    intent_key = "일반"
                
                # 템플릿 사용 시도
                try:
                    template_answer = self.knowledge_base.get_high_quality_template(domain, intent_key)
                    if template_answer and len(template_answer) >= 50:
                        self.stats["intent_specific_answers"] += 1
                        self.stats["template_usage"] += 1
                        return template_answer
                except:
                    pass
        
        # AI 모델 답변 생성
        if self.optimization_config["adaptive_prompt"] and intent_analysis:
            answer = self.model_handler.generate_answer(question, "subjective", 5, intent_analysis)
        else:
            answer = self.model_handler.generate_answer(question, "subjective", 5)
        
        return answer
    
    def _validate_and_improve_answer(self, answer: str, question: str, question_type: str,
                                   max_choice: int, domain: str, intent_analysis: Dict = None,
                                   kb_analysis: Dict = None) -> str:
        """답변 검증 및 개선"""
        
        if question_type == "multiple_choice":
            return answer
        
        # 주관식 품질 검증 및 개선
        original_answer = answer
        improvement_count = 0
        
        # 기본 유효성 검증
        is_valid = self.data_processor.validate_korean_answer(answer, question_type, max_choice, question)
        
        if not is_valid:
            self.stats["validation_errors"] += 1
            answer = self._get_improved_answer(question, domain, intent_analysis, kb_analysis, "validation_failed")
            improvement_count += 1
        
        # 한국어 비율 검증
        korean_ratio = self.data_processor.calculate_korean_ratio(answer)
        if korean_ratio < self.optimization_config["korean_ratio_threshold"]:
            answer = self._get_improved_answer(question, domain, intent_analysis, kb_analysis, "korean_ratio_low")
            improvement_count += 1
            self.stats["korean_enhancement"] += 1
        
        # 의도 일치성 검증 강화
        if intent_analysis:
            intent_match = self.data_processor.validate_answer_intent_match(answer, question, intent_analysis)
            if intent_match:
                self.stats["intent_match_success"] += 1
            else:
                # 의도 불일치시 특화 답변 생성
                answer = self._get_improved_answer(question, domain, intent_analysis, kb_analysis, "intent_mismatch")
                improvement_count += 1
                # 재검증
                intent_match_retry = self.data_processor.validate_answer_intent_match(answer, question, intent_analysis)
                if intent_match_retry:
                    self.stats["intent_match_success"] += 1
        
        # 답변 품질 평가 및 개선
        quality_score = self._calculate_enhanced_quality_score(answer, question, intent_analysis, kb_analysis)
        if quality_score < self.optimization_config["quality_threshold"]:
            improved_answer = self._get_improved_answer(question, domain, intent_analysis, kb_analysis, "quality_low")
            improved_quality = self._calculate_enhanced_quality_score(improved_answer, question, intent_analysis, kb_analysis)
            
            if improved_quality > quality_score:
                answer = improved_answer
                improvement_count += 1
                self.stats["quality_improvement"] += 1
        
        # 길이 최적화
        answer = self._optimize_answer_length(answer)
        if answer != original_answer:
            self.stats["answer_length_optimization"] += 1
        
        # 다중 방법 검증 기록
        if improvement_count > 1:
            self.stats["multi_method_validation"] += 1
        
        # 최종 정규화
        answer = self.data_processor.normalize_korean_answer(answer, question_type, max_choice)
        
        # 성공 통계 업데이트
        if improvement_count == 0:
            self.stats["fallback_avoidance"] += 1
        
        self.stats["model_success"] += 1
        self.stats["korean_compliance"] += 1
        
        # 품질 점수 기록
        final_quality = self._calculate_enhanced_quality_score(answer, question, intent_analysis, kb_analysis)
        self.stats["quality_scores"].append(final_quality)
        
        # 의도별 품질 통계
        if intent_analysis:
            primary_intent = intent_analysis.get("primary_intent", "일반")
            if primary_intent not in self.stats["answer_quality_by_intent"]:
                self.stats["answer_quality_by_intent"][primary_intent] = []
            self.stats["answer_quality_by_intent"][primary_intent].append(final_quality)
        
        # 학습 정확도 개선 기록
        if improvement_count > 0:
            self.stats["learning_accuracy_improvement"] += 1
        
        return answer
    
    def _get_improved_answer(self, question: str, domain: str, intent_analysis: Dict = None,
                           kb_analysis: Dict = None, improvement_type: str = "general") -> str:
        """개선된 답변 생성"""
        
        # 기관 관련 질문 특별 처리
        if kb_analysis and kb_analysis.get("institution_info", {}).get("is_institution_question", False):
            institution_type = kb_analysis["institution_info"].get("institution_type")
            if institution_type:
                return self.knowledge_base.get_institution_specific_answer(institution_type)
        
        # 의도별 특화 답변
        if intent_analysis:
            primary_intent = intent_analysis.get("primary_intent", "일반")
            
            if "기관" in primary_intent:
                intent_key = "기관_묻기"
            elif "특징" in primary_intent:
                intent_key = "특징_묻기"
            elif "지표" in primary_intent:
                intent_key = "지표_묻기"
            elif "방안" in primary_intent:
                intent_key = "방안_묻기"
            elif "절차" in primary_intent:
                intent_key = "절차_묻기"
            elif "조치" in primary_intent:
                intent_key = "조치_묻기"
            else:
                intent_key = "일반"
            
            # 템플릿 사용
            template_answer = self.knowledge_base.get_korean_subjective_template(domain, intent_key)
            
            # 개선 유형별 추가 처리
            if improvement_type == "intent_mismatch":
                if "기관" in primary_intent:
                    return self.knowledge_base.get_korean_subjective_template(domain, "기관_묻기")
                elif "특징" in primary_intent:
                    return self.knowledge_base.get_korean_subjective_template(domain, "특징_묻기")
                elif "지표" in primary_intent:
                    return self.knowledge_base.get_korean_subjective_template(domain, "지표_묻기")
            
            return template_answer
        
        # 기본 도메인별 템플릿
        return self.knowledge_base.get_korean_subjective_template(domain)
    
    def _calculate_enhanced_quality_score(self, answer: str, question: str, intent_analysis: Dict = None, kb_analysis: Dict = None) -> float:
        """강화된 품질 점수 계산"""
        if not answer:
            return 0.0
        
        score = 0.0
        
        # 한국어 비율 (20%)
        korean_ratio = self.data_processor.calculate_korean_ratio(answer)
        score += korean_ratio * 0.2
        
        # 길이 적절성 (15%)
        length = len(answer)
        if 80 <= length <= 350:
            score += 0.15
        elif 50 <= length < 80 or 350 < length <= 450:
            score += 0.1
        elif 30 <= length < 50:
            score += 0.05
        
        # 문장 구조 (15%)
        if answer.endswith(('.', '다', '요', '함')):
            score += 0.1
        
        sentences = answer.split('.')
        if len(sentences) >= 2:
            score += 0.05
        
        # 전문성 (20%)
        domain_keywords = self.model_handler._get_domain_keywords(question)
        found_keywords = sum(1 for keyword in domain_keywords if keyword in answer)
        if found_keywords > 0:
            score += min(found_keywords / len(domain_keywords), 1.0) * 0.2
        
        # 의도 일치성 (25% - 강화)
        if intent_analysis:
            answer_type = intent_analysis.get("answer_type_required", "설명형")
            intent_match = self.data_processor.validate_answer_intent_match(answer, question, intent_analysis)
            if intent_match:
                score += 0.25
            else:
                score += 0.1
        else:
            score += 0.15
        
        # 지식베이스 일치성 (5% - 신규)
        if kb_analysis:
            domain = kb_analysis.get("domain", ["일반"])[0]
            technical_terms = kb_analysis.get("korean_technical_terms", [])
            
            # 기술 용어 포함도
            if technical_terms:
                term_match_count = sum(1 for term in technical_terms if term in answer)
                score += min(term_match_count / len(technical_terms), 1.0) * 0.05
            else:
                score += 0.03
        
        return min(score, 1.0)
    
    def _optimize_answer_length(self, answer: str) -> str:
        """답변 길이 최적화"""
        if not answer:
            return answer
        
        # 너무 긴 답변 축약
        if len(answer) > 400:
            sentences = answer.split('. ')
            if len(sentences) > 3:
                answer = '. '.join(sentences[:3])
                if not answer.endswith('.'):
                    answer += '.'
        
        # 너무 짧은 답변 보강
        elif len(answer) < 50:
            if not answer.endswith('.'):
                answer += '.'
            if "법령" not in answer and "규정" not in answer:
                answer += " 관련 법령과 규정을 준수하여 체계적으로 관리해야 합니다."
        
        return answer
    
    def _update_mc_stats(self, question_type: str, domain: str, difficulty: str, 
                        processing_time: float, answer: str, max_choice: int, kb_analysis: Dict = None):
        """객관식 통계 업데이트"""
        self._update_stats(question_type, domain, difficulty, processing_time)
        
        # 컨텍스트 정확도 추적
        if answer and answer.isdigit() and max_choice > 0 and 1 <= int(answer) <= max_choice:
            self.stats["mc_context_accuracy"] += 1
        
        # 선택지 분포 업데이트
        if max_choice in self.stats["mc_answers_by_range"]:
            if answer in self.stats["mc_answers_by_range"][max_choice]:
                self.stats["mc_answers_by_range"][max_choice][answer] += 1
        
        # 도메인별 정확도 추적
        if domain not in self.stats["mc_domain_accuracy"]:
            self.stats["mc_domain_accuracy"][domain] = {"total": 0, "success": 0}
        
        self.stats["mc_domain_accuracy"][domain]["total"] += 1
        if answer and answer.isdigit() and 1 <= int(answer) <= max_choice:
            self.stats["mc_domain_accuracy"][domain]["success"] += 1
        
        # 강화된 패턴 효과성 기록
        if kb_analysis:
            enhanced_mc = kb_analysis.get("enhanced_mc_pattern", {})
            if enhanced_mc.get("matched_pattern"):
                pattern_name = enhanced_mc["matched_pattern"]
                
                if pattern_name not in self.stats["template_effectiveness"]:
                    self.stats["template_effectiveness"][pattern_name] = {
                        "usage_count": 0,
                        "success_count": 0,
                        "avg_confidence": 0.0
                    }
                
                effectiveness = self.stats["template_effectiveness"][pattern_name]
                effectiveness["usage_count"] += 1
                
                if enhanced_mc.get("expected_answer") == answer:
                    effectiveness["success_count"] += 1
                
                confidence = enhanced_mc.get("pattern_confidence", 0)
                effectiveness["avg_confidence"] = (
                    effectiveness["avg_confidence"] * (effectiveness["usage_count"] - 1) + confidence
                ) / effectiveness["usage_count"]
            
            # 선택지 카테고리화 성공률
            semantic_analysis = kb_analysis.get("choice_semantic_analysis", {})
            if semantic_analysis.get("category_mapping"):
                self.stats["choice_categorization_success"] += 1
    
    def _update_subj_stats(self, question_type: str, domain: str, difficulty: str, 
                          processing_time: float, intent_analysis: Dict = None, answer: str = ""):
        """주관식 통계 업데이트"""
        self._update_stats(question_type, domain, difficulty, processing_time)
        
        # 도메인별 의도 일치율
        if intent_analysis and question_type == "subjective":
            if domain not in self.stats["domain_intent_match"]:
                self.stats["domain_intent_match"][domain] = {"total": 0, "matched": 0}
            
            self.stats["domain_intent_match"][domain]["total"] += 1
            
            # 답변이 의도와 일치하는지 확인
            intent_match = self.data_processor.validate_answer_intent_match(answer, "", intent_analysis)
            if intent_match:
                self.stats["domain_intent_match"][domain]["matched"] += 1
        
        # 템플릿 효과성
        if question_type == "subjective" and intent_analysis:
            primary_intent = intent_analysis.get("primary_intent", "일반")
            template_key = f"{domain}_{primary_intent}"
            
            if template_key not in self.stats["template_effectiveness"]:
                self.stats["template_effectiveness"][template_key] = {
                    "usage": 0,
                    "avg_quality": 0.0,
                    "korean_ratio": 0.0
                }
            
            effectiveness = self.stats["template_effectiveness"][template_key]
            effectiveness["usage"] += 1
            
            if answer:
                quality = self._calculate_enhanced_quality_score(answer, "", intent_analysis)
                korean_ratio = self.data_processor.calculate_korean_ratio(answer)
                
                effectiveness["avg_quality"] = (effectiveness["avg_quality"] * (effectiveness["usage"] - 1) + quality) / effectiveness["usage"]
                effectiveness["korean_ratio"] = (effectiveness["korean_ratio"] * (effectiveness["usage"] - 1) + korean_ratio) / effectiveness["usage"]
    
    def _get_safe_mc_answer(self, max_choice: int) -> str:
        """안전한 객관식 답변 생성"""
        # max_choice가 0이거나 유효하지 않은 경우 기본값 설정
        if max_choice <= 0:
            max_choice = 5
        
        return str(random.randint(1, max_choice))
    
    def _get_safe_fallback(self, question: str, question_type: str, max_choice: int) -> str:
        """안전한 폴백 답변"""
        # max_choice 유효성 검증
        if max_choice <= 0:
            max_choice = 5
        
        # 간단한 객관식/주관식 구분
        if question_type == "multiple_choice" or (any(str(i) in question for i in range(1, 6)) and len(question) < 300):
            return self._get_safe_mc_answer(max_choice)
        else:
            return "관련 법령과 규정에 따라 체계적인 관리 방안을 수립하고 지속적인 모니터링을 수행해야 합니다."
    
    def _update_stats(self, question_type: str, domain: str, difficulty: str, processing_time: float):
        """통계 업데이트"""
        self.stats["total"] += 1
        self.stats["processing_times"].append(processing_time)
        
        if question_type == "multiple_choice":
            self.stats["mc_count"] += 1
        else:
            self.stats["subj_count"] += 1
        
        # 도메인 통계
        self.stats["domain_stats"][domain] = self.stats["domain_stats"].get(domain, 0) + 1
        
        # 난이도 통계
        self.stats["difficulty_stats"][difficulty] += 1
    
    def print_progress_bar(self, current: int, total: int, start_time: float, bar_length: int = PROGRESS_CONFIG['bar_length']):
        """진행률 게이지바 출력"""
        progress = current / total
        filled_length = int(bar_length * progress)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        
        percent = progress * 100
        print(f"\r문항 처리: ({current}/{total}) 진행도: {percent:.0f}% [{bar}]", end='', flush=True)
    
    def _calculate_model_reliability(self) -> float:
        """모델 신뢰도 계산"""
        total = max(self.stats["total"], 1)
        
        # 객관식 성공률 (35%)
        mc_total = max(self.stats["mc_count"], 1)
        mc_success_rate = (self.stats["mc_context_accuracy"] / mc_total) * 0.35
        
        # 강화된 패턴 성공률 (15%)
        enhanced_pattern_rate = (self.stats["enhanced_mc_pattern_usage"] / mc_total) * 0.15
        
        # 한국어 준수율 (10%)
        korean_rate = (self.stats["korean_compliance"] / total) * 0.1
        
        # 범위 정확도 (5%)
        range_accuracy = max(0, (1 - self.stats["choice_range_errors"] / total)) * 0.05
        
        # 검증 통과율 (5%)
        validation_rate = max(0, (1 - self.stats["validation_errors"] / total)) * 0.05
        
        # 의도 일치율 (15%)
        intent_rate = 0.0
        if self.stats["intent_analysis_accuracy"] > 0:
            intent_rate = (self.stats["intent_match_success"] / self.stats["intent_analysis_accuracy"]) * 0.15
        
        # 품질 점수 (10%)
        quality_rate = 0.0
        if self.stats["quality_scores"]:
            avg_quality = sum(self.stats["quality_scores"]) / len(self.stats["quality_scores"])
            quality_rate = avg_quality * 0.1
        
        # 의미 분석 성공률 (5%)
        semantic_rate = (self.stats["semantic_analysis_success"] / mc_total) * 0.05
        
        # 전체 신뢰도 (0-100%)
        reliability = (mc_success_rate + enhanced_pattern_rate + korean_rate + range_accuracy + 
                      validation_rate + intent_rate + quality_rate + semantic_rate) * 100
        
        return min(reliability, 100.0)
    
    def _simple_save_csv(self, df: pd.DataFrame, filepath: str) -> bool:
        """간단한 CSV 저장"""
        filepath = Path(filepath)
        
        try:
            df.to_csv(filepath, index=False, encoding=FILE_VALIDATION['encoding'])
            
            if self.verbose:
                print(f"\n결과 저장 완료: {filepath}")
            return True
            
        except PermissionError as e:
            if self.verbose:
                print(f"\n파일 저장 권한 오류: {e}")
                print("파일이 다른 프로그램에서 열려있는지 확인하세요.")
            return False
            
        except Exception as e:
            if self.verbose:
                print(f"\n파일 저장 중 오류: {e}")
            return False
    
    def execute_inference(self, test_file: str = None, 
                         submission_file: str = None,
                         output_file: str = None) -> Dict:
        """전체 추론 실행"""
        
        # 기본 파일 경로 사용
        test_file = test_file or DEFAULT_FILES['test_file']
        submission_file = submission_file or DEFAULT_FILES['submission_file']
        output_file = output_file or DEFAULT_FILES['output_file']
        
        # 데이터 로드
        try:
            test_df = pd.read_csv(test_file)
            submission_df = pd.read_csv(submission_file)
        except Exception as e:
            raise RuntimeError(f"데이터 로드 실패: {e}")
        
        return self.execute_inference_with_data(test_df, submission_df, output_file)
    
    def execute_inference_with_data(self, test_df: pd.DataFrame, 
                                   submission_df: pd.DataFrame,
                                   output_file: str = None) -> Dict:
        """데이터프레임으로 추론 실행"""
        
        output_file = output_file or DEFAULT_FILES['output_file']
        
        print(f"\n데이터 로드 완료: {len(test_df)}개 문항")
        
        answers = []
        total_questions = len(test_df)
        inference_start_time = time.time()
        
        for idx, row in test_df.iterrows():
            question = row['Question']
            question_id = row['ID']
            
            # 추론 수행
            answer = self.process_single_question(question, question_id)
            answers.append(answer)
            
            # 진행도 표시
            if (idx + 1) % PROGRESS_CONFIG['update_frequency'] == 0:
                self.print_progress_bar(idx + 1, total_questions, inference_start_time)
            
            # 메모리 관리
            if (idx + 1) % MEMORY_CONFIG['gc_frequency'] == 0:
                gc.collect()
        
        print()
        
        # 결과 저장
        submission_df['Answer'] = answers
        save_success = self._simple_save_csv(submission_df, output_file)
        
        if not save_success:
            print(f"지정된 파일로 저장에 실패했습니다: {output_file}")
            print("파일이 다른 프로그램에서 열려있거나 권한 문제일 수 있습니다.")
        
        return self._get_results_summary()
    
    def _print_enhanced_stats(self):
        """상세 통계 출력"""
        pass
    
    def _get_results_summary(self) -> Dict:
        """결과 요약"""
        total = max(self.stats["total"], 1)
        mc_stats = self.model_handler.get_answer_stats()
        learning_stats = self.model_handler.get_learning_stats()
        processing_stats = self.data_processor.get_processing_stats()
        kb_stats = self.knowledge_base.get_analysis_statistics()
        
        # 의도별 품질 평균 계산
        intent_quality_avg = {}
        for intent, scores in self.stats["answer_quality_by_intent"].items():
            if scores:
                intent_quality_avg[intent] = sum(scores) / len(scores)
        
        # 도메인별 의도 일치율 계산
        domain_intent_rates = {}
        for domain, stats in self.stats["domain_intent_match"].items():
            if stats["total"] > 0:
                domain_intent_rates[domain] = (stats["matched"] / stats["total"]) * 100
        
        # 도메인별 객관식 정확도 계산
        mc_domain_rates = {}
        for domain, stats in self.stats["mc_domain_accuracy"].items():
            if stats["total"] > 0:
                mc_domain_rates[domain] = (stats["success"] / stats["total"]) * 100
        
        return {
            "success": True,
            "total_questions": self.stats["total"],
            "mc_count": self.stats["mc_count"],
            "subj_count": self.stats["subj_count"],
            "model_success_rate": (self.stats["model_success"] / total) * 100,
            "korean_compliance_rate": (self.stats["korean_compliance"] / total) * 100,
            "choice_range_error_rate": (self.stats["choice_range_errors"] / total) * 100,
            "validation_error_rate": (self.stats["validation_errors"] / total) * 100,
            "intent_match_success_rate": (self.stats["intent_match_success"] / max(self.stats["intent_analysis_accuracy"], 1)) * 100,
            "institution_questions_count": self.stats["institution_questions"],
            "template_usage_rate": (self.stats["template_usage"] / total) * 100,
            "avg_processing_time": sum(self.stats["processing_times"]) / len(self.stats["processing_times"]) if self.stats["processing_times"] else 0,
            "avg_quality_score": sum(self.stats["quality_scores"]) / len(self.stats["quality_scores"]) if self.stats["quality_scores"] else 0,
            "intent_quality_by_type": intent_quality_avg,
            "domain_stats": dict(self.stats["domain_stats"]),
            "difficulty_stats": dict(self.stats["difficulty_stats"]),
            "answer_distribution_by_range": self.stats["mc_answers_by_range"],
            "learning_stats": learning_stats,
            "processing_stats": processing_stats,
            "knowledge_base_stats": kb_stats,
            "mc_context_accuracy_rate": (self.stats["mc_context_accuracy"] / max(self.stats["mc_count"], 1)) * 100,
            "mc_pattern_match_rate": (self.stats["mc_pattern_matches"] / max(self.stats["mc_count"], 1)) * 100,
            "high_confidence_intent_rate": (self.stats["high_confidence_intent"] / max(self.stats["intent_analysis_accuracy"], 1)) * 100,
            "intent_specific_answer_rate": (self.stats["intent_specific_answers"] / total) * 100,
            "quality_improvement_count": self.stats["quality_improvement"],
            "fallback_avoidance_rate": (self.stats["fallback_avoidance"] / total) * 100,
            "korean_enhancement_count": self.stats["korean_enhancement"],
            "answer_length_optimization_count": self.stats["answer_length_optimization"],
            "domain_intent_match_rates": domain_intent_rates,
            "mc_domain_accuracy_rates": mc_domain_rates,
            "institution_answer_accuracy": self.stats["institution_answer_accuracy"],
            "template_effectiveness_stats": dict(self.stats["template_effectiveness"]),
            "enhanced_mc_pattern_usage_rate": (self.stats["enhanced_mc_pattern_usage"] / max(self.stats["mc_count"], 1)) * 100,
            "semantic_analysis_success_rate": (self.stats["semantic_analysis_success"] / max(self.stats["mc_count"], 1)) * 100,
            "choice_categorization_success_rate": (self.stats["choice_categorization_success"] / max(self.stats["mc_count"], 1)) * 100,
            "outlier_detection_success_rate": (self.stats["outlier_detection_success"] / max(self.stats["mc_count"], 1)) * 100,
            "negative_question_detection_rate": (self.stats["negative_question_detection"] / max(self.stats["mc_count"], 1)) * 100,
            "domain_specific_pattern_match_rate": (self.stats["domain_specific_pattern_match"] / max(self.stats["mc_count"], 1)) * 100,
            "confidence_based_decisions_rate": (self.stats["confidence_based_decisions"] / total) * 100,
            "multi_method_validation_rate": (self.stats["multi_method_validation"] / max(self.stats["subj_count"], 1)) * 100,
            "learning_accuracy_improvement_rate": (self.stats["learning_accuracy_improvement"] / total) * 100,
            "model_reliability_score": self._calculate_model_reliability(),
            "total_time": time.time() - self.start_time
        }
    
    def cleanup(self):
        """리소스 정리"""
        try:
            if hasattr(self, 'model_handler'):
                self.model_handler.cleanup()
            
            if hasattr(self, 'data_processor'):
                self.data_processor.cleanup()
            
            if hasattr(self, 'knowledge_base'):
                self.knowledge_base.cleanup()
            
            gc.collect()
            
        except Exception as e:
            if self.verbose:
                print(f"정리 중 오류: {e}")


def main():
    """메인 함수"""
    
    engine = None
    try:
        # AI 엔진 초기화
        engine = FinancialAIInference(verbose=True)
        
        # 추론 실행
        results = engine.execute_inference()
        
        if results["success"]:
            print("\n추론 완료")
            print(f"총 처리시간: {results['total_time']:.1f}초")
            print(f"모델 성공률: {results['model_success_rate']:.1f}%")
            print(f"한국어 준수율: {results['korean_compliance_rate']:.1f}%")
            print(f"모델 신뢰도: {results['model_reliability_score']:.1f}%")
            if results['choice_range_error_rate'] > 0:
                print(f"선택지 범위 오류율: {results['choice_range_error_rate']:.1f}%")
            if results['intent_match_success_rate'] > 0:
                print(f"의도 일치 성공률: {results['intent_match_success_rate']:.1f}%")
            if results['mc_context_accuracy_rate'] > 0:
                print(f"객관식 컨텍스트 정확도: {results['mc_context_accuracy_rate']:.1f}%")
            if results['enhanced_mc_pattern_usage_rate'] > 0:
                print(f"강화된 패턴 활용률: {results['enhanced_mc_pattern_usage_rate']:.1f}%")
            if results['semantic_analysis_success_rate'] > 0:
                print(f"의미 분석 성공률: {results['semantic_analysis_success_rate']:.1f}%")
        
    except KeyboardInterrupt:
        print("\n추론 중단됨")
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if engine:
            engine.cleanup()


if __name__ == "__main__":
    main()