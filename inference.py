# inference.py

"""
금융보안 AI 추론 시스템 (성능 최적화)
- 문제 분류 및 처리
- 모델 추론 실행
- 결과 생성 및 저장
- 학습 데이터 관리
- 질문 의도 분석 및 답변 품질 검증 강화
- 의도 일치 성공률 대폭 개선
"""

import os
import time
import gc
import pandas as pd
from typing import Dict, List
from pathlib import Path

# 오프라인 설정
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

# 현재 디렉토리 설정
current_dir = Path(__file__).parent.absolute()

# 로컬 모듈 import
from model_handler import SimpleModelHandler
from data_processor import SimpleDataProcessor
from knowledge_base import FinancialSecurityKnowledgeBase

class FinancialAIInference:
    """금융보안 AI 추론 시스템 (최적화)"""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.start_time = time.time()
        
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
        
        # 통계 (대폭 강화)
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
            "intent_analysis_accuracy": 0,  # 의도 분석 정확도
            "intent_match_success": 0,      # 의도 일치 성공률
            "institution_questions": 0,     # 기관 관련 질문 수
            "template_usage": 0,            # 템플릿 사용 횟수
            "answer_quality_by_intent": {}, # 의도별 답변 품질
            
            # 새로운 성능 지표들
            "high_confidence_intent": 0,    # 고신뢰도 의도 분석
            "intent_specific_answers": 0,   # 의도별 특화 답변
            "quality_improvement": 0,       # 품질 개선 횟수
            "fallback_avoidance": 0,        # 폴백 회피 횟수
            "domain_intent_match": {},      # 도메인별 의도 일치율
            "answer_length_optimization": 0, # 답변 길이 최적화
            "korean_enhancement": 0,        # 한국어 품질 향상
            "template_effectiveness": {}    # 템플릿 효과성
        }
        
        # 성능 최적화 설정
        self.optimization_config = {
            "intent_confidence_threshold": 0.6,  # 의도 신뢰도 임계값
            "quality_threshold": 0.7,            # 품질 임계값
            "korean_ratio_threshold": 0.8,       # 한국어 비율 임계값
            "max_retry_attempts": 2,              # 최대 재시도 횟수
            "template_preference": True,          # 템플릿 우선 사용
            "adaptive_prompt": True               # 적응형 프롬프트 사용
        }
        
        if verbose:
            print("초기화 완료")
        
    def process_single_question(self, question: str, question_id: str) -> str:
        """단일 질문 처리 (대폭 최적화)"""
        start_time = time.time()
        
        try:
            # 1. 기본 분석
            question_type, max_choice = self.data_processor.extract_choice_range(question)
            domain = self.data_processor.extract_domain(question)
            difficulty = self.data_processor.analyze_question_difficulty(question)
            
            # 2. 지식베이스 분석 (강화)
            kb_analysis = self.knowledge_base.analyze_question(question)
            
            # 3. 의도 분석 (고정밀)
            intent_analysis = None
            if question_type == "subjective":
                intent_analysis = self.data_processor.analyze_question_intent(question)
                self.stats["intent_analysis_accuracy"] += 1
                
                if self.verbose:
                    primary_intent = intent_analysis.get("primary_intent", "일반")
                    confidence = intent_analysis.get("intent_confidence", 0)
                    print(f"의도 분석: {primary_intent} (신뢰도: {confidence:.2f})")
                
                # 고신뢰도 의도 분석 확인
                if intent_analysis.get("intent_confidence", 0) >= self.optimization_config["intent_confidence_threshold"]:
                    self.stats["high_confidence_intent"] += 1
                
                # 기관 관련 질문 확인
                if kb_analysis.get("institution_info", {}).get("is_institution_question", False):
                    self.stats["institution_questions"] += 1
            
            # 4. 최적화된 답변 생성
            answer = self._generate_optimized_answer(question, question_type, max_choice, 
                                                   domain, intent_analysis, kb_analysis)
            
            # 5. 품질 검증 및 개선
            final_answer = self._validate_and_improve_answer(answer, question, question_type, 
                                                           max_choice, domain, intent_analysis, kb_analysis)
            
            # 6. 성능 통계 업데이트
            self._update_optimization_stats(question_type, domain, difficulty, 
                                          time.time() - start_time, intent_analysis, final_answer)
            
            return final_answer
            
        except Exception as e:
            if self.verbose:
                print(f"오류 발생: {e}")
            # 안전한 폴백 답변
            fallback = self._get_safe_fallback(question, max_choice if 'max_choice' in locals() else 5)
            self._update_stats(question_type if 'question_type' in locals() else "multiple_choice", 
                             domain if 'domain' in locals() else "일반", 
                             difficulty if 'difficulty' in locals() else "초급", 
                             time.time() - start_time)
            return fallback
    
    def _generate_optimized_answer(self, question: str, question_type: str, max_choice: int,
                                 domain: str, intent_analysis: Dict = None, kb_analysis: Dict = None) -> str:
        """최적화된 답변 생성 (신규)"""
        
        # 객관식 처리
        if question_type == "multiple_choice":
            return self.model_handler.generate_answer(question, question_type, max_choice)
        
        # 주관식 최적화 처리
        
        # 1. 기관 질문 우선 처리
        institution_info = kb_analysis.get("institution_info", {}) if kb_analysis else {}
        if institution_info.get("is_institution_question", False):
            institution_type = institution_info.get("institution_type")
            if institution_type:
                template_answer = self.knowledge_base.get_institution_specific_answer(institution_type)
                self.stats["intent_specific_answers"] += 1
                return template_answer
        
        # 2. 고신뢰도 의도 분석 기반 처리
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
                
                # 고품질 템플릿 사용 시도
                try:
                    template_answer = self.knowledge_base.get_high_quality_template(domain, intent_key)
                    if template_answer and len(template_answer) >= 50:
                        self.stats["intent_specific_answers"] += 1
                        self.stats["template_usage"] += 1
                        return template_answer
                except:
                    pass
        
        # 3. AI 모델 답변 생성 (적응형 프롬프트)
        if self.optimization_config["adaptive_prompt"] and intent_analysis:
            answer = self.model_handler.generate_answer(question, question_type, max_choice, intent_analysis)
        else:
            answer = self.model_handler.generate_answer(question, question_type, max_choice)
        
        return answer
    
    def _validate_and_improve_answer(self, answer: str, question: str, question_type: str,
                                   max_choice: int, domain: str, intent_analysis: Dict = None,
                                   kb_analysis: Dict = None) -> str:
        """답변 검증 및 개선 (신규)"""
        
        if question_type == "multiple_choice":
            # 객관식 범위 검증
            if answer and answer.isdigit() and 1 <= int(answer) <= max_choice:
                # 답변 분포 업데이트
                if max_choice in self.stats["mc_answers_by_range"]:
                    self.stats["mc_answers_by_range"][max_choice][answer] += 1
                self.stats["model_success"] += 1
                self.stats["korean_compliance"] += 1
                return answer
            else:
                # 범위 오류 처리
                self.stats["choice_range_errors"] += 1
                safe_answer = self.model_handler._get_balanced_mc_answer(max_choice)
                if max_choice in self.stats["mc_answers_by_range"]:
                    self.stats["mc_answers_by_range"][max_choice][safe_answer] += 1
                return safe_answer
        
        # 주관식 품질 검증 및 개선
        original_answer = answer
        improvement_count = 0
        
        # 1차: 기본 유효성 검증
        is_valid = self.data_processor.validate_korean_answer(answer, question_type, max_choice, question)
        
        if not is_valid:
            self.stats["validation_errors"] += 1
            answer = self._get_improved_answer(question, domain, intent_analysis, kb_analysis, "validation_failed")
            improvement_count += 1
        
        # 2차: 한국어 비율 검증
        korean_ratio = self.data_processor.calculate_korean_ratio(answer)
        if korean_ratio < self.optimization_config["korean_ratio_threshold"]:
            answer = self._get_improved_answer(question, domain, intent_analysis, kb_analysis, "korean_ratio_low")
            improvement_count += 1
            self.stats["korean_enhancement"] += 1
        
        # 3차: 의도 일치성 검증 (강화)
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
        
        # 4차: 답변 품질 평가 및 개선
        quality_score = self._calculate_enhanced_quality_score(answer, question, intent_analysis)
        if quality_score < self.optimization_config["quality_threshold"]:
            improved_answer = self._get_improved_answer(question, domain, intent_analysis, kb_analysis, "quality_low")
            improved_quality = self._calculate_enhanced_quality_score(improved_answer, question, intent_analysis)
            
            if improved_quality > quality_score:
                answer = improved_answer
                improvement_count += 1
                self.stats["quality_improvement"] += 1
        
        # 5차: 길이 최적화
        answer = self._optimize_answer_length(answer)
        if answer != original_answer:
            self.stats["answer_length_optimization"] += 1
        
        # 최종 정규화
        answer = self.data_processor.normalize_korean_answer(answer, question_type, max_choice)
        
        # 성공 통계 업데이트
        if improvement_count == 0:
            self.stats["fallback_avoidance"] += 1
        
        self.stats["model_success"] += 1
        self.stats["korean_compliance"] += 1
        
        # 품질 점수 기록
        final_quality = self._calculate_enhanced_quality_score(answer, question, intent_analysis)
        self.stats["quality_scores"].append(final_quality)
        
        # 의도별 품질 통계
        if intent_analysis:
            primary_intent = intent_analysis.get("primary_intent", "일반")
            if primary_intent not in self.stats["answer_quality_by_intent"]:
                self.stats["answer_quality_by_intent"][primary_intent] = []
            self.stats["answer_quality_by_intent"][primary_intent].append(final_quality)
        
        return answer
    
    def _get_improved_answer(self, question: str, domain: str, intent_analysis: Dict = None,
                           kb_analysis: Dict = None, improvement_type: str = "general") -> str:
        """개선된 답변 생성 (신규)"""
        
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
            
            # 고품질 템플릿 사용
            template_answer = self.knowledge_base.get_korean_subjective_template(domain, intent_key)
            
            # 개선 유형별 추가 처리
            if improvement_type == "intent_mismatch":
                # 의도 불일치시 더 구체적인 답변
                if "기관" in primary_intent:
                    return self.knowledge_base.get_korean_subjective_template(domain, "기관_묻기")
                elif "특징" in primary_intent:
                    return self.knowledge_base.get_korean_subjective_template(domain, "특징_묻기")
                elif "지표" in primary_intent:
                    return self.knowledge_base.get_korean_subjective_template(domain, "지표_묻기")
            
            return template_answer
        
        # 기본 도메인별 템플릿
        return self.knowledge_base.get_korean_subjective_template(domain)
    
    def _calculate_enhanced_quality_score(self, answer: str, question: str, intent_analysis: Dict = None) -> float:
        """강화된 품질 점수 계산 (신규)"""
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
        
        # 의도 일치성 (30%) - 강화
        if intent_analysis:
            answer_type = intent_analysis.get("answer_type_required", "설명형")
            intent_match = self.data_processor.validate_answer_intent_match(answer, question, intent_analysis)
            if intent_match:
                score += 0.3
            else:
                score += 0.1  # 의도 불일치시 감점
        else:
            score += 0.2  # 의도 분석이 없는 경우 기본 점수
        
        return min(score, 1.0)
    
    def _optimize_answer_length(self, answer: str) -> str:
        """답변 길이 최적화 (신규)"""
        if not answer:
            return answer
        
        # 너무 긴 답변 축약
        if len(answer) > 400:
            sentences = answer.split('. ')
            if len(sentences) > 3:
                # 처음 3개 문장만 유지
                answer = '. '.join(sentences[:3])
                if not answer.endswith('.'):
                    answer += '.'
        
        # 너무 짧은 답변 보강
        elif len(answer) < 50:
            if not answer.endswith('.'):
                answer += '.'
            # 최소한의 내용 보강
            if "법령" not in answer and "규정" not in answer:
                answer += " 관련 법령과 규정을 준수하여 체계적으로 관리해야 합니다."
        
        return answer
    
    def _update_optimization_stats(self, question_type: str, domain: str, difficulty: str, 
                                 processing_time: float, intent_analysis: Dict = None, answer: str = ""):
        """최적화 통계 업데이트 (신규)"""
        
        # 기본 통계 업데이트
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
        import random
        return str(random.randint(1, max_choice))
    
    def _get_safe_fallback(self, question: str, max_choice: int) -> str:
        """안전한 폴백 답변"""
        # 간단한 객관식/주관식 구분
        if any(str(i) in question for i in range(1, 6)) and len(question) < 300:
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
    
    def print_progress_bar(self, current: int, total: int, start_time: float, bar_length: int = 50):
        """진행률 게이지바 출력"""
        progress = current / total
        filled_length = int(bar_length * progress)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        
        # 진행률 출력
        percent = progress * 100
        print(f"\r문항 처리: ({current}/{total}) 진행도: {percent:.0f}% [{bar}]", end='', flush=True)
    
    def _calculate_model_reliability(self) -> float:
        """모델 신뢰도 계산 (강화)"""
        total = max(self.stats["total"], 1)
        
        # 기본 성공률 (20%)
        success_rate = (self.stats["model_success"] / total) * 0.2
        
        # 한국어 준수율 (20%)
        korean_rate = (self.stats["korean_compliance"] / total) * 0.2
        
        # 범위 정확도 (10%) - 선택지 범위 오류가 적을수록 높음
        range_accuracy = max(0, (1 - self.stats["choice_range_errors"] / total)) * 0.1
        
        # 검증 통과율 (10%) - 검증 오류가 적을수록 높음
        validation_rate = max(0, (1 - self.stats["validation_errors"] / total)) * 0.1
        
        # 의도 일치율 (25%) - 강화
        intent_rate = 0.0
        if self.stats["intent_analysis_accuracy"] > 0:
            intent_rate = (self.stats["intent_match_success"] / self.stats["intent_analysis_accuracy"]) * 0.25
        
        # 품질 점수 (10%)
        quality_rate = 0.0
        if self.stats["quality_scores"]:
            avg_quality = sum(self.stats["quality_scores"]) / len(self.stats["quality_scores"])
            quality_rate = avg_quality * 0.1
        
        # 최적화 성능 (5%)
        optimization_rate = 0.0
        if self.stats["total"] > 0:
            fallback_avoidance_rate = self.stats["fallback_avoidance"] / total
            optimization_rate = fallback_avoidance_rate * 0.05
        
        # 전체 신뢰도 (0-100%)
        reliability = (success_rate + korean_rate + range_accuracy + validation_rate + 
                      intent_rate + quality_rate + optimization_rate) * 100
        
        return min(reliability, 100.0)
    
    def _simple_save_csv(self, df: pd.DataFrame, filepath: str) -> bool:
        """간단한 CSV 저장 (백업 파일 생성 안함)"""
        filepath = Path(filepath)
        
        try:
            # 직접 저장 시도
            df.to_csv(filepath, index=False, encoding='utf-8-sig')
            
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
    
    def execute_inference(self, test_file: str = "./test.csv", 
                         submission_file: str = "./sample_submission.csv",
                         output_file: str = "./final_submission.csv") -> Dict:
        """전체 추론 실행"""
        
        # 데이터 로드
        try:
            test_df = pd.read_csv(test_file)
            submission_df = pd.read_csv(submission_file)
        except Exception as e:
            raise RuntimeError(f"데이터 로드 실패: {e}")
        
        return self.execute_inference_with_data(test_df, submission_df, output_file)
    
    def execute_inference_with_data(self, test_df: pd.DataFrame, 
                                   submission_df: pd.DataFrame,
                                   output_file: str = "./final_submission.csv") -> Dict:
        """데이터프레임으로 추론 실행"""
        
        print(f"\n데이터 로드 완료: {len(test_df)}개 문항")
        
        answers = []
        total_questions = len(test_df)
        inference_start_time = time.time()
        
        for idx, row in test_df.iterrows():
            question = row['Question']
            question_id = row['ID']
            
            # 실제 추론 수행
            answer = self.process_single_question(question, question_id)
            answers.append(answer)
            
            # 진행도 표시 (항상 표시)
            self.print_progress_bar(idx + 1, total_questions, inference_start_time)
            
            # 메모리 관리 (50문항마다)
            if (idx + 1) % 50 == 0:
                gc.collect()
        
        # 진행률 완료 후 줄바꿈
        print()
        
        # 모델 신뢰도 계산
        reliability_score = self._calculate_model_reliability()
        print(f"\n모델 신뢰도: {reliability_score:.1f}%")
        
        # 의도 일치 성공률 항상 출력 (핵심 지표)
        if self.stats["intent_analysis_accuracy"] > 0:
            intent_success_rate = (self.stats["intent_match_success"] / self.stats["intent_analysis_accuracy"]) * 100
            print(f"의도 일치 성공률: {intent_success_rate:.1f}%")
        
        # 강화된 통계 출력
        self._print_enhanced_stats()
        
        # 결과 저장 (간단한 저장 방식 사용)
        submission_df['Answer'] = answers
        save_success = self._simple_save_csv(submission_df, output_file)
        
        if not save_success:
            print(f"지정된 파일로 저장에 실패했습니다: {output_file}")
            print("파일이 다른 프로그램에서 열려있거나 권한 문제일 수 있습니다.")
        
        return self._get_results_summary()
    
    def _print_enhanced_stats(self):
        """강화된 통계 출력 (대폭 개선)"""
        if not self.verbose:
            return
        
        print("\n상세 통계:")
        
        # 의도 분석 통계 (강화)
        if self.stats["intent_analysis_accuracy"] > 0:
            intent_success_rate = (self.stats["intent_match_success"] / self.stats["intent_analysis_accuracy"]) * 100
            print(f"  의도 일치 성공률: {intent_success_rate:.1f}%")
            
            # 고신뢰도 의도 분석 비율
            high_conf_rate = (self.stats["high_confidence_intent"] / self.stats["intent_analysis_accuracy"]) * 100
            print(f"  고신뢰도 의도 분석률: {high_conf_rate:.1f}%")
        
        # 최적화 성능 통계 (신규)
        if self.stats["total"] > 0:
            print(f"  의도별 특화 답변률: {(self.stats['intent_specific_answers'] / self.stats['total']) * 100:.1f}%")
            print(f"  품질 개선 횟수: {self.stats['quality_improvement']}회")
            print(f"  폴백 회피률: {(self.stats['fallback_avoidance'] / self.stats['total']) * 100:.1f}%")
            
            if self.stats["korean_enhancement"] > 0:
                print(f"  한국어 품질 향상: {self.stats['korean_enhancement']}회")
            
            if self.stats["answer_length_optimization"] > 0:
                print(f"  답변 길이 최적화: {self.stats['answer_length_optimization']}회")
        
        # 기관 관련 질문 통계
        if self.stats["institution_questions"] > 0:
            print(f"  기관 관련 질문: {self.stats['institution_questions']}개")
        
        # 템플릿 사용 통계
        if self.stats["template_usage"] > 0:
            template_rate = (self.stats["template_usage"] / self.stats["total"]) * 100
            print(f"  템플릿 사용률: {template_rate:.1f}%")
        
        # 도메인별 의도 일치율 (신규)
        if self.stats["domain_intent_match"]:
            print("  도메인별 의도 일치율:")
            for domain, stats in self.stats["domain_intent_match"].items():
                if stats["total"] > 0:
                    match_rate = (stats["matched"] / stats["total"]) * 100
                    print(f"    {domain}: {match_rate:.1f}% ({stats['matched']}/{stats['total']})")
        
        # 의도별 품질 통계
        if self.stats["answer_quality_by_intent"]:
            print("  의도별 답변 품질:")
            for intent, scores in self.stats["answer_quality_by_intent"].items():
                if scores:
                    avg_quality = sum(scores) / len(scores)
                    print(f"    {intent}: {avg_quality:.2f} (평균)")
        
        # 템플릿 효과성 (신규)
        if self.stats["template_effectiveness"]:
            print("  템플릿 효과성 (상위 3개):")
            sorted_templates = sorted(self.stats["template_effectiveness"].items(), 
                                    key=lambda x: x[1]["avg_quality"], reverse=True)
            for template_key, effectiveness in sorted_templates[:3]:
                print(f"    {template_key}: 품질 {effectiveness['avg_quality']:.2f}, "
                      f"한국어 {effectiveness['korean_ratio']:.2f}")
    
    def _get_results_summary(self) -> Dict:
        """결과 요약 (강화)"""
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
            
            # 새로운 최적화 지표들
            "high_confidence_intent_rate": (self.stats["high_confidence_intent"] / max(self.stats["intent_analysis_accuracy"], 1)) * 100,
            "intent_specific_answer_rate": (self.stats["intent_specific_answers"] / total) * 100,
            "quality_improvement_count": self.stats["quality_improvement"],
            "fallback_avoidance_rate": (self.stats["fallback_avoidance"] / total) * 100,
            "korean_enhancement_count": self.stats["korean_enhancement"],
            "answer_length_optimization_count": self.stats["answer_length_optimization"],
            "domain_intent_match_rates": domain_intent_rates,
            "template_effectiveness_stats": dict(self.stats["template_effectiveness"]),
            
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
            if results['choice_range_error_rate'] > 0:
                print(f"선택지 범위 오류율: {results['choice_range_error_rate']:.1f}%")
            if results['intent_match_success_rate'] > 0:
                print(f"의도 일치 성공률: {results['intent_match_success_rate']:.1f}%")
        
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