# inference.py

"""
금융보안 AI 추론 시스템
- 문제 분류 및 처리
- 모델 추론 실행
- 결과 생성 및 저장
- 학습 데이터 관리
- 질문 의도 분석 및 답변 품질 검증 강화
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
    """금융보안 AI 추론 시스템"""
    
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
        
        # 통계 (강화)
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
            "answer_quality_by_intent": {}  # 의도별 답변 품질
        }
        
        if verbose:
            print("초기화 완료")
        
    def process_single_question(self, question: str, question_id: str) -> str:
        """단일 질문 처리 (강화)"""
        start_time = time.time()
        
        try:
            # 1. 질문 분석 및 선택지 범위 추출
            question_type, max_choice = self.data_processor.extract_choice_range(question)
            domain = self.data_processor.extract_domain(question)
            difficulty = self.data_processor.analyze_question_difficulty(question)
            
            # 2. 지식베이스 분석 (강화)
            kb_analysis = self.knowledge_base.analyze_question(question)
            
            # 3. 주관식 질문의 경우 의도 분석 수행
            intent_analysis = None
            if question_type == "subjective":
                intent_analysis = self.data_processor.analyze_question_intent(question)
                
                # 의도 분석 통계 업데이트
                self.stats["intent_analysis_accuracy"] += 1
                
                # 기관 관련 질문 확인
                if kb_analysis.get("institution_info", {}).get("is_institution_question", False):
                    self.stats["institution_questions"] += 1
            
            # 4. AI 모델로 답변 생성 (의도 정보 전달)
            if question_type == "multiple_choice":
                answer = self.model_handler.generate_answer(question, question_type, max_choice)
            else:
                answer = self.model_handler.generate_answer(question, question_type, max_choice, intent_analysis)
            
            # 5. 강화된 답변 검증 및 정규화
            is_valid = self.data_processor.validate_korean_answer(answer, question_type, max_choice, question)
            
            if is_valid:
                # 추가 의도 일치성 검증 (주관식)
                if question_type == "subjective" and intent_analysis:
                    intent_match = self.data_processor.validate_answer_intent_match(answer, question, intent_analysis)
                    if intent_match:
                        self.stats["intent_match_success"] += 1
                    else:
                        # 의도 불일치시 특화 답변 생성
                        answer = self._generate_intent_specific_answer(question, intent_analysis, kb_analysis)
                        self.stats["template_usage"] += 1
                
                answer = self.data_processor.normalize_korean_answer(answer, question_type, max_choice)
                self.stats["model_success"] += 1
                
                # 선택지별 답변 분포 추적
                if question_type == "multiple_choice" and answer.isdigit():
                    answer_num = int(answer)
                    if 1 <= answer_num <= max_choice and max_choice in self.stats["mc_answers_by_range"]:
                        self.stats["mc_answers_by_range"][max_choice][answer] += 1
                
                # 주관식 품질 평가
                if question_type == "subjective":
                    korean_ratio = self.data_processor.calculate_korean_ratio(answer)
                    if korean_ratio >= 0.8:
                        self.stats["korean_compliance"] += 1
                        quality_score = self._calculate_answer_quality(answer, question, intent_analysis)
                        self.stats["quality_scores"].append(quality_score)
                        
                        # 의도별 품질 통계
                        if intent_analysis:
                            primary_intent = intent_analysis.get("primary_intent", "일반")
                            if primary_intent not in self.stats["answer_quality_by_intent"]:
                                self.stats["answer_quality_by_intent"][primary_intent] = []
                            self.stats["answer_quality_by_intent"][primary_intent].append(quality_score)
                    else:
                        # 한국어 비율이 낮으면 템플릿으로 대체
                        answer = self._get_korean_fallback_answer(question_type, domain, max_choice, intent_analysis, kb_analysis)
                        self.stats["korean_compliance"] += 1
                        self.stats["template_usage"] += 1
                else:
                    self.stats["korean_compliance"] += 1
                    
            else:
                # 검증 실패시 강화된 폴백
                self.stats["validation_errors"] += 1
                answer = self._get_korean_fallback_answer(question_type, domain, max_choice, intent_analysis, kb_analysis)
                
                # 객관식인 경우 답변 분포 업데이트
                if question_type == "multiple_choice" and answer.isdigit():
                    answer_num = int(answer)
                    if 1 <= answer_num <= max_choice and max_choice in self.stats["mc_answers_by_range"]:
                        self.stats["mc_answers_by_range"][max_choice][answer] += 1
                
                self.stats["korean_compliance"] += 1
                self.stats["template_usage"] += 1
            
            # 6. 최종 답변 범위 검증
            if question_type == "multiple_choice":
                if not answer.isdigit() or not (1 <= int(answer) <= max_choice):
                    self.stats["choice_range_errors"] += 1
                    # 범위 내 답변으로 강제 수정
                    answer = self._get_safe_mc_answer(max_choice)
                    if max_choice in self.stats["mc_answers_by_range"]:
                        self.stats["mc_answers_by_range"][max_choice][answer] += 1
            
            # 7. 통계 업데이트
            self._update_stats(question_type, domain, difficulty, time.time() - start_time)
            
            return answer
            
        except Exception as e:
            if self.verbose:
                print(f"오류 발생: {e}")
            # 안전한 폴백 답변
            fallback = self._get_safe_fallback(question, max_choice if 'max_choice' in locals() else 5)
            self._update_stats("multiple_choice", "일반", "초급", time.time() - start_time)
            return fallback
    
    def _generate_intent_specific_answer(self, question: str, intent_analysis: Dict, kb_analysis: Dict) -> str:
        """의도별 특화 답변 생성 (신규)"""
        primary_intent = intent_analysis.get("primary_intent", "일반")
        answer_type = intent_analysis.get("answer_type_required", "설명형")
        domain = self.data_processor.extract_domain(question)
        
        # 기관 관련 질문인 경우
        institution_info = kb_analysis.get("institution_info", {})
        if institution_info.get("is_institution_question", False):
            institution_type = institution_info.get("institution_type")
            if institution_type:
                return self.knowledge_base.get_institution_specific_answer(institution_type)
        
        # 의도별 템플릿 사용
        if "기관" in primary_intent:
            intent_key = "기관_묻기"
        elif "특징" in primary_intent:
            intent_key = "특징_묻기"
        elif "지표" in primary_intent:
            intent_key = "지표_묻기"
        elif "방안" in primary_intent:
            intent_key = "방안_묻기"
        else:
            intent_key = "일반"
        
        return self.knowledge_base.get_korean_subjective_template(domain, intent_key)
    
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
    
    def _get_korean_fallback_answer(self, question_type: str, domain: str, max_choice: int, intent_analysis: Dict = None, kb_analysis: Dict = None) -> str:
        """한국어 폴백 답변 (강화)"""
        if question_type == "multiple_choice":
            # 모델 핸들러의 균등 분포 답변 사용
            return self.model_handler._get_balanced_mc_answer(max_choice)
        else:
            # 기관 관련 질문인 경우 특화 답변
            if kb_analysis and kb_analysis.get("institution_info", {}).get("is_institution_question", False):
                institution_type = kb_analysis["institution_info"].get("institution_type")
                if institution_type:
                    return self.knowledge_base.get_institution_specific_answer(institution_type)
            
            # 의도별 템플릿 사용
            if intent_analysis:
                primary_intent = intent_analysis.get("primary_intent", "일반")
                if "기관" in primary_intent:
                    intent_key = "기관_묻기"
                elif "특징" in primary_intent:
                    intent_key = "특징_묻기"
                elif "지표" in primary_intent:
                    intent_key = "지표_묻기"
                else:
                    intent_key = "일반"
                
                return self.knowledge_base.get_korean_subjective_template(domain, intent_key)
            
            # 기본 템플릿
            return self.knowledge_base.get_korean_subjective_template(domain)
    
    def _calculate_answer_quality(self, answer: str, question: str, intent_analysis: Dict = None) -> float:
        """답변 품질 점수 계산 (강화)"""
        if not answer:
            return 0.0
        
        score = 0.0
        
        # 한국어 비율 (40%)
        korean_ratio = self.data_processor.calculate_korean_ratio(answer)
        score += korean_ratio * 0.4
        
        # 길이 적절성 (20%)
        length = len(answer)
        if 50 <= length <= 350:
            score += 0.2
        elif 30 <= length < 50 or 350 < length <= 500:
            score += 0.1
        
        # 문장 구조 (15%)
        if answer.endswith(('.', '다', '요', '함')):
            score += 0.1
        
        sentences = answer.split('.')
        if len(sentences) >= 2:
            score += 0.05
        
        # 의도 일치성 (25%) - 강화
        if intent_analysis:
            intent_match = self.data_processor.validate_answer_intent_match(answer, question, intent_analysis)
            if intent_match:
                score += 0.25
        else:
            score += 0.15  # 의도 분석이 없는 경우 기본 점수
        
        return min(score, 1.0)
    
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
        
        # 기본 성공률 (30%)
        success_rate = (self.stats["model_success"] / total) * 0.3
        
        # 한국어 준수율 (25%)
        korean_rate = (self.stats["korean_compliance"] / total) * 0.25
        
        # 범위 정확도 (15%) - 선택지 범위 오류가 적을수록 높음
        range_accuracy = max(0, (1 - self.stats["choice_range_errors"] / total)) * 0.15
        
        # 검증 통과율 (10%) - 검증 오류가 적을수록 높음
        validation_rate = max(0, (1 - self.stats["validation_errors"] / total)) * 0.1
        
        # 의도 일치율 (20%) - 의도 분석 성공률
        intent_rate = 0.0
        if self.stats["intent_analysis_accuracy"] > 0:
            intent_rate = (self.stats["intent_match_success"] / self.stats["intent_analysis_accuracy"]) * 0.2
        
        # 전체 신뢰도 (0-100%)
        reliability = (success_rate + korean_rate + range_accuracy + validation_rate + intent_rate) * 100
        
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
        """강화된 통계 출력 (신규)"""
        if not self.verbose:
            return
        
        print("\n상세 통계:")
        
        # 의도 분석 통계
        if self.stats["intent_analysis_accuracy"] > 0:
            intent_success_rate = (self.stats["intent_match_success"] / self.stats["intent_analysis_accuracy"]) * 100
            print(f"  의도 일치 성공률: {intent_success_rate:.1f}%")
        
        # 기관 관련 질문 통계
        if self.stats["institution_questions"] > 0:
            print(f"  기관 관련 질문: {self.stats['institution_questions']}개")
        
        # 템플릿 사용 통계
        if self.stats["template_usage"] > 0:
            template_rate = (self.stats["template_usage"] / self.stats["total"]) * 100
            print(f"  템플릿 사용률: {template_rate:.1f}%")
        
        # 의도별 품질 통계
        if self.stats["answer_quality_by_intent"]:
            print("  의도별 답변 품질:")
            for intent, scores in self.stats["answer_quality_by_intent"].items():
                if scores:
                    avg_quality = sum(scores) / len(scores)
                    print(f"    {intent}: {avg_quality:.2f} (평균)")
    
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