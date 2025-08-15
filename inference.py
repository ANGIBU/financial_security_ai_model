# inference.py

"""
금융보안 AI 추론 시스템
- Self-Consistency 기법 적용
- 의도별 특화 처리
- 신뢰도 보정 시스템
- 답변 품질 평가
- 성능 일관성 개선
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
        
        # 성능 통계 (강화)
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
            "mc_answers_by_range": {
                3: {"1": 0, "2": 0, "3": 0}, 
                4: {"1": 0, "2": 0, "3": 0, "4": 0}, 
                5: {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0}
            },
            "choice_range_errors": 0,
            "validation_errors": 0,
            "intent_analysis_success": 0,
            "intent_match_accuracy": 0,
            "institution_questions": 0,
            "template_usage": 0,
            "self_consistency_usage": 0,
            "confidence_scores": [],
            "answer_quality_by_intent": {},
            "error_patterns": {}
        }
        
        if verbose:
            print("초기화 완료")
        
    def process_single_question(self, question: str, question_id: str) -> str:
        """단일 질문 처리 (강화)"""
        start_time = time.time()
        
        try:
            # 1. 질문 분석 및 의도 파악
            question_type, max_choice = self.data_processor.extract_choice_range(question)
            
            # max_choice 검증 및 보정
            if question_type == "multiple_choice" and max_choice <= 0:
                max_choice = 5  # 기본값으로 설정
            
            domain = self.data_processor.extract_domain(question)
            difficulty = self.data_processor.analyze_question_difficulty(question)
            
            # 2. 지식베이스 분석
            kb_analysis = self.knowledge_base.analyze_question(question)
            
            # 3. 질문 의도 분석 (강화)
            intent_analysis = None
            if question_type == "subjective":
                intent_analysis = self.data_processor.analyze_question_intent(question)
                self.stats["intent_analysis_success"] += 1
                
                # 기관 관련 질문 확인
                if kb_analysis.get("institution_info", {}).get("is_institution_question", False):
                    self.stats["institution_questions"] += 1
            
            # 4. Self-Consistency 기법으로 답변 생성
            answer = self.model_handler.generate_answer(question, question_type, max_choice, intent_analysis)
            self.stats["self_consistency_usage"] += 1
            
            # 5. 답변 검증 및 품질 평가
            is_valid = self.data_processor.validate_korean_answer(answer, question_type, max_choice, question)
            
            if is_valid:
                # 의도 일치성 검증 (주관식)
                if question_type == "subjective" and intent_analysis:
                    intent_match = self.data_processor.validate_answer_intent_match(answer, question, intent_analysis)
                    if intent_match:
                        self.stats["intent_match_accuracy"] += 1
                    else:
                        # 의도 불일치시 특화 답변으로 대체
                        answer = self._generate_specialized_answer(question, intent_analysis, kb_analysis)
                        self.stats["template_usage"] += 1
                
                # 답변 정규화
                answer = self.data_processor.normalize_korean_answer(answer, question_type, max_choice)
                self.stats["model_success"] += 1
                
                # 객관식 분포 추적
                if question_type == "multiple_choice" and answer.isdigit():
                    answer_num = int(answer)
                    if 1 <= answer_num <= max_choice and max_choice in self.stats["mc_answers_by_range"]:
                        self.stats["mc_answers_by_range"][max_choice][answer] += 1
                
                # 주관식 품질 평가
                if question_type == "subjective":
                    korean_ratio = self.data_processor.calculate_korean_ratio(answer)
                    if korean_ratio >= 0.8:
                        self.stats["korean_compliance"] += 1
                        quality_score = self._calculate_comprehensive_quality(answer, question, intent_analysis)
                        self.stats["quality_scores"].append(quality_score)
                        self.stats["confidence_scores"].append(quality_score)
                        
                        # 의도별 품질 통계
                        if intent_analysis:
                            primary_intent = intent_analysis.get("primary_intent", "일반")
                            if primary_intent not in self.stats["answer_quality_by_intent"]:
                                self.stats["answer_quality_by_intent"][primary_intent] = []
                            self.stats["answer_quality_by_intent"][primary_intent].append(quality_score)
                    else:
                        # 한국어 비율 부족시 템플릿 대체
                        answer = self._get_enhanced_fallback_answer(question_type, domain, max_choice, intent_analysis, kb_analysis)
                        self.stats["korean_compliance"] += 1
                        self.stats["template_usage"] += 1
                else:
                    self.stats["korean_compliance"] += 1
                    
            else:
                # 검증 실패시 특화된 폴백
                self.stats["validation_errors"] += 1
                answer = self._get_enhanced_fallback_answer(question_type, domain, max_choice, intent_analysis, kb_analysis)
                
                # 오류 패턴 분석
                error_key = f"{question_type}_{domain}"
                if error_key not in self.stats["error_patterns"]:
                    self.stats["error_patterns"][error_key] = 0
                self.stats["error_patterns"][error_key] += 1
                
                # 객관식 분포 업데이트
                if question_type == "multiple_choice" and answer.isdigit():
                    answer_num = int(answer)
                    if 1 <= answer_num <= max_choice and max_choice in self.stats["mc_answers_by_range"]:
                        self.stats["mc_answers_by_range"][max_choice][answer] += 1
                
                self.stats["korean_compliance"] += 1
                self.stats["template_usage"] += 1
            
            # 6. 최종 범위 검증
            if question_type == "multiple_choice":
                if not answer.isdigit() or not (1 <= int(answer) <= max_choice):
                    self.stats["choice_range_errors"] += 1
                    # 안전한 답변으로 강제 수정
                    answer = self._get_safe_mc_answer(max_choice)
                    if max_choice in self.stats["mc_answers_by_range"]:
                        self.stats["mc_answers_by_range"][max_choice][answer] += 1
            
            # 7. 통계 업데이트
            self._update_comprehensive_stats(question_type, domain, difficulty, time.time() - start_time)
            
            return answer
            
        except Exception as e:
            if self.verbose:
                print(f"오류 발생: {e}")
            # 안전한 폴백 답변
            fallback = self._get_safe_fallback(question, max_choice if 'max_choice' in locals() and max_choice > 0 else 5)
            self._update_comprehensive_stats("multiple_choice", "일반", "초급", time.time() - start_time)
            return fallback
    
    def _generate_specialized_answer(self, question: str, intent_analysis: Dict, kb_analysis: Dict) -> str:
        """특화된 답변 생성"""
        primary_intent = intent_analysis.get("primary_intent", "일반")
        answer_type = intent_analysis.get("answer_type_required", "설명형")
        domain = self.data_processor.extract_domain(question)
        
        # 기관 관련 질문 특화 처리
        institution_info = kb_analysis.get("institution_info", {})
        if institution_info.get("is_institution_question", False):
            institution_type = institution_info.get("institution_type")
            if institution_type:
                return self.knowledge_base.get_institution_specific_answer(institution_type)
        
        # 의도별 템플릿 매핑
        intent_mapping = {
            "기관_요청": "기관_요청",
            "특징_분석": "특징_분석", 
            "지표_나열": "지표_나열",
            "절차_설명": "절차_설명",
            "법령_해석": "일반"
        }
        
        intent_key = intent_mapping.get(primary_intent, "일반")
        
        return self.knowledge_base.get_korean_subjective_template(domain, intent_key)
    
    def _calculate_comprehensive_quality(self, answer: str, question: str, intent_analysis: Dict = None) -> float:
        """종합 품질 평가"""
        if not answer:
            return 0.0
        
        score = 0.0
        
        # 한국어 비율 (25%)
        korean_ratio = self.data_processor.calculate_korean_ratio(answer)
        score += korean_ratio * 0.25
        
        # 길이 적절성 (20%)
        length = len(answer)
        if 50 <= length <= 400:
            score += 0.2
        elif 30 <= length < 50 or 400 < length <= 500:
            score += 0.15
        elif 20 <= length < 30:
            score += 0.1
        
        # 문장 구조 (15%)
        if answer.endswith(('.', '다', '요', '함')):
            score += 0.1
        
        sentences = answer.split('.')
        if len(sentences) >= 2:
            score += 0.05
        
        # 의도 일치성 (25%)
        if intent_analysis:
            intent_match = self.data_processor.validate_answer_intent_match(answer, question, intent_analysis)
            if intent_match:
                score += 0.25
            else:
                score += 0.1  # 부분 점수
        else:
            score += 0.15
        
        # 전문성 (15%)
        domain_keywords = self._get_domain_keywords(question)
        if domain_keywords:
            found_keywords = sum(1 for keyword in domain_keywords if keyword in answer)
            keyword_ratio = found_keywords / len(domain_keywords)
            score += keyword_ratio * 0.15
        else:
            score += 0.1
        
        return min(score, 1.0)
    
    def _get_domain_keywords(self, question: str) -> List[str]:
        """도메인별 키워드 반환"""
        question_lower = question.lower()
        
        if "개인정보" in question_lower:
            return ["개인정보보호법", "정보주체", "처리", "보호조치", "동의"]
        elif "전자금융" in question_lower:
            return ["전자금융거래법", "접근매체", "인증", "보안", "분쟁조정"]
        elif "보안" in question_lower or "악성코드" in question_lower:
            return ["보안정책", "탐지", "대응", "모니터링", "방어"]
        else:
            return ["법령", "규정", "관리", "조치", "절차"]
    
    def _get_enhanced_fallback_answer(self, question_type: str, domain: str, max_choice: int, intent_analysis: Dict = None, kb_analysis: Dict = None) -> str:
        """향상된 폴백 답변"""
        if question_type == "multiple_choice":
            # 모델 핸들러의 균등 분포 답변 사용
            return self.model_handler._get_balanced_mc_answer(max_choice)
        else:
            # 기관 관련 질문 특화 답변
            if kb_analysis and kb_analysis.get("institution_info", {}).get("is_institution_question", False):
                institution_type = kb_analysis["institution_info"].get("institution_type")
                if institution_type:
                    return self.knowledge_base.get_institution_specific_answer(institution_type)
            
            # 의도별 템플릿 사용
            if intent_analysis:
                primary_intent = intent_analysis.get("primary_intent", "일반")
                intent_mapping = {
                    "기관_요청": "기관_요청",
                    "특징_분석": "특징_분석",
                    "지표_나열": "지표_나열",
                    "절차_설명": "절차_설명"
                }
                intent_key = intent_mapping.get(primary_intent, "일반")
                return self.knowledge_base.get_korean_subjective_template(domain, intent_key)
            
            # 기본 템플릿
            return self.knowledge_base.get_korean_subjective_template(domain)
    
    def _get_safe_mc_answer(self, max_choice: int) -> str:
        """안전한 객관식 답변 생성"""
        import random
        # max_choice 검증
        if max_choice <= 0:
            max_choice = 5
        return str(random.randint(1, max_choice))
    
    def _get_safe_fallback(self, question: str, max_choice: int) -> str:
        """안전한 폴백 답변"""
        # 간단한 객관식/주관식 구분
        if any(str(i) in question for i in range(1, 6)) and len(question) < 300:
            return self._get_safe_mc_answer(max_choice)
        else:
            return "관련 법령과 규정에 따라 체계적인 관리 방안을 수립하고 지속적인 모니터링을 수행해야 합니다."
    
    def _update_comprehensive_stats(self, question_type: str, domain: str, difficulty: str, processing_time: float):
        """종합 통계 업데이트"""
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
    
    def _calculate_model_reliability(self) -> float:
        """모델 신뢰도 계산 (강화)"""
        total = max(self.stats["total"], 1)
        
        # 기본 성공률 (25%)
        success_rate = (self.stats["model_success"] / total) * 0.25
        
        # 한국어 준수율 (20%)
        korean_rate = (self.stats["korean_compliance"] / total) * 0.2
        
        # 범위 정확도 (15%)
        range_accuracy = max(0, (1 - self.stats["choice_range_errors"] / total)) * 0.15
        
        # 검증 통과율 (15%)
        validation_rate = max(0, (1 - self.stats["validation_errors"] / total)) * 0.15
        
        # 의도 일치율 (15%)
        intent_rate = 0.0
        if self.stats["intent_analysis_success"] > 0:
            intent_rate = (self.stats["intent_match_accuracy"] / self.stats["intent_analysis_success"]) * 0.15
        
        # Self-Consistency 활용률 (10%)
        consistency_rate = min(self.stats["self_consistency_usage"] / total, 1.0) * 0.1
        
        # 전체 신뢰도 (0-100%)
        reliability = (success_rate + korean_rate + range_accuracy + validation_rate + intent_rate + consistency_rate) * 100
        
        return min(reliability, 100.0)
    
    def _simple_save_csv(self, df: pd.DataFrame, filepath: str) -> bool:
        """간단한 CSV 저장"""
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
        
        # 진행률 표시 변수 초기화
        last_percent = -1
        
        for idx, row in test_df.iterrows():
            question = row['Question']
            question_id = row['ID']
            
            # 실제 추론 수행
            answer = self.process_single_question(question, question_id)
            answers.append(answer)
            
            # 통일된 진행률 표시 (test_runner 스타일)
            current_percent = int((idx + 1) / total_questions * 100)
            if current_percent != last_percent:
                elapsed_time = time.time() - inference_start_time
                filled_length = int(50 * (idx + 1) / total_questions)
                bar = '█' * filled_length + '░' * (50 - filled_length)
                print(f"\r문항 처리: ({idx + 1}/{total_questions}) 진행도: {current_percent}% [{bar}] ", end='', flush=True)
                last_percent = current_percent
            
            # 메모리 관리 (50문항마다)
            if (idx + 1) % 50 == 0:
                gc.collect()
        
        # 진행률 완료 후 줄바꿈
        print()
        
        # 모델 신뢰도 계산
        reliability_score = self._calculate_model_reliability()
        print(f"\n모델 신뢰도: {reliability_score:.1f}%")
        
        # 상세 통계 출력
        self._print_detailed_stats()
        
        # 결과 저장
        submission_df['Answer'] = answers
        save_success = self._simple_save_csv(submission_df, output_file)
        
        if not save_success:
            print(f"지정된 파일로 저장에 실패했습니다: {output_file}")
            print("파일이 다른 프로그램에서 열려있거나 권한 문제일 수 있습니다.")
        
        return self._get_comprehensive_results_summary()
    
    def _print_detailed_stats(self):
        """상세 통계 출력"""
        if not self.verbose:
            return
        
        print("\n상세 통계:")
        
        # Self-Consistency 통계
        if self.stats["self_consistency_usage"] > 0:
            consistency_rate = (self.stats["self_consistency_usage"] / self.stats["total"]) * 100
            print(f"  Self-Consistency 적용률: {consistency_rate:.1f}%")
        
        # 의도 분석 통계
        if self.stats["intent_analysis_success"] > 0:
            intent_accuracy = (self.stats["intent_match_accuracy"] / self.stats["intent_analysis_success"]) * 100
            print(f"  의도 일치 정확도: {intent_accuracy:.1f}%")
        
        # 기관 관련 질문 통계
        if self.stats["institution_questions"] > 0:
            print(f"  기관 관련 질문: {self.stats['institution_questions']}개")
        
        # 템플릿 사용 통계
        if self.stats["template_usage"] > 0:
            template_rate = (self.stats["template_usage"] / self.stats["total"]) * 100
            print(f"  템플릿 사용률: {template_rate:.1f}%")
        
        # 품질 점수 통계
        if self.stats["quality_scores"]:
            avg_quality = sum(self.stats["quality_scores"]) / len(self.stats["quality_scores"])
            print(f"  평균 품질 점수: {avg_quality:.2f}")
        
        # 의도별 품질 통계
        if self.stats["answer_quality_by_intent"]:
            print("  의도별 답변 품질:")
            for intent, scores in self.stats["answer_quality_by_intent"].items():
                if scores:
                    avg_quality = sum(scores) / len(scores)
                    print(f"    {intent}: {avg_quality:.2f} (평균)")
        
        # 오류 패턴 분석
        if self.stats["error_patterns"]:
            print("  오류 패턴:")
            for pattern, count in self.stats["error_patterns"].items():
                print(f"    {pattern}: {count}회")
    
    def _get_comprehensive_results_summary(self) -> Dict:
        """종합 결과 요약"""
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
        
        # 신뢰도 점수 평균
        avg_confidence = sum(self.stats["confidence_scores"]) / len(self.stats["confidence_scores"]) if self.stats["confidence_scores"] else 0
        
        return {
            "success": True,
            "total_questions": self.stats["total"],
            "mc_count": self.stats["mc_count"],
            "subj_count": self.stats["subj_count"],
            "model_success_rate": (self.stats["model_success"] / total) * 100,
            "korean_compliance_rate": (self.stats["korean_compliance"] / total) * 100,
            "choice_range_error_rate": (self.stats["choice_range_errors"] / total) * 100,
            "validation_error_rate": (self.stats["validation_errors"] / total) * 100,
            "intent_match_accuracy_rate": (self.stats["intent_match_accuracy"] / max(self.stats["intent_analysis_success"], 1)) * 100,
            "self_consistency_usage_rate": (self.stats["self_consistency_usage"] / total) * 100,
            "institution_questions_count": self.stats["institution_questions"],
            "template_usage_rate": (self.stats["template_usage"] / total) * 100,
            "avg_processing_time": sum(self.stats["processing_times"]) / len(self.stats["processing_times"]) if self.stats["processing_times"] else 0,
            "avg_quality_score": sum(self.stats["quality_scores"]) / len(self.stats["quality_scores"]) if self.stats["quality_scores"] else 0,
            "avg_confidence_score": avg_confidence,
            "intent_quality_by_type": intent_quality_avg,
            "domain_stats": dict(self.stats["domain_stats"]),
            "difficulty_stats": dict(self.stats["difficulty_stats"]),
            "error_patterns": dict(self.stats["error_patterns"]),
            "answer_distribution_by_range": self.stats["mc_answers_by_range"],
            "learning_stats": learning_stats,
            "processing_stats": processing_stats,
            "knowledge_base_stats": kb_stats,
            "model_reliability": self._calculate_model_reliability(),
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
            print(f"모델 신뢰도: {results['model_reliability']:.1f}%")
            if results['choice_range_error_rate'] > 0:
                print(f"선택지 범위 오류율: {results['choice_range_error_rate']:.1f}%")
            if results['intent_match_accuracy_rate'] > 0:
                print(f"의도 일치 정확도: {results['intent_match_accuracy_rate']:.1f}%")
            if results['self_consistency_usage_rate'] > 0:
                print(f"Self-Consistency 적용률: {results['self_consistency_usage_rate']:.1f}%")
        
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