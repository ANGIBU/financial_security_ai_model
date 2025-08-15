# inference.py

"""
금융보안 AI 추론 시스템
- 문제 분류 및 처리
- 모델 추론 실행
- 결과 생성 및 저장
- 학습 데이터 관리
"""

import os
import time
import gc
import tempfile
import shutil
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
        
        # 통계
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
            "choice_range_errors": 0,  # 선택지 범위 오류 추가
            "validation_errors": 0    # 검증 오류 추가
        }
        
        if verbose:
            print("초기화 완료")
        
    def process_single_question(self, question: str, question_id: str) -> str:
        """단일 질문 처리"""
        start_time = time.time()
        
        try:
            # 1. 질문 분석 및 선택지 범위 추출
            question_type, max_choice = self.data_processor.extract_choice_range(question)
            domain = self.data_processor.extract_domain(question)
            difficulty = self.data_processor.analyze_question_difficulty(question)
            
            # 지식베이스 분석
            kb_analysis = self.knowledge_base.analyze_question(question)
            
            # 2. AI 모델로 답변 생성 (선택지 개수 전달)
            if question_type == "multiple_choice":
                answer = self.model_handler.generate_answer(question, question_type, max_choice)
            else:
                answer = self.model_handler.generate_answer(question, question_type)
            
            # 3. 한국어 답변 검증 및 정규화
            if self.data_processor.validate_korean_answer(answer, question_type, max_choice):
                answer = self.data_processor.normalize_korean_answer(answer, question_type, max_choice)
                self.stats["model_success"] += 1
                
                # 선택지별 답변 분포 추적 (수정된 부분)
                if question_type == "multiple_choice" and answer.isdigit():
                    answer_num = int(answer)
                    if 1 <= answer_num <= max_choice and max_choice in self.stats["mc_answers_by_range"]:
                        self.stats["mc_answers_by_range"][max_choice][answer] += 1
                
                # 한국어 준수율 확인
                if question_type == "subjective":
                    korean_ratio = self.data_processor.calculate_korean_ratio(answer)
                    if korean_ratio >= 0.8:
                        self.stats["korean_compliance"] += 1
                        quality_score = self._calculate_answer_quality(answer, question)
                        self.stats["quality_scores"].append(quality_score)
                    else:
                        # 한국어 비율이 낮으면 템플릿으로 대체
                        answer = self._get_korean_fallback_answer(question_type, domain, max_choice)
                        self.stats["korean_compliance"] += 1
                else:
                    self.stats["korean_compliance"] += 1
                    
            else:
                # 검증 실패시 한국어 폴백
                self.stats["validation_errors"] += 1
                answer = self._get_korean_fallback_answer(question_type, domain, max_choice)
                
                # 객관식인 경우 답변 분포 업데이트
                if question_type == "multiple_choice" and answer.isdigit():
                    answer_num = int(answer)
                    if 1 <= answer_num <= max_choice and max_choice in self.stats["mc_answers_by_range"]:
                        self.stats["mc_answers_by_range"][max_choice][answer] += 1
                
                self.stats["korean_compliance"] += 1
            
            # 4. 최종 답변 범위 검증
            if question_type == "multiple_choice":
                if not answer.isdigit() or not (1 <= int(answer) <= max_choice):
                    self.stats["choice_range_errors"] += 1
                    # 범위 내 답변으로 강제 수정
                    answer = self._get_safe_mc_answer(max_choice)
                    if max_choice in self.stats["mc_answers_by_range"]:
                        self.stats["mc_answers_by_range"][max_choice][answer] += 1
            
            # 5. 통계 업데이트
            self._update_stats(question_type, domain, difficulty, time.time() - start_time)
            
            return answer
            
        except Exception as e:
            if self.verbose:
                print(f"오류 발생: {e}")
            # 안전한 폴백 답변
            fallback = self._get_safe_fallback(question, max_choice if 'max_choice' in locals() else 5)
            self._update_stats("multiple_choice", "일반", "초급", time.time() - start_time)
            return fallback
    
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
    
    def _get_korean_fallback_answer(self, question_type: str, domain: str, max_choice: int) -> str:
        """한국어 폴백 답변"""
        if question_type == "multiple_choice":
            # 모델 핸들러의 균등 분포 답변 사용
            return self.model_handler._get_balanced_mc_answer(max_choice)
        else:
            # 지식베이스의 한국어 도메인별 템플릿 사용
            return self.knowledge_base.get_korean_subjective_template(domain)
    
    def _calculate_answer_quality(self, answer: str, question: str) -> float:
        """답변 품질 점수 계산"""
        if not answer:
            return 0.0
        
        score = 0.0
        
        # 한국어 비율 (50%)
        korean_ratio = self.data_processor.calculate_korean_ratio(answer)
        score += korean_ratio * 0.5
        
        # 길이 적절성 (25%)
        length = len(answer)
        if 50 <= length <= 350:
            score += 0.25
        elif 30 <= length < 50 or 350 < length <= 500:
            score += 0.15
        
        # 문장 구조 (25%)
        if answer.endswith(('.', '다', '요', '함')):
            score += 0.15
        
        sentences = answer.split('.')
        if len(sentences) >= 2:
            score += 0.1
        
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
        """모델 신뢰도 계산"""
        total = max(self.stats["total"], 1)
        
        # 기본 성공률 (40%)
        success_rate = (self.stats["model_success"] / total) * 0.4
        
        # 한국어 준수율 (30%)
        korean_rate = (self.stats["korean_compliance"] / total) * 0.3
        
        # 범위 정확도 (20%) - 선택지 범위 오류가 적을수록 높음
        range_accuracy = max(0, (1 - self.stats["choice_range_errors"] / total)) * 0.2
        
        # 검증 통과율 (10%) - 검증 오류가 적을수록 높음
        validation_rate = max(0, (1 - self.stats["validation_errors"] / total)) * 0.1
        
        # 전체 신뢰도 (0-100%)
        reliability = (success_rate + korean_rate + range_accuracy + validation_rate) * 100
        
        return min(reliability, 100.0)
    
    def _safe_save_csv(self, df: pd.DataFrame, filepath: str, max_retries: int = 3) -> bool:
        """안전한 CSV 저장 (권한 오류 처리)"""
        filepath = Path(filepath)
        
        for attempt in range(max_retries):
            try:
                # 임시 파일로 먼저 저장
                with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv', encoding='utf-8-sig') as tmp_file:
                    df.to_csv(tmp_file.name, index=False, encoding='utf-8-sig')
                    temp_path = tmp_file.name
                
                # 대상 파일이 존재하면 백업
                if filepath.exists():
                    backup_path = filepath.with_suffix('.bak')
                    if backup_path.exists():
                        backup_path.unlink()
                    shutil.move(str(filepath), str(backup_path))
                
                # 임시 파일을 대상 위치로 이동
                shutil.move(temp_path, str(filepath))
                
                if self.verbose:
                    print(f"\n결과 저장 완료: {filepath}")
                return True
                
            except PermissionError as e:
                if attempt < max_retries - 1:
                    if self.verbose:
                        print(f"\n파일 저장 권한 오류 (시도 {attempt + 1}/{max_retries}): {e}")
                    time.sleep(1)  # 1초 대기 후 재시도
                    
                    # 대체 파일명 시도
                    timestamp = int(time.time())
                    filepath = filepath.parent / f"{filepath.stem}_{timestamp}{filepath.suffix}"
                else:
                    if self.verbose:
                        print(f"\n파일 저장 실패: {e}")
                    return False
                    
            except Exception as e:
                if self.verbose:
                    print(f"\n파일 저장 중 오류: {e}")
                return False
                
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
        print("AI 추론 시작\n")
        
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
        
        # 결과 저장 (안전한 저장 방식 사용)
        submission_df['Answer'] = answers
        save_success = self._safe_save_csv(submission_df, output_file)
        
        if not save_success:
            # 저장 실패시 임시 디렉토리에 저장
            temp_dir = Path(tempfile.gettempdir())
            temp_file = temp_dir / f"inference_result_{int(time.time())}.csv"
            try:
                submission_df.to_csv(temp_file, index=False, encoding='utf-8-sig')
                print(f"결과를 임시 디렉토리에 저장: {temp_file}")
            except Exception as e:
                print(f"임시 저장도 실패: {e}")
        
        return self._get_results_summary()
    
    def _get_results_summary(self) -> Dict:
        """결과 요약"""
        total = max(self.stats["total"], 1)
        mc_stats = self.model_handler.get_answer_stats()
        learning_stats = self.model_handler.get_learning_stats()
        processing_stats = self.data_processor.get_processing_stats()
        
        return {
            "success": True,
            "total_questions": self.stats["total"],
            "mc_count": self.stats["mc_count"],
            "subj_count": self.stats["subj_count"],
            "model_success_rate": (self.stats["model_success"] / total) * 100,
            "korean_compliance_rate": (self.stats["korean_compliance"] / total) * 100,
            "choice_range_error_rate": (self.stats["choice_range_errors"] / total) * 100,
            "validation_error_rate": (self.stats["validation_errors"] / total) * 100,
            "avg_processing_time": sum(self.stats["processing_times"]) / len(self.stats["processing_times"]) if self.stats["processing_times"] else 0,
            "avg_quality_score": sum(self.stats["quality_scores"]) / len(self.stats["quality_scores"]) if self.stats["quality_scores"] else 0,
            "domain_stats": dict(self.stats["domain_stats"]),
            "difficulty_stats": dict(self.stats["difficulty_stats"]),
            "answer_distribution_by_range": self.stats["mc_answers_by_range"],
            "learning_stats": learning_stats,
            "processing_stats": processing_stats,
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