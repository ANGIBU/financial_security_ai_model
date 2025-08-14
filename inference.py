# inference.py

"""
금융보안 AI 추론 시스템
- 문제 분류 및 처리
- 모델 추론 실행
- 결과 생성 및 저장
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
        
        print("=" * 50)
        print("금융보안 AI 추론 시스템 초기화")
        print("=" * 50)
        
        # 컴포넌트 초기화
        print("1/3 모델 핸들러 초기화...")
        self.model_handler = SimpleModelHandler(verbose=verbose)
        
        print("2/3 데이터 프로세서 초기화...")
        self.data_processor = SimpleDataProcessor()
        
        print("3/3 지식베이스 초기화...")
        self.knowledge_base = FinancialSecurityKnowledgeBase()
        
        # 통계
        self.stats = {
            "total": 0,
            "mc_count": 0,
            "subj_count": 0,
            "model_success": 0,
            "processing_times": [],
            "domain_stats": {},
            "difficulty_stats": {"초급": 0, "중급": 0, "고급": 0}
        }
        
        print("초기화 완료")
        
    def process_single_question(self, question: str, question_id: str) -> str:
        """단일 질문 처리"""
        start_time = time.time()
        
        try:
            # 1. 질문 분석
            question_type = self.data_processor.analyze_question_type(question)
            domain = self.data_processor.extract_domain(question)
            difficulty = self.data_processor.analyze_question_difficulty(question)
            
            # 지식베이스 분석
            kb_analysis = self.knowledge_base.analyze_question(question)
            
            if self.verbose:
                print(f"질문 {question_id}: {question_type}, {domain}, {difficulty}")
            
            # 2. AI 모델로 답변 생성
            answer = self.model_handler.generate_answer(question, question_type)
            
            # 3. 답변 검증 및 정규화
            if self.data_processor.validate_answer(answer, question_type):
                answer = self.data_processor.normalize_answer(answer, question_type)
                self.stats["model_success"] += 1
            else:
                # 검증 실패시 폴백
                answer = self._get_fallback_answer(question_type, domain)
            
            # 4. 통계 업데이트
            self._update_stats(question_type, domain, difficulty, time.time() - start_time)
            
            return answer
            
        except Exception as e:
            if self.verbose:
                print(f"오류 발생: {e}")
            return self._get_fallback_answer("multiple_choice", "일반")
    
    def _get_fallback_answer(self, question_type: str, domain: str) -> str:
        """폴백 답변"""
        if question_type == "multiple_choice":
            # 모델 핸들러의 균등 분포 답변 사용
            return self.model_handler._get_balanced_mc_answer()
        else:
            # 지식베이스의 도메인별 템플릿 사용
            return self.knowledge_base.get_subjective_template(domain)
    
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
        
        print(f"\n데이터 로드 완료: {len(test_df)}개 문항")
        
        # 전체 추론 진행
        print("=" * 50)
        print("AI 추론 시작")
        print("=" * 50)
        
        answers = []
        total_questions = len(test_df)
        
        for idx, row in test_df.iterrows():
            question = row['Question']
            question_id = row['ID']
            
            # 진행률 표시
            if (idx + 1) % 10 == 0 or idx == 0:
                progress = (idx + 1) / total_questions * 100
                elapsed = time.time() - self.start_time
                eta = (elapsed / (idx + 1)) * (total_questions - idx - 1) if idx > 0 else 0
                print(f"진행: {idx+1}/{total_questions} ({progress:.1f}%) - 남은시간: {eta/60:.1f}분")
            
            # 실제 추론 수행
            answer = self.process_single_question(question, question_id)
            answers.append(answer)
            
            # 중간 통계 (100문항마다)
            if (idx + 1) % 100 == 0:
                self._print_interim_stats()
            
            # 메모리 관리 (50문항마다)
            if (idx + 1) % 50 == 0:
                gc.collect()
        
        # 결과 저장
        submission_df['Answer'] = answers
        submission_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        # 최종 결과 출력
        self._print_final_results(output_file)
        
        return self._get_results_summary()
    
    def _print_interim_stats(self):
        """중간 통계 출력"""
        total = self.stats["total"]
        if total == 0:
            return
        
        model_success_rate = (self.stats["model_success"] / total) * 100
        avg_time = sum(self.stats["processing_times"]) / len(self.stats["processing_times"])
        
        print(f"  중간 통계: 모델성공률 {model_success_rate:.1f}%, 평균처리시간 {avg_time:.2f}초")
        
        # 모델 답변 분포
        mc_stats = self.model_handler.get_answer_stats()
        if mc_stats["total_mc"] > 0:
            dist = [f"{k}:{v}" for k, v in mc_stats["distribution"].items() if v > 0]
            print(f"  객관식 분포: {', '.join(dist)}")
    
    def _print_final_results(self, output_file: str):
        """최종 결과 출력"""
        total_time = time.time() - self.start_time
        
        print("\n" + "=" * 50)
        print("AI 추론 완료")
        print("=" * 50)
        
        print(f"총 처리시간: {total_time:.1f}초 ({total_time/60:.1f}분)")
        print(f"총 문항수: {self.stats['total']}개")
        print(f"객관식: {self.stats['mc_count']}개")
        print(f"주관식: {self.stats['subj_count']}개")
        
        if self.stats["total"] > 0:
            success_rate = (self.stats["model_success"] / self.stats["total"]) * 100
            avg_time = sum(self.stats["processing_times"]) / len(self.stats["processing_times"])
            
            print(f"모델 성공률: {success_rate:.1f}%")
            print(f"평균 처리시간: {avg_time:.2f}초/문항")
        
        # 도메인별 분포
        print(f"\n도메인별 분포:")
        for domain, count in self.stats["domain_stats"].items():
            pct = (count / self.stats["total"]) * 100
            print(f"  {domain}: {count}개 ({pct:.1f}%)")
        
        # 난이도별 분포
        print(f"\n난이도별 분포:")
        for difficulty, count in self.stats["difficulty_stats"].items():
            pct = (count / self.stats["total"]) * 100
            print(f"  {difficulty}: {count}개 ({pct:.1f}%)")
        
        # 객관식 답변 분포
        mc_stats = self.model_handler.get_answer_stats()
        if mc_stats["total_mc"] > 0:
            print(f"\n객관식 답변 분포:")
            for num in range(1, 6):
                count = mc_stats["distribution"][str(num)]
                pct = (count / mc_stats["total_mc"]) * 100
                print(f"  {num}번: {count}개 ({pct:.1f}%)")
            
            # 다양성 평가
            used_answers = len([v for v in mc_stats["distribution"].values() if v > 0])
            diversity = "우수" if used_answers >= 4 else "양호" if used_answers >= 3 else "개선필요"
            print(f"  답변 다양성: {diversity} ({used_answers}/5개 번호 사용)")
        
        print(f"\n결과 파일: {output_file}")
    
    def _get_results_summary(self) -> Dict:
        """결과 요약"""
        total = max(self.stats["total"], 1)
        mc_stats = self.model_handler.get_answer_stats()
        
        return {
            "success": True,
            "total_questions": self.stats["total"],
            "mc_count": self.stats["mc_count"],
            "subj_count": self.stats["subj_count"],
            "model_success_rate": (self.stats["model_success"] / total) * 100,
            "avg_processing_time": sum(self.stats["processing_times"]) / len(self.stats["processing_times"]) if self.stats["processing_times"] else 0,
            "domain_stats": dict(self.stats["domain_stats"]),
            "difficulty_stats": dict(self.stats["difficulty_stats"]),
            "answer_distribution": mc_stats["distribution"],
            "total_time": time.time() - self.start_time
        }
    
    def cleanup(self):
        """리소스 정리"""
        try:
            print("\n시스템 정리 중...")
            
            if hasattr(self, 'model_handler'):
                self.model_handler.cleanup()
            
            if hasattr(self, 'data_processor'):
                self.data_processor.cleanup()
            
            if hasattr(self, 'knowledge_base'):
                self.knowledge_base.cleanup()
            
            gc.collect()
            print("시스템 정리 완료")
            
        except Exception as e:
            print(f"정리 중 오류: {e}")


def main():
    """메인 함수"""
    print("금융보안 AI 시스템")
    print("=" * 50)
    
    engine = None
    try:
        # AI 엔진 초기화
        engine = FinancialAIInference(verbose=True)
        
        # 추론 실행
        results = engine.execute_inference()
        
        if results["success"]:
            print(f"\n완료됨!")
            print(f"모델 성공률: {results['model_success_rate']:.1f}%")
            print(f"총 처리시간: {results['total_time']:.1f}초")
        
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