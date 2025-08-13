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
- 통합된 추론 파이프라인
- 해당 파일은 inference.py의 50문항 테스트를 실행하는 스크립트입니다.
"""

import os
import sys
import time
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple

current_dir = Path(__file__).parent.absolute()
sys.path.append(str(current_dir))

from inference import FinancialAIInference

DEFAULT_TEST_SIZE = 50
MAX_TEST_SIZE = 500
MIN_TEST_SIZE = 1

class TestRunner:
    
    def __init__(self, test_size: int = DEFAULT_TEST_SIZE, use_finetuned: bool = False):
        """테스트 실행기 초기화"""
        self.test_size = max(MIN_TEST_SIZE, min(test_size, MAX_TEST_SIZE))
        self.use_finetuned = use_finetuned
        self.start_time = time.time()
        
        print(f"테스트 실행기 초기화 중... (대상: {self.test_size}문항)")
        
        # 파인튜닝 모델 경로 확인
        if use_finetuned and not os.path.exists("./finetuned_model"):
            print("파인튜닝 모델을 찾을 수 없습니다. 기본 모델을 사용합니다.")
            self.use_finetuned = False
        
        # inference.py의 FinancialAIInference 사용
        try:
            self.inference_engine = FinancialAIInference(
                enable_learning=True,
                verbose=False,
                use_finetuned=self.use_finetuned
            )
            print("추론 엔진 초기화 완료")
        except Exception as e:
            raise RuntimeError(f"추론 엔진 초기화 실패: {e}")
        
        model_type = "파인튜닝된 모델" if self.use_finetuned else "기본 모델"
        print(f"초기화 완료 - {model_type} 사용\n")
    
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
                structure = self.inference_engine.data_processor.analyze_question_structure(question)
                
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
    
    def run_test(self, test_file: str = "./test.csv", submission_file: str = "./sample_submission.csv") -> None:
        """테스트 실행"""
        print("="*50)
        print(f"테스트 실행 시작 ({self.test_size}문항)")
        if self.use_finetuned:
            print("파인튜닝된 모델 사용")
        print("="*50)
        
        # 데이터 로드
        data_result = self.load_test_data(test_file, submission_file)
        if data_result is None:
            return
        
        test_df, submission_df = data_result
        
        # 질문 분석
        question_analysis = self.analyze_questions(test_df)
        
        print(f"\n추론 시작...")
        
        # 답변 생성 - inference.py의 기능 활용
        answers = []
        
        try:
            for idx, row in test_df.iterrows():
                question = row['Question']
                question_id = row['ID']
                
                # inference.py의 process_question 메서드 사용
                answer = self.inference_engine.process_question(question, question_id, idx)
                answers.append(answer)
                
                # 진행 상황 출력
                if (idx + 1) % 10 == 0:
                    progress = (idx + 1) / self.test_size * 100
                    print(f"  진행: {idx + 1}/{self.test_size} ({progress:.0f}%)")
            
            # 결과 저장
            submission_df['Answer'] = answers
            
            output_file = f"./test_result_{self.test_size}.csv"
            submission_df.to_csv(output_file, index=False, encoding='utf-8-sig')
            
            # 결과 출력
            self._print_test_results(output_file, question_analysis)
            
        except KeyboardInterrupt:
            print("\n테스트 중단됨")
        except Exception as e:
            print(f"테스트 실행 중 오류: {e}")
            raise
    
    def _print_test_results(self, output_file: str, question_analysis: Dict) -> None:
        """테스트 결과 출력"""
        total_time = time.time() - self.start_time
        
        print(f"\n" + "="*50)
        print("테스트 완료")
        print("="*50)
        
        # 처리 시간 정보
        print(f"처리 시간: {total_time:.1f}초")
        print(f"문항당 평균: {total_time/self.test_size:.2f}초")
        
        # 모델 정보
        model_type = "파인튜닝된 모델" if self.use_finetuned else "기본 모델"
        print(f"사용 모델: {model_type}")
        
        # inference.py의 통계 정보 활용
        stats = self.inference_engine.stats
        
        # 처리 통계
        print(f"\n처리 통계:")
        success_rate = stats["model_generation_success"] / max(stats["total"], 1) * 100
        pattern_rate = stats["pattern_based_answers"] / max(stats["total"], 1) * 100
        fallback_rate = stats["fallback_used"] / max(stats["total"], 1) * 100
        
        print(f"  모델 생성 성공: {stats['model_generation_success']}/{stats['total']} ({success_rate:.1f}%)")
        print(f"  패턴 기반 답변: {stats['pattern_based_answers']}회 ({pattern_rate:.1f}%)")
        print(f"  스마트 힌트 사용: {stats['smart_hints_used']}회")
        print(f"  고신뢰도 답변: {stats['high_confidence_answers']}회")
        print(f"  폴백 사용: {stats['fallback_used']}/{stats['total']} ({fallback_rate:.1f}%)")
        print(f"  처리 오류: {stats['errors']}회")
        
        # 추론 엔진 통계
        if self.inference_engine.reasoning_engine:
            print(f"\n추론 엔진 성능:")
            reasoning_rate = stats["reasoning_engine_usage"] / max(stats["total"], 1) * 100
            cot_rate = stats["cot_prompts_used"] / max(stats["total"], 1) * 100
            
            print(f"  추론 엔진 사용: {stats['reasoning_engine_usage']}/{stats['total']} ({reasoning_rate:.1f}%)")
            print(f"  CoT 프롬프트 사용: {stats['cot_prompts_used']}/{stats['total']} ({cot_rate:.1f}%)")
            print(f"  추론 성공: {stats['reasoning_successful']}회")
            print(f"  추론 실패: {stats['reasoning_failed']}회")
            print(f"  검증 통과: {stats['verification_passed']}회")
            print(f"  검증 실패: {stats['verification_failed']}회")
        
        # 파인튜닝 통계
        if self.use_finetuned:
            finetuned_rate = stats["finetuned_usage"] / max(stats["total"], 1) * 100
            print(f"\n파인튜닝 모델 사용률: {finetuned_rate:.1f}%")
        
        # 한국어 품질 통계
        if stats["quality_scores"]:
            import numpy as np
            avg_quality = np.mean(stats["quality_scores"])
            print(f"\n한국어 품질:")
            print(f"  평균 품질 점수: {avg_quality:.2f}")
            
            if avg_quality > 0.8:
                quality_level = "우수"
            elif avg_quality > 0.65:
                quality_level = "양호"
            else:
                quality_level = "개선 필요"
            
            print(f"  품질 평가: {quality_level}")
        
        # 객관식 분포
        distribution = stats["answer_distribution"]
        total_mc = sum(distribution.values())
        if total_mc > 0:
            print(f"\n객관식 답변 분포:")
            for ans in sorted(distribution.keys()):
                count = distribution[ans]
                if count > 0:
                    pct = count / total_mc * 100
                    print(f"  {ans}번: {count}개 ({pct:.1f}%)")
            
            # 다양성 평가
            unique_answers = len([k for k, v in distribution.items() if v > 0])
            print(f"  답변 다양성: {unique_answers}/5개 번호 사용")
        
        # 학습 통계
        if self.inference_engine.enable_learning:
            print(f"\n학습 통계:")
            print(f"  학습된 샘플: {stats['learned']}개")
            accuracy = self.inference_engine.learning_system.get_current_accuracy()
            print(f"  현재 정확도: {accuracy:.2%}")
        
        print(f"\n결과 파일: {output_file}")
    
    def get_quick_performance_summary(self) -> Dict:
        """빠른 성능 요약"""
        stats = self.inference_engine.stats
        
        if stats["total"] == 0:
            return {"error": "아직 테스트가 실행되지 않았습니다"}
        
        summary = {
            "총_문항": stats["total"],
            "모델_성공률": f"{stats['model_generation_success']/stats['total']*100:.1f}%",
            "추론_엔진_사용률": f"{stats['reasoning_engine_usage']/stats['total']*100:.1f}%",
            "폴백_사용률": f"{stats['fallback_used']/stats['total']*100:.1f}%",
            "평균_처리시간": f"{sum(stats['processing_times'])/len(stats['processing_times']):.2f}초" if stats['processing_times'] else "N/A"
        }
        
        if self.use_finetuned:
            summary["파인튜닝_사용률"] = f"{stats['finetuned_usage']/stats['total']*100:.1f}%"
        
        if stats["quality_scores"]:
            import numpy as np
            summary["한국어_품질"] = f"{np.mean(stats['quality_scores']):.2f}"
        
        return summary
    
    def cleanup(self) -> None:
        """리소스 정리"""
        try:
            print("\n시스템 정리 중...")
            
            # inference.py의 cleanup 메서드 사용
            if hasattr(self, 'inference_engine'):
                self.inference_engine.cleanup()
            
            print("정리 완료")
            
        except Exception as e:
            print(f"정리 중 오류: {e}")


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
        
        # 빠른 성능 요약 출력
        summary = runner.get_quick_performance_summary()
        print(f"\n성능 요약:")
        for key, value in summary.items():
            print(f"  {key}: {value}")
        
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