# test_runner.py

"""
테스트 실행기
- 시스템 테스트 실행
- 의도 일치 성공률 명확 표시
- 핵심 성능 지표만 출력
"""

import os
import sys
from pathlib import Path

# 현재 디렉토리 설정
current_dir = Path(__file__).parent.absolute()
sys.path.append(str(current_dir))

from inference import FinancialAIInference

def run_test(test_size: int = 50, verbose: bool = True):
    """테스트 실행"""
    
    # 파일 존재 확인
    test_file = "./test.csv"
    submission_file = "./sample_submission.csv"
    
    for file_path in [test_file, submission_file]:
        if not os.path.exists(file_path):
            print(f"오류: {file_path} 파일이 없습니다")
            return False
    
    engine = None
    try:
        # AI 엔진 초기화
        print("\n시스템 초기화 중...")
        engine = FinancialAIInference(verbose=False)  # verbose=False로 설정
        
        # 테스트 데이터 준비
        import pandas as pd
        test_df = pd.read_csv(test_file)
        submission_df = pd.read_csv(submission_file)
        
        # 지정된 크기로 제한
        if len(test_df) > test_size:
            test_df = test_df.head(test_size)
            temp_submission = submission_df.head(test_size).copy()
            
            output_file = "./test_result.csv"
            results = engine.execute_inference_with_data(
                test_df, 
                temp_submission, 
                output_file
            )
        else:
            output_file = "./test_result.csv"
            results = engine.execute_inference(
                test_file,
                submission_file,
                output_file
            )
        
        # 결과 분석 (간소화)
        print_simplified_results(results, output_file, test_size)
        
        return True
        
    except Exception as e:
        print(f"테스트 실행 오류: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        if engine:
            engine.cleanup()

def print_simplified_results(results: dict, output_file: str, test_size: int):
    """핵심 결과만 간단 출력"""

    print("\n=== 테스트 완료 ===")
    
    # 기본 정보
    total_time_minutes = results['total_time'] / 60
    print(f"처리 시간: {total_time_minutes:.1f}분")
    print(f"처리 문항: {results['total_questions']}개")
    
    # 핵심 성능 지표만 표시
    print(f"\n=== 핵심 성능 지표 ===")
    print(f"모델 성공률: {results['model_success_rate']:.1f}%")
    print(f"한국어 준수율: {results['korean_compliance_rate']:.1f}%")
    
    # 의도 일치 성공률 - 핵심 지표
    intent_success_rate = results.get('intent_match_success_rate', 0)
    print(f"의도 일치 성공률: {intent_success_rate:.1f}%")
    
    # 추가 중요 지표들
    if results.get('choice_range_error_rate', 0) > 0:
        print(f"선택지 범위 오류율: {results['choice_range_error_rate']:.1f}%")
    
    if results.get('validation_error_rate', 0) > 0:
        print(f"검증 오류율: {results['validation_error_rate']:.1f}%")
    
    # 문항 분포 (간단히)
    mc_count = results.get('mc_count', 0)
    subj_count = results.get('subj_count', 0)
    print(f"\n문항 분포: 객관식 {mc_count}개, 주관식 {subj_count}개")
    
    # 의도별 성과 (주관식이 있을 때만)
    if subj_count > 0:
        intent_quality = results.get('intent_quality_by_type', {})
        if intent_quality:
            print(f"\n=== 의도별 성과 ===")
            for intent, quality in intent_quality.items():
                print(f"{intent}: 품질 {quality:.2f}")
    
    # 성능 판정
    print(f"\n=== 성능 평가 ===")
    
    # 의도 일치 성공률 기준 평가
    if intent_success_rate >= 80:
        intent_status = "우수"
    elif intent_success_rate >= 60:
        intent_status = "양호"
    elif intent_success_rate >= 40:
        intent_status = "보통"
    else:
        intent_status = "개선 필요"
    
    print(f"의도 일치 성능: {intent_status}")
    
    # 전체 성능 예측
    predicted_score = estimate_performance_score(results)
    print(f"예상 점수: {predicted_score:.3f}")
    
    if predicted_score >= 0.65:
        print("✅ 목표 점수 달성 가능")
    else:
        print("❌ 목표 점수 미달 - 개선 필요")
        suggest_improvements(results)

def estimate_performance_score(results: dict) -> float:
    """성능 점수 예측"""
    # 간단한 점수 예측 공식
    model_success = results.get('model_success_rate', 0) / 100
    korean_compliance = results.get('korean_compliance_rate', 0) / 100
    intent_success = results.get('intent_match_success_rate', 0) / 100
    
    # 가중 평균으로 점수 계산
    predicted_score = (model_success * 0.3 + 
                      korean_compliance * 0.3 + 
                      intent_success * 0.4)
    
    return predicted_score

def suggest_improvements(results: dict):
    """개선 제안"""
    print(f"\n=== 개선 제안 ===")
    
    intent_rate = results.get('intent_match_success_rate', 0)
    if intent_rate < 60:
        print("- 의도 일치 성공률이 낮습니다. 질문 의도 분석 정확도를 높여야 합니다.")
    
    model_rate = results.get('model_success_rate', 0)
    if model_rate < 90:
        print("- 모델 성공률이 낮습니다. 프롬프트 최적화가 필요합니다.")
    
    validation_error = results.get('validation_error_rate', 0)
    if validation_error > 5:
        print("- 검증 오류가 많습니다. 답변 검증 로직을 개선해야 합니다.")
    
    choice_error = results.get('choice_range_error_rate', 0)
    if choice_error > 2:
        print("- 선택지 범위 오류가 있습니다. 객관식 답변 처리를 개선해야 합니다.")

def select_test_size():
    """테스트 문항 수 선택"""
    print("\n테스트할 문항 수를 선택하세요:")
    print("1. 5문항 (빠른 테스트)")
    print("2. 10문항 (기본 테스트)")
    print("3. 50문항 (정밀 테스트)")
    print("4. 100문항 (성능 테스트)")
    print()
    
    while True:
        try:
            choice = input("선택 (1-4): ").strip()
            
            if choice == "1":
                return 5
            elif choice == "2":
                return 10
            elif choice == "3":
                return 50
            elif choice == "4":
                return 100
            else:
                print("잘못된 선택입니다. 1, 2, 3, 4 중 하나를 입력하세요.")
                
        except KeyboardInterrupt:
            print("\n프로그램을 종료합니다.")
            sys.exit(0)
        except Exception:
            print("잘못된 입력입니다. 다시 시도하세요.")

def main():
    """메인 함수"""
    # 테스트 크기 선택
    test_size = select_test_size()
    
    print(f"\n선택된 테스트: {test_size}문항")
    print("의도 일치 성공률 분석을 시작합니다...")
    
    success = run_test(test_size, verbose=True)
    
    if success:
        print(f"\n테스트 완료 - 결과 파일: ./test_result.csv")
    else:
        print("\n테스트 실패")
        sys.exit(1)

if __name__ == "__main__":
    main()